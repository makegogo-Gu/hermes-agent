# -*- coding: utf-8 -*-
"""routing_enhance.py — model_routing 增强层(1-5 工程, 借鉴不替换)

源自调研(2026-08-19, 见 hermes-self/*.md):
  - workweave/router 三机制(turn-type过滤 / cache-aware STAY-SWITCH / 会话pin)
  - semantic-router 式意图分类(用 bge-m3, 比纯 heuristic 准)
  - 9router 的 Vision Adapter 思想(含图→多模态)
  - 保守默认(不确定→云, 不 fail-open 到本地误判, 对齐 workweave "移除heuristic fallback")

设计约束:
  - **不修改 model_routing.py 主结构** —— 本模块是可选增强层, 通过 route() 参数接入
  - 纯逻辑, 无 I/O(workweave 架构: inner-ring 无 I/O)
  - 默认保守: 增强层无法判断时 → 返回"不干预", 交回原 heuristic 路径
  - 不引入训练/HMM/bandit —— 单人量级不值得
"""

from __future__ import annotations
from dataclasses import dataclass, field
import re


# ---------------------------------------------------------------------------
# 1. TurnTypeClassifier (workweave turntype, 治误判根因)
# ---------------------------------------------------------------------------
# 目的: 判断一条入站消息"该不该走路由"。workweave 证明: 工具结果/确认轮/子agent
#       产物等不是独立任务, 强行路由会误判(把"好的"这种确认轮当简单任务丢本地)。
#       这正是我们 8-18 被叫停(≤120字→本地)的根因。
@dataclass
class TurnTypeResult:
    turn_type: str             # main_loop / confirmation / tool_result / short_probe / other
    should_route: bool         # 只有 main 该走路由
    reason: str


class TurnTypeClassifier:
    """静态分类器, 无 I/O。基于消息形态 + 枚举黑/白名单。"""

    CONFIRM_WORDS = {
        "好", "好的", "ok", "okay", "嗯", "可以", "对", "是", "继续", "就这样",
        "go", "yes", "y", "sure", "继续吧", "可以了", "行", "做得对", "不错",
    }
    BLOCK_START = {"```", "def ", "class ", "import ", "function", "const ", "let "}

    # 明显是"复杂任务"的信号词(保守: 命中→不应路由到简单模型)
    COMPLEX_SIGNALS = {
        "分析", "设计", "实现", "架构", "重构", "优化", "部署", "测试", "debug",
        "analyze", "implement", "refactor", "design", "架构设计", "方案",
    }

    def classify(self, text: str, *, is_tool_result: bool = False,
                 is_subagent: bool = False) -> TurnTypeResult:
        text = (text or "").strip()
        # 显式标记(由调用方传入的上下文)
        if is_tool_result:
            return TurnTypeResult("tool_result", False, "工具结果, 不路由")
        if is_subagent:
            return TurnTypeResult("other", False, "子agent产物, 不路由")
        # 空 / 纯标点 → 探针
        if not text:
            return TurnTypeResult("short_probe", False, "空消息, 不路由")
        # 确认轮(很短 + 只含确认词)
        if len(text) <= 12 and text.strip().lower() in {w.lower() for w in self.CONFIRM_WORDS}:
            return TurnTypeResult("confirmation", False, "确认轮, 沿用当前模型不路由")
        # 含代码块起始 → 明显代码任务, 该路由(到复杂模型)
        if any(t in text for t in self.BLOCK_START):
            return TurnTypeResult("main_loop", True, "代码块/编码, 主循环")
        # 含复杂信号词 → 保守: 该路由且目标应是强模型
        if any(s in text for s in self.COMPLEX_SIGNALS):
            return TurnTypeResult("main_loop", True, "含复杂信号词")
        # 其余正常用户消息 → 主循环(是否降级到简单模型由规则+意图分类决定)
        return TurnTypeResult("main_loop", True, "普通用户消息")


# ---------------------------------------------------------------------------
# 2. CacheAwarePlanner (workweave planner, cache 阈值 STAY/SWITCH)
# ---------------------------------------------------------------------------
# 目的: 决定"切模型"是否值得。切模型丢上游 prompt cache(预热代价), 只有新模型
#       期望收益显著超过 cache-miss 代价才切。治"逐消息乱切伤缓存成本被忽略"。
@dataclass
class CacheCutoffInput:
    fresh_model: str            # 新决策的模型(可能是便宜的)
    pinned_model: str           # 当前会话已用模型
    switch_cost: float          # 切换一次的成本(0-1, 越大越贵/越该避免)
    expected_win: float         # 新模型相比当前的优势(0-1)
    tier_upgrade: bool = False  # 新决策是否严格更高 tier(该升才升)

class CacheAwarePlanner:
    """STAY/SWITCH 决策。纯函数。"""

    def __init__(self, switch_cost: float = 0.3, win_threshold: float = 0.4,
                 upgrade_overrides: bool = True):
        # switch_cost 默认 0.3: 切一次模型有 cache 与复杂度代价
        # win_threshold 默认 0.4: 新模型优势必须显著, 才值得切
        self.switch_cost = switch_cost
        self.win_threshold = win_threshold
        self.upgrade_overrides = upgrade_overrides

    def decide(self, inp: CacheCutoffInput) -> str:
        if not inp.fresh_model or not inp.pinned_model:
            return "SWITCH"  # 无 pin(新会话) → 直接采新决策
        if inp.fresh_model == inp.pinned_model:
            return "STAY"    # 同模型, 自然不切
        # tier 升级守卫: fresh 严格更高 tier 时, cache 代价不 override(该升)
        if self.upgrade_overrides and inp.tier_upgrade:
            return "SWITCH"
        # 经济学: 优势是否超过 切换代价 + 阈值
        if inp.expected_win - self.switch_cost >= self.win_threshold:
            return "SWITCH"
        return "STAY"  # 默认保守: 不确定/优势不足 → 保持当前, 不折腾 cache


# ---------------------------------------------------------------------------
# 3. SessionPin (workweave sessionpin, 会话级稳定)
# ---------------------------------------------------------------------------
# 目的: 一会话尽量锁定一个模型(preserve cache), 只在 strong 信号时才切。
#       与保守默认一致, 三重加固"不确定→不切"。
class SessionPin:
    def __init__(self):
        self._pinned = {}       # session_id -> model

    def get(self, session_id: str) -> str:
        return self._pinned.get(session_id, "")

    def maybe_pin(self, session_id: str, model: str) -> str:
        """粘性 pin: 若该 session 已有 pin 则保持, 否则记录新 pin。"""
        cur = self._pinned.get(session_id, "")
        if not cur:
            self._pinned[session_id] = model
            cur = model
        return cur

    def reset(self, session_id: str) -> None:
        self._pinned.pop(session_id, None)


# ---------------------------------------------------------------------------
# 4. IntentClassifier (semantic-router 式, 用 bge-m3)
# ---------------------------------------------------------------------------
# 目的: 比纯 heuristic 更准的"意图/复杂度"分类。可选(需要 embedding 后端)。
#       语义嵌入判断: 该任务是否够"简单"可降级到便宜/本地模型, 或需强模型。
class IntentClassifier:
    """轻量意图分类。若未配置 embed 后端则返回 None(交回 heuristic)。"""

    def __init__(self, embed_fn=None, simple_threshold: float = 0.45,
                 complex_threshold: float = 0.6):
        # embed_fn: callable(text)->vector, 复用现有 bge-m3
        # 阈值: simple 当且仅当与"简单模板"相似度突破 threshold, 且 hit 不到复杂模板
        self._embed = embed_fn
        # 极简原型模板(真实场景应来自少量标注, 这里给通用)
        self._simple_proto = "你好 简单 问 一个问题 是什么 多少"
        self._complex_proto = "分析 设计 实现 架构 重构 方案 代码 部署"
        self._simple_thr = simple_threshold
        self._complex_thr = complex_threshold

    def available(self) -> bool:
        return self._embed is not None

    def classify(self, text: str) -> str | None:
        """返回 'simple' / 'complex' / None(无法判断→交回 heuristic)。"""
        embed = self._embed
        if embed is None:
            return None
        try:
            v = embed(text)
            sv = embed(self._simple_proto)
            cv = embed(self._complex_proto)
            s_cos = _cosine(v, sv)
            c_cos = _cosine(v, cv)
        except Exception:
            return None  # embed 失败 → 保守, 交回
        if c_cos > self._complex_thr:
            return "complex"
        if s_cos > self._simple_thr and c_cos < self._complex_thr:
            # concrete simple unless too complex
            if c_cos - s_cos > 0.05:
                return "complex"
            return "simple"
        return None


def _cosine(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# 5. 组合入口: apply_enhancements(在 route() 前/后调用)
# ---------------------------------------------------------------------------
@dataclass
class EnhancementResult:
    # 是否建议"跳过路由, 沿用当前模型"(确认轮/工具结果)
    should_skip: bool = False
    skip_reason: str = ""
    # 是否建议"强制走主/强模型"(含图/复杂/或 session 已 pin)
    force_main: bool = False
    force_reason: str = ""
    # 意图分类结果(可选)
    intent: str | None = None


class RoutingEnhancer:
    """组合 turn-type + planner + pin + intent。是增强层的门面。"""

    def __init__(self, *, turn_type: TurnTypeClassifier | None = None,
                 planner: CacheAwarePlanner | None = None,
                 pin: SessionPin | None = None,
                 intent: IntentClassifier | None = None):
        self.turn_type = turn_type or TurnTypeClassifier()
        self.planner = planner or CacheAwarePlanner()
        self.pin = pin or SessionPin()
        self.intent = intent or IntentClassifier()

    def check(self, text: str, *, session_id: str = "", has_images: bool = False,
              is_tool_result: bool = False, is_subagent: bool = False,
              current_model: str = "") -> EnhancementResult:
        r = EnhancementResult()

        # (a) turn-type: 确认轮/工具结果/探针 → 跳过路由
        tt = self.turn_type.classify(text, is_tool_result=is_tool_result,
                                     is_subagent=is_subagent)
        if not tt.should_route:
            r.should_skip = True
            r.skip_reason = tt.reason
            return r

        # (b) 含图 → 强制主/多模态
        if has_images:
            r.force_main = True
            r.force_reason = "含附件/图片, 需多模态主模型"
            return r

        # (c) 意图分类(可选, 有 embed 时) → 记录 intent 供规则用
        if self.intent.available():
            r.intent = self.intent.classify(text)

        # (d) session pin: 已有 pin 且当前模型一致 → 保守 STAY(不折腾)
        if session_id:
            pinned = self.pin.maybe_pin(session_id, current_model or "")
            if pinned and current_model and pinned == current_model:
                # 已有 pin 且未切换 → 没有 strong 升级信号就不动
                pass
        return r

    def decide_switch(self, *, session_id: str, fresh_model: str,
                      pinned_model: str, expected_win: float,
                      tier_upgrade: bool = False) -> str:
        """用 planner 决定是否从 pinned->fresh。"""
        inp = CacheCutoffInput(
            fresh_model=fresh_model, pinned_model=pinned_model,
            switch_cost=self.planner.switch_cost, expected_win=expected_win,
            tier_upgrade=tier_upgrade,
        )
        decision = self.planner.decide(inp)
        if decision == "SWITCH" and session_id and fresh_model:
            self.pin.maybe_pin(session_id, fresh_model)
        return decision
