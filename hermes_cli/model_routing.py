# -*- coding: utf-8 -*-
"""model_routing.py — Hermes 模型路由引擎(A 方案 Phase 1)

定位:请求前选择模型的决策层。纯逻辑,不依赖 Hermes 运行时代码,
    便于独立单测与后续合并进 hermes_cli/。

设计原则(SPEC §0):
1. 谓词规则(AnythingLLM 概念):路由决策 = 可评估的规则列表
2. 保守默认:未命中任何规则 → default(不升级不降级)
3. 可观测:每次决策记日志(命中规则/原因/选择),便于调规则
4. 与 fallback 正交:routing 选"用谁",fallback 管"挂了换谁"

规则引擎:
- rules 按顺序评估,第一条命中生效
- heuristic 匹配:长度/词数/代码块/URL/关键词/领域
- 支持 AND(同规则内多条件)/OR(OR_ 前缀字段)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("model_routing")

# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class RouteDecision:
    """一次路由决策的结果。"""
    matched_rule: Optional[str]      # 命中的规则名(None = 未命中,用 default)
    provider: str                    # 选中的 provider
    model: str                       # 选中的模型
    reason: str                      # 命中原因(可解释)
    evaluated: List[str] = field(default_factory=list)  # 评估过的规则名

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched_rule": self.matched_rule,
            "provider": self.provider,
            "model": self.model,
            "reason": self.reason,
            "evaluated": self.evaluated,
        }


@dataclass
class MessageFeatures:
    """从用户消息提取的启发式特征(供规则匹配)。"""
    text: str
    char_count: int
    word_count: int
    has_code_block: bool
    has_url: bool
    has_single_backtick: bool
    keywords_hit: List[str] = field(default_factory=list)

    @classmethod
    def extract(cls, text: str) -> "MessageFeatures":
        text = text or ""
        has_code = "```" in text
        has_url = bool(re.search(r"https?://\S+", text))
        has_tick = "`" in text
        words = text.split()
        return cls(
            text=text,
            char_count=len(text),
            word_count=len(words),
            has_code_block=has_code,
            has_url=has_url,
            has_single_backtick=has_tick,
        )


# ---------------------------------------------------------------------------
# 路由引擎
# ---------------------------------------------------------------------------

class ModelRouter:
    """模型路由引擎:加载配置 → 评估规则 → 选择模型 → 记录决策。

    用法:
        router = ModelRouter(cfg_dict)
        decision = router.route("帮我把这句话翻译成英文")
        # decision.provider == "nvidia", decision.model == "z-ai/glm-5.2"
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.enabled = bool(config.get("enabled", False))
        self.default = config.get("default", {}) or {}
        self.rules: List[Dict[str, Any]] = config.get("rules", []) or []
        self.capabilities: Dict[str, Any] = config.get("capabilities", {}) or {}
        self.log_level = config.get("log_level", "info")
        self.stats_file = config.get("stats_file")
        self._log = logger or log
        # 增强层(routing_enhance.RoutingEnhancer), 默认 None = 完全向后兼容
        self.enhancer = None

        # 校验
        if self.enabled and not self.rules and not self.capabilities:
            self._log.warning("model_routing enabled 但无 rules/capabilities,所有请求走 default")

    # -- 入口 ------------------------------------------------------------

    def route(self, message: str, *, capabilities: Optional[List[str]] = None,
              session_id: str = "", has_images: bool = False,
              is_tool_result: bool = False, is_subagent: bool = False,
              current_model: str = "") -> RouteDecision:
        """对一条用户消息做路由决策。

        增强层(routing_enhance): 若配置了 enhancer, 在规则评估前做 gate:
          - 确认轮/工具结果/探针 → 沿用当前模型(不路由, 治误判根因)
          - 含图/复杂度信号 → force_main(不降级到简单模型)
        未配置 enhancer 时完全向后兼容(原 heuristic 行为不变)。
        """
        if not self.enabled:
            return self._default_decision("routing_disabled", message)

        # 增强层 gate(可选)
        if self.enhancer is not None:
            enh = self.enhancer.check(
                message, session_id=session_id, has_images=has_images,
                is_tool_result=is_tool_result, is_subagent=is_subagent,
                current_model=current_model)
            if enh.should_skip:
                # 沿用当前模型(若有), 否则保守 default
                if current_model:
                    return RouteDecision(
                        matched_rule=None, provider=self.default.get("provider", ""),
                        model=current_model, reason=f"增强层跳过路由: {enh.skip_reason}",
                        evaluated=["routing_enhance:skip"])
                return self._default_decision(f"enh skip: {enh.skip_reason}", message)
            if enh.force_main:
                # 含图/复杂度 → 强制主模型, 不降级
                self._log.info("增强层 force_main: %s", enh.force_reason)
                return self._default_decision(f"enh main: {enh.force_reason}", message,
                                              evaluated=["routing_enhance:force_main"])

        feats = MessageFeatures.extract(message)
        evaluated: List[str] = []

        # 优先级 1:领域能力匹配(显式指定)
        if capabilities:
            for cap in capabilities:
                if cap in self.capabilities:
                    evaluated.append(f"capability:{cap}")
                    return self._build_decision(
                        f"capability:{cap}", self.capabilities[cap],
                        f"领域能力 '{cap}' 命中", evaluated)

        # 优先级 2:复杂度规则(顺序评估,第一条命中)
        for rule in self.rules:
            name = rule.get("name", "<unnamed>")
            evaluated.append(name)
            match = rule.get("match", {})
            if self._evaluate_rule(match, feats):
                route = rule.get("route", {})
                return self._build_decision(
                    name, route, self._reason_for(rule, feats), evaluated)

        # 优先级 3:default(保守,不升级不降级)
        return self._default_decision("no_rule_matched", message, evaluated=evaluated)

    # -- 规则评估 ----------------------------------------------------------

    def _evaluate_rule(self, match: Dict[str, Any], feats: MessageFeatures) -> bool:
        """评估一条规则的 match 条件(heuristic)。"""
        kind = match.get("kind", "heuristic")
        if kind != "heuristic":
            # llm_classify / context_aware 是 Phase 2 扩展,当前视为不命中
            return False

        # AND 条件(全部满足)
        checks = []
        if "max_chars" in match:
            checks.append(feats.char_count <= match["max_chars"])
        if "max_words" in match:
            checks.append(feats.word_count <= match["max_words"])
        if "min_chars" in match:
            checks.append(feats.char_count >= match["min_chars"])
        if "no_code" in match and match["no_code"]:
            checks.append(not feats.has_code_block)
        if "has_code_block" in match and match["has_code_block"]:
            checks.append(feats.has_code_block)
        if "no_url" in match and match["no_url"]:
            checks.append(not feats.has_url)
        if "no_keywords" in match:
            hits = [k for k in match["no_keywords"] if k.lower() in feats.text.lower()]
            checks.append(not hits)
        if "has_keywords" in match:
            hits = [k for k in match["has_keywords"] if k.lower() in feats.text.lower()]
            checks.append(bool(hits))
        if checks and not all(checks):
            return False

        # OR 条件(任一满足;OR_ 前缀字段)
        or_checks = []
        for key, val in match.items():
            if key.startswith("OR_"):
                if key == "OR_has_keywords":
                    hits = [k for k in val if k.lower() in feats.text.lower()]
                    or_checks.append(bool(hits))
                elif key == "OR_has_code_block":
                    or_checks.append(feats.has_code_block)
                elif key == "OR_min_chars":
                    or_checks.append(feats.char_count >= val)
        if or_checks:
            return any(or_checks)
        return True  # 只有 AND 条件且全过

    # -- 辅助 ------------------------------------------------------------

    def _build_decision(self, rule_name: str, route: Dict[str, Any],
                        reason: str, evaluated: List[str]) -> RouteDecision:
        provider = route.get("provider", self.default.get("provider", ""))
        model = route.get("model", self.default.get("model", ""))
        d = RouteDecision(rule_name, provider, model, reason, evaluated)
        self._log_decision(d, feats_text="")
        return d

    def _default_decision(self, why: str, message: str,
                          evaluated: Optional[List[str]] = None) -> RouteDecision:
        provider = self.default.get("provider", "")
        model = self.default.get("model", "")
        reasons = {
            "routing_disabled": "model_routing 未启用,使用 default",
            "no_rule_matched": "未命中任何规则,保守使用 default",
        }
        d = RouteDecision(None, provider, model, reasons.get(why, why), evaluated or [])
        self._log_decision(d, feats_text=message[:80])
        return d

    def _reason_for(self, rule: Dict[str, Any], feats: MessageFeatures) -> str:
        match = rule.get("match", {})
        parts = []
        if "max_chars" in match and feats.char_count <= match["max_chars"]:
            parts.append(f"长度{feats.char_count}<= {match['max_chars']}")
        if "max_words" in match and feats.word_count <= match["max_words"]:
            parts.append(f"词数{feats.word_count}<= {match['max_words']}")
        if "has_code_block" in match and match["has_code_block"] and feats.has_code_block:
            parts.append("含代码块")
        if "no_code" in match and match["no_code"] and not feats.has_code_block:
            parts.append("无代码")
        if "OR_has_keywords" in match:
            hits = [k for k in match["OR_has_keywords"] if k.lower() in feats.text.lower()]
            if hits:
                parts.append(f"关键词:{'/'.join(hits[:3])}")
        return "; ".join(parts) or "规则条件满足"

    def _log_decision(self, d: RouteDecision, feats_text: str) -> None:
        """记录决策(内存 log + 可选 stats 文件)。"""
        entry = {
            "ts": None,  # 调用方注入时间戳(保持纯逻辑无依赖)
            "rule": d.matched_rule,
            "provider": d.provider,
            "model": d.model,
            "reason": d.reason,
        }
        if self.log_level in ("info", "debug"):
            self._log.info("route → %s/%s (rule=%s, %s)",
                           d.provider, d.model, d.matched_rule, d.reason)
        if self.log_level == "debug":
            self._log.debug("evaluated rules: %s", d.evaluated)
        if self.stats_file and d.matched_rule:
            try:
                Path(self.stats_file).parent.mkdir(parents=True, exist_ok=True)
                with open(self.stats_file, "a") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except OSError as e:
                self._log.warning("stats 文件写入失败: %s", e)


# ---------------------------------------------------------------------------
# 配置加载(与 config.yaml 兼容)
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    """从 YAML 文件加载 model_routing 配置。"""
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("model_routing", {}) if isinstance(data, dict) else {}
