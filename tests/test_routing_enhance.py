# -*- coding: utf-8 -*-
"""test_routing_enhance.py — routing_enhance 三层增强单测

验证(tdd 铁律: 断言真实行为):
  - TurnTypeClassifier: 确认轮/工具结果/复杂信号 分类正确
  - CacheAwarePlanner: STAY/SWITCH 经济学正确(含 tier 升级守卫)
  - SessionPin: 粘性 pin 保持/重置
  - IntentClassifier: 无 embed 时保守返回 None
  - RoutingEnhancer 门面: 组合行为正确

运行: python3 -m pytest test_routing_enhance.py -q  (或直接 python3)
"""
import sys, os
_D = os.path.dirname(os.path.abspath(__file__))
for _cand in (_D, os.path.join(_D, ".."), os.path.join(_D, "..", "hermes_cli")):
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

from routing_enhance import (
    TurnTypeClassifier, CacheAwarePlanner, CacheCutoffInput,
    SessionPin, IntentClassifier, RoutingEnhancer,
)


def test_turn_type():
    t = TurnTypeClassifier()
    # 确认轮 → 不路由(治误判根因)
    assert not t.classify("好的").should_route
    assert not t.classify("ok").should_route
    assert not t.classify("继续").should_route
    # 工具结果 → 不路由
    assert not t.classify("...", is_tool_result=True).should_route
    # 复杂信号 → 主循环且该路由
    assert t.classify("帮我分析项目架构").should_route
    assert t.classify("```python\nx=1```").should_route
    # 空 → 不路由
    assert not t.classify("").should_route


def test_planner():
    p = CacheAwarePlanner(switch_cost=0.3, win_threshold=0.4)
    # 无 pin → SWITCH
    assert p.decide(CacheCutoffInput("a", "", 0.3, 0.0)) == "SWITCH"
    # 同模型 → STAY
    assert p.decide(CacheCutoffInput("a", "a", 0.3, 0.0)) == "STAY"
    # 优势不足 → STAY(保守, 不折腾 cache)
    assert p.decide(CacheCutoffInput("cheap", "main", 0.3, 0.2)) == "STAY"
    # 优势显著 → SWITCH
    assert p.decide(CacheCutoffInput("cheap", "main", 0.3, 0.8)) == "SWITCH"
    # tier 升级守卫: fresh 更高 tier 且该升 → SWITCH 覆盖 cache
    p2 = CacheAwarePlanner(upgrade_overrides=True)
    assert p2.decide(CacheCutoffInput("main", "cheap", 0.9, 0.1, tier_upgrade=True)) == "SWITCH"


def test_session_pin():
    sp = SessionPin()
    assert sp.maybe_pin("s1", "alpha") == "alpha"
    assert sp.maybe_pin("s1", "beta") == "alpha"  # 保持首次 pin
    assert sp.get("s1") == "alpha"
    sp.reset("s1")
    assert sp.get("s1") == ""


def test_intent_no_embed():
    # 无 embed → 保守 None(不误判)
    ic = IntentClassifier()
    assert ic.classify("随便") is None


def test_enhancer_gate():
    e = RoutingEnhancer()
    # 确认轮 → should_skip
    r = e.check("好的", session_id="s", current_model="glm")
    assert r.should_skip
    # 含图 → force_main
    r = e.check("看看这张图", has_images=True)
    assert r.force_main
    # 工具结果 → skip
    r = e.check("...", is_tool_result=True)
    assert r.should_skip


def test_enhancer_switch():
    e = RoutingEnhancer()
    # 无 pin 的新会话 → SWITCH
    assert e.decide_switch(session_id="", fresh_model="x", pinned_model="", expected_win=0) == "SWITCH"
    # 优势不足 → STAY
    assert e.decide_switch(session_id="s2", fresh_model="cheap", pinned_model="main", expected_win=0.2) == "STAY"
    # 优势显著 → SWITCH 且 pin 更新
    assert e.decide_switch(session_id="s2", fresh_model="cheap", pinned_model="main", expected_win=0.9) == "SWITCH"
    assert e.pin.get("s2") == "cheap"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL PASS")
