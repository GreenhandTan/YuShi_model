"""
规则引擎 — 多层级内容审核的第一层

设计目标:
- 极速响应 (<1ms)，处理高置信度的明确违规和明确安全场景
- 只有灰色地带才需要交给模型推理
- 支持自定义规则扩展
- 支持正则表达式匹配变体/谐音词

使用方式:
    engine = RuleEngine()
    result = engine.check("这是一段测试文本")
    if result is not None:
        # 规则引擎已判定，无需模型推理
        return result
    else:
        # 灰色地带，交给模型
        return model.audit(text)
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class RuleMatch:
    """规则匹配结果"""
    is_violation: bool
    risk_level: str
    violation_type: str
    confidence: float
    reason: str
    matched_rules: List[str] = field(default_factory=list)
    layer: str = "rule"


class RuleEngine:
    """
    规则引擎：快速处理高置信度的明确场景
    
    三级判定:
    1. 黑名单关键词 → 直接判定违规
    2. 白名单关键词 → 直接判定安全
    3. 正则模式匹配 → 捕捉变体/谐音
    4. 以上都不匹配 → 返回 None，交给模型
    """

    def __init__(self, rules_dir: Optional[str] = None):
        """
        Args:
            rules_dir: 规则文件目录，包含 blacklist.json / whitelist.json / patterns.json
                       如果为 None，使用内置默认规则
        """
        self._blacklist: List[Dict[str, Any]] = []
        self._whitelist: List[str] = []
        self._patterns: List[Dict[str, Any]] = []

        if rules_dir and Path(rules_dir).exists():
            self._load_rules(rules_dir)
        else:
            self._load_default_rules()

        # 编译正则表达式
        self._compiled_patterns: List[Tuple[re.Pattern, Dict[str, Any]]] = []
        for p in self._patterns:
            try:
                compiled = re.compile(p["pattern"], re.IGNORECASE)
                self._compiled_patterns.append((compiled, p))
            except re.error:
                pass

        # 预编译黑名单为 set 加速查找
        self._blacklist_map: Dict[str, Dict[str, Any]] = {}
        for item in self._blacklist:
            word = item.get("word", "")
            if word:
                self._blacklist_map[word] = item

    def _load_rules(self, rules_dir: str):
        """从文件加载规则"""
        base = Path(rules_dir)

        bl_path = base / "blacklist.json"
        if bl_path.exists():
            with open(bl_path, "r", encoding="utf-8") as f:
                self._blacklist = json.load(f)

        wl_path = base / "whitelist.json"
        if wl_path.exists():
            with open(wl_path, "r", encoding="utf-8") as f:
                self._whitelist = json.load(f)

        pat_path = base / "patterns.json"
        if pat_path.exists():
            with open(pat_path, "r", encoding="utf-8") as f:
                self._patterns = json.load(f)

    def _load_default_rules(self):
        """加载内置默认规则"""
        # ---- 黑名单: 高置信度违规关键词 ----
        # 每条规则包含: word(关键词), type(违规类型), risk(风险等级)
        self._blacklist = [
            # 色情类
            {"word": "约炮", "type": "pornography", "risk": "high"},
            {"word": "裸聊", "type": "pornography", "risk": "high"},
            {"word": "援交", "type": "pornography", "risk": "high"},
            {"word": "成人视频", "type": "pornography", "risk": "high"},
            {"word": "色情网站", "type": "pornography", "risk": "high"},
            {"word": "黄色网站", "type": "pornography", "risk": "high"},
            {"word": "AV种子", "type": "pornography", "risk": "high"},
            {"word": "一夜情", "type": "pornography", "risk": "medium"},

            # 暴力类
            {"word": "杀人方法", "type": "violence", "risk": "critical"},
            {"word": "制造炸弹", "type": "violence", "risk": "critical"},
            {"word": "如何自杀", "type": "violence", "risk": "critical"},
            {"word": "自杀方法", "type": "violence", "risk": "critical"},

            # 欺诈类
            {"word": "刷单兼职", "type": "fraud", "risk": "high"},
            {"word": "日赚千元", "type": "fraud", "risk": "medium"},
            {"word": "零风险高回报", "type": "fraud", "risk": "high"},
            {"word": "代开发票", "type": "fraud", "risk": "high"},

            # 垃圾广告类
            {"word": "加微信领", "type": "spam", "risk": "low"},
            {"word": "扫码免费领", "type": "spam", "risk": "low"},
        ]

        # ---- 白名单: 明确安全的关键词/短语 ----
        self._whitelist = [
            "天气真好",
            "今天吃什么",
            "早上好",
            "晚安",
            "谢谢",
            "你好",
            "再见",
        ]

        # ---- 正则模式: 捕捉变体/谐音/隐晦表达 ----
        self._patterns = [
            # 联系方式类 (QQ/微信/手机号引流)
            {
                "name": "contact_redirect",
                "pattern": r"(加我|加v|加微|加扣|私聊|私信)\s*.{0,5}(QQ|qq|微信|VX|wx|薇信)",
                "type": "spam",
                "risk": "medium",
                "confidence": 0.7,
            },
            # 金钱交易诱导
            {
                "name": "money_lure",
                "pattern": r"(转账|汇款|打款)\s*.{0,10}(到|至)\s*.{0,20}(账户|银行卡|支付宝)",
                "type": "fraud",
                "risk": "high",
                "confidence": 0.75,
            },
            # 赌博类
            {
                "name": "gambling",
                "pattern": r"(网赌|网赚|投注|下注|赔率|博彩|彩票预测)",
                "type": "other",
                "risk": "high",
                "confidence": 0.7,
            },
            # 毒品类
            {
                "name": "drugs",
                "pattern": r"(冰毒|大麻|海洛因|摇头丸|K粉|麻古|溜冰)",
                "type": "other",
                "risk": "critical",
                "confidence": 0.9,
            },
            # 政治敏感 (涉及推翻、颠覆等极端言论)
            {
                "name": "political_extreme",
                "pattern": r"(推翻政府|颠覆政权|分裂国家|恐怖主义)",
                "type": "politics",
                "risk": "critical",
                "confidence": 0.85,
            },
        ]

    def check(self, text: str) -> Optional[Dict[str, Any]]:
        """
        对文本执行规则检查

        Returns:
            - 如果规则引擎能判定: 返回审核结果 dict (与模型输出格式一致)
            - 如果是灰色地带: 返回 None，应交给模型推理
        """
        if not text or not text.strip():
            return self._make_result(
                is_violation=False, risk_level="safe", violation_type="safe",
                confidence=1.0, reason="空文本", matched_rules=["empty_text"]
            )

        text_stripped = text.strip()

        # ---- 第一优先级: 黑名单关键词匹配 ----
        for word, item in self._blacklist_map.items():
            if word in text_stripped:
                return self._make_result(
                    is_violation=True,
                    risk_level=item.get("risk", "high"),
                    violation_type=item.get("type", "other"),
                    confidence=0.95,
                    reason=f"命中违规关键词: {word}",
                    matched_rules=[f"blacklist:{word}"]
                )

        # ---- 第二优先级: 正则模式匹配 ----
        for pattern, item in self._compiled_patterns:
            match = pattern.search(text_stripped)
            if match:
                return self._make_result(
                    is_violation=True,
                    risk_level=item.get("risk", "medium"),
                    violation_type=item.get("type", "other"),
                    confidence=item.get("confidence", 0.7),
                    reason=f"命中违规模式: {item.get('name', 'unknown')} (匹配: {match.group()})",
                    matched_rules=[f"pattern:{item.get('name', 'unknown')}"]
                )

        # ---- 第三优先级: 白名单关键词匹配 (仅对短文本生效) ----
        if len(text_stripped) <= 20:
            for word in self._whitelist:
                if word in text_stripped:
                    return self._make_result(
                        is_violation=False, risk_level="safe", violation_type="safe",
                        confidence=0.9, reason=f"命中安全关键词: {word}",
                        matched_rules=[f"whitelist:{word}"]
                    )

        # ---- 灰色地带: 无法判定，交给模型 ----
        return None

    def _make_result(
        self,
        is_violation: bool,
        risk_level: str,
        violation_type: str,
        confidence: float,
        reason: str,
        matched_rules: List[str],
    ) -> Dict[str, Any]:
        return {
            "is_violation": is_violation,
            "risk_level": risk_level,
            "violation_type": violation_type if is_violation else "safe",
            "confidence": round(confidence, 4),
            "reason": reason,
            "matched_rules": matched_rules,
            "layer": "rule",
        }

    def add_blacklist(self, word: str, violation_type: str = "other", risk_level: str = "high"):
        """动态添加黑名单关键词"""
        item = {"word": word, "type": violation_type, "risk": risk_level}
        self._blacklist.append(item)
        self._blacklist_map[word] = item

    def add_whitelist(self, word: str):
        """动态添加白名单关键词"""
        if word not in self._whitelist:
            self._whitelist.append(word)

    def save_rules(self, rules_dir: str):
        """保存当前规则到文件"""
        base = Path(rules_dir)
        base.mkdir(parents=True, exist_ok=True)

        with open(base / "blacklist.json", "w", encoding="utf-8") as f:
            json.dump(self._blacklist, f, ensure_ascii=False, indent=2)
        with open(base / "whitelist.json", "w", encoding="utf-8") as f:
            json.dump(self._whitelist, f, ensure_ascii=False, indent=2)
        with open(base / "patterns.json", "w", encoding="utf-8") as f:
            json.dump(self._patterns, f, ensure_ascii=False, indent=2)

    def stats(self) -> Dict[str, int]:
        return {
            "blacklist_count": len(self._blacklist),
            "whitelist_count": len(self._whitelist),
            "pattern_count": len(self._patterns),
        }


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    import time

    print("=" * 60)
    print("规则引擎测试")
    print("=" * 60)

    engine = RuleEngine()

    stats = engine.stats()
    print(f"\n规则统计: {stats}")

    test_cases = [
        "今天天气真好，准备去跑步",
        "约炮加我微信",
        "制造炸弹教程",
        "刷单兼职日赚千元",
        "你好呀",
        "这是一段比较模糊的文本内容",
        "加我QQ12345678",
        "网赌平台推荐",
    ]

    print(f"\n{'文本':<30} {'判定':<8} {'类型':<15} {'风险':<10} {'来源':<8}")
    print("-" * 85)

    for text in test_cases:
        t0 = time.time()
        result = engine.check(text)
        elapsed = (time.time() - t0) * 1000

        if result:
            status = "违规" if result["is_violation"] else "安全"
            print(f"{text:<30} {status:<8} {result['violation_type']:<15} "
                  f"{result['risk_level']:<10} 规则({elapsed:.2f}ms)")
        else:
            print(f"{text:<30} {'待审':<8} {'---':<15} {'---':<10} 模型")

    print("\n[OK] 测试通过!")
