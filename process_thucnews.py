"""
THUCNews 数据集处理脚本
将清华大学 THUCNews 新闻分类数据集转换为内容审核模型训练格式

THUCNews 包含 14 个新闻类别:
  体育、娱乐、家居、彩票、房产、教育、时尚、星座、
  游戏、社会、科技、股票、财经、时政

对于内容审核模型的价值:
  1. [核心价值] 作为 safe (合规) 类的大量负样本
     - 所有 14 类新闻都是正规媒体发布的合法内容
     - 可提供约 80万+ 条高质量 safe 样本

  2. [补充价值] 时政/社会类 → 可筛选出部分 politics 相关样本
     - 时政新闻涉及国内外政治，需人工/规则筛选边界样本
     - 社会新闻含社会事件报道，部分可作为 borderline 样本

  3. [注意] THUCNews 本身不含违规内容 (色情/暴力/辱骂/诈骗/垃圾广告)
     - 违规样本需从其他数据源补充
     - THUCNews 的主要作用是为模型提供"什么是正常文本"的判断基准

使用方式:
  python process_thucnews.py --thucnews_dir ./data/THUCNews --output_dir ./data
"""

import os
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# ============================================================
# THUCNews 类别与审核标签的映射策略
# ============================================================

# 每个类别的审核标签分配策略
CATEGORY_STRATEGY = {
    # 类别名: (violation_type, risk_level, 描述)
    "体育": ("safe", "safe", "体育新闻 - 合规"),
    "娱乐": ("safe", "safe", "娱乐新闻 - 合规"),
    "家居": ("safe", "safe", "家居新闻 - 合规"),
    "彩票": ("safe", "safe", "彩票新闻 - 合规(正规报道)"),
    "房产": ("safe", "safe", "房产新闻 - 合规"),
    "教育": ("safe", "safe", "教育新闻 - 合规"),
    "时尚": ("safe", "safe", "时尚新闻 - 合规"),
    "星座": ("safe", "safe", "星座新闻 - 合规"),
    "游戏": ("safe", "safe", "游戏新闻 - 合规"),
    "科技": ("safe", "safe", "科技新闻 - 合规"),
    "股票": ("safe", "safe", "股票新闻 - 合规(正规财经报道)"),
    "财经": ("safe", "safe", "财经新闻 - 合规"),
    # 以下两类需要特殊处理
    "社会": ("safe", "safe", "社会新闻 - 合规(含少量边界样本)"),
    "时政": ("safe", "safe", "时政新闻 - 合规(含少量边界样本)"),
}

# 时政类关键词筛选 - 可用于识别 politics 边界样本
POLITICS_KEYWORDS = [
    "颠覆", "推翻", "分裂", "反动", "叛乱", "政变",
    "极端势力", "恐怖组织", "邪教", "分裂势力",
    "煽动", "蛊惑", "策反", "叛国",
]

# 社会类中可能与 fraud 相关的关键词
FRAUD_KEYWORDS = [
    "诈骗", "被骗", "骗局", "诈骗案", "电信诈骗",
    "非法集资", "传销", "钓鱼网站", "虚假广告",
]

# 股票类中可能与 spam 相关的关键词
SPAM_KEYWORDS = [
    "加微信", "免费领取", "日赚", "稳赚不赔", "限时优惠",
    "点击链接", "红包领取", "优惠码", "折扣码",
]


def read_txt_file(filepath: str) -> Optional[str]:
    """读取 THUCNews 的 txt 文件，返回纯文本"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        # 清理多余空白
        text = re.sub(r"\s+", " ", text)
        return text
    except (UnicodeDecodeError, IOError):
        return None


def process_category(
    category_dir: str,
    category_name: str,
    max_samples: int = 0,
) -> List[Dict]:
    """
    处理 THUCNews 中某个类别的所有 txt 文件
    
    Args:
        category_dir: 类别目录路径
        category_name: 类别名称
        max_samples: 最大读取样本数 (0=全部)
    
    Returns:
        标准格式的样本列表
    """
    vtype, rlevel, desc = CATEGORY_STRATEGY.get(
        category_name, ("safe", "safe", f"{category_name} - 合规")
    )
    
    samples = []
    txt_files = sorted(Path(category_dir).glob("*.txt"))
    
    if max_samples > 0:
        txt_files = txt_files[:max_samples]
    
    for txt_file in txt_files:
        text = read_txt_file(str(txt_file))
        if not text or len(text.strip()) < 10:
            continue
        
        # 截断过长文本
        text = text[:512]
        
        # 对特殊类别做关键词检测，调整标签
        final_vtype = vtype
        final_rlevel = rlevel
        
        if category_name == "时政":
            for kw in POLITICS_KEYWORDS:
                if kw in text:
                    final_vtype = "politics"
                    final_rlevel = "medium"
                    break
        
        if category_name == "社会":
            for kw in FRAUD_KEYWORDS:
                if kw in text:
                    final_vtype = "fraud"
                    final_rlevel = "low"
                    break
        
        is_violation = 0 if final_vtype == "safe" else 1
        
        samples.append({
            "text": text,
            "is_violation": is_violation,
            "violation_type": final_vtype,
            "risk_level": final_rlevel,
            "source": f"THUCNews/{category_name}",
        })
    
    return samples


def process_thucnews(
    thucnews_dir: str,
    output_dir: str = "./data",
    samples_per_category: int = 0,
    val_ratio: float = 0.1,
    seed: int = 42,
    combine_with_existing: bool = True,
):
    """
    处理整个 THUCNews 数据集
    
    Args:
        thucnews_dir: THUCNews 根目录 (包含14个子文件夹)
        output_dir: 输出目录
        samples_per_category: 每个类别最多取多少条 (0=全部)
        val_ratio: 验证集比例
        seed: 随机种子
        combine_with_existing: 是否与已有的 train.jsonl 合并
    """
    root = Path(thucnews_dir)
    
    if not root.exists():
        print(f"[ERROR] 目录不存在: {thucnews_dir}")
        return
    
    # 发现所有类别目录
    categories = sorted([
        d.name for d in root.iterdir()
        if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
    ])
    
    print("\n" + "=" * 60)
    print("THUCNews 数据集处理")
    print("=" * 60)
    print(f"  数据目录: {thucnews_dir}")
    print(f"  发现类别: {len(categories)} 个")
    print(f"  每类上限: {samples_per_category if samples_per_category > 0 else '不限'}")
    
    # 逐类别处理
    all_samples = []
    category_stats = {}
    
    for cat_name in categories:
        cat_dir = root / cat_name
        if not cat_dir.is_dir():
            continue
        
        strategy = CATEGORY_STRATEGY.get(cat_name, ("safe", "safe", ""))
        print(f"\n  处理: {cat_name}/ ({strategy[2]})")
        
        samples = process_category(
            str(cat_dir), cat_name,
            max_samples=samples_per_category,
        )
        
        all_samples.extend(samples)
        
        # 统计
        vtype_counts = {}
        for s in samples:
            vt = s["violation_type"]
            vtype_counts[vt] = vtype_counts.get(vt, 0) + 1
        
        category_stats[cat_name] = {
            "total": len(samples),
            "types": vtype_counts,
        }
        
        type_str = ", ".join(f"{k}:{v}" for k, v in vtype_counts.items())
        print(f"    -> {len(samples)} 条 ({type_str})")
    
    # 打印汇总统计
    print("\n" + "-" * 60)
    print("  汇总统计:")
    
    total_by_type = {}
    for s in all_samples:
        vt = s["violation_type"]
        total_by_type[vt] = total_by_type.get(vt, 0) + 1
    
    for vt, count in sorted(total_by_type.items()):
        print(f"    {vt:15s}: {count:6d} 条")
    
    total = len(all_samples)
    safe_count = total_by_type.get("safe", 0)
    violation_count = total - safe_count
    print(f"    {'总计':15s}: {total:6d} 条 (safe:{safe_count}, violation:{violation_count})")
    
    # 与已有数据合并
    if combine_with_existing:
        existing_train = os.path.join(output_dir, "train.jsonl")
        if os.path.exists(existing_train):
            print(f"\n  检测到已有训练数据: {existing_train}")
            old_data = []
            with open(existing_train, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            old_data.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            
            print(f"  已有数据: {len(old_data)} 条")
            
            # 合并并去重 (按 text 去重)
            existing_texts = set(s["text"] for s in all_samples)
            new_from_old = []
            for item in old_data:
                if item.get("text", "") not in existing_texts:
                    new_from_old.append(item)
                    existing_texts.add(item["text"])
            
            all_samples.extend(new_from_old)
            print(f"  合并后总计: {len(all_samples)} 条 (+{len(new_from_old)} from old)")
    
    # 打乱 & 分割
    random.seed(seed)
    random.shuffle(all_samples)
    
    split_idx = max(1, int(len(all_samples) * (1.0 - val_ratio)))
    train_data = all_samples[:split_idx]
    val_data = all_samples[split_idx:]
    
    # 写出
    ensure_dir(output_dir)
    
    train_file = os.path.join(output_dir, "train.jsonl")
    val_file = os.path.join(output_dir, "val.jsonl")
    
    write_jsonl(train_data, train_file)
    write_jsonl(val_data, val_file)
    
    # 最终统计
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    _print_final_stats("训练集", train_data)
    _print_final_stats("验证集", val_data)
    
    # 数据价值分析
    _print_value_analysis(all_samples)
    
    return train_file, val_file


# ============================================================
# 辅助函数
# ============================================================

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_jsonl(data: List[Dict], filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  已写入: {filepath} ({len(data)} 条)")


def _print_final_stats(name: str, data: List[Dict]):
    if not data:
        return
    type_counts = {}
    for item in data:
        t = item.get("violation_type", "?")
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"\n  [{name}] {len(data)} 条:")
    for t, c in sorted(type_counts.items()):
        print(f"    {t:15s}: {c}")


def _print_value_analysis(data: List[Dict]):
    """打印数据价值分析"""
    print("\n" + "=" * 60)
    print("THUCNews 对本模型的数据价值分析")
    print("=" * 60)
    
    type_counts = {}
    for item in data:
        t = item.get("violation_type", "?")
        type_counts[t] = type_counts.get(t, 0) + 1
    
    safe_count = type_counts.get("safe", 0)
    total = len(data)
    
    print(f"""
  [1] THUCNews 提供了大量 safe (合规) 样本: {safe_count} 条 ({safe_count*100//max(total,1)}%)
      -> 核心价值: 让模型学会"什么是正常文本"
      -> 如果只有违规样本没有合规样本，模型会误判正常内容为违规
      -> safe 类建议占总训练数据的 40~60%

  [2] THUCNews 缺少的违规类别 (需从其他来源补充):
      -> politics (涉政敏感): 仅少量时政边界样本
      -> pornography (色情低俗): 完全缺失
      -> violence (暴力血腥): 完全缺失
      -> abuse (辱骂攻击): 完全缺失
      -> spam (垃圾广告): 完全缺失
      -> fraud (欺诈诈骗): 仅少量社会新闻提及

  [3] 建议方案:
      a) THUCNews 的 safe 样本 + prepare_data.py 的合成违规样本 -> 基线模型
      b) 从 HuggingFace/GitHub 下载 COLD 等数据集补充 abuse 类
      c) 用大模型 API 生成 pornography/violence/fraud/spam 类样本
      d) 最终目标比例:
         safe:    40~50%  (THUCNews 提供)
         abuse:   15~20%  (COLD + 合成)
         spam:    10~15%  (合成)
         violence: 5~10%  (合成)
         fraud:    5~10%  (合成)
         politics: 5~10%  (时政边界 + 合成)
         pornography: 5%   (合成)
         other:    5%     (合成)
    """)


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="THUCNews 数据集处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

  # 处理全部 THUCNews 数据 (每类不限量)
  python process_thucnews.py --thucnews_dir ./data/THUCNews --output_dir ./data

  # 每类只取 5000 条 (快速处理)
  python process_thucnews.py --thucnews_dir ./data/THUCNews --output_dir ./data \\
                              --samples_per_category 5000

  # 不与已有数据合并
  python process_thucnews.py --thucnews_dir ./data/THUCNews --output_dir ./data \\
                              --no_combine
        """)
    
    parser.add_argument("--thucnews_dir", type=str, default="./data/THUCNews",
                        help="THUCNews 根目录路径")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="输出目录")
    parser.add_argument("--samples_per_category", type=int, default=5000,
                        help="每类最多取多少条 (0=全部, 默认5000)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="验证集比例 (默认0.1)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no_combine", action="store_true",
                        help="不与已有 train.jsonl 合并")
    
    args = parser.parse_args()
    
    process_thucnews(
        thucnews_dir=args.thucnews_dir,
        output_dir=args.output_dir,
        samples_per_category=args.samples_per_category,
        val_ratio=args.val_ratio,
        seed=args.seed,
        combine_with_existing=not args.no_combine,
    )
