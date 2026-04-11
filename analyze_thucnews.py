"""
THUCNews 数据集分析与预处理脚本

THUCNews 是清华大学 NLP 实验室发布的中文新闻分类数据集。
本脚本完成:
  1. 全面分析数据集结构、规模、分布
  2. 预处理为内容审核模型可用的训练格式 (JSONL)
  3. 分析对审核模型的训练价值

使用方式:
  python3 analyze_thucnews.py --data_dir ./data/THUCNews --output_dir ./data/thucnews_processed
"""

import os
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple


# ============================================================
# 分析模块
# ============================================================

def discover_categories(data_dir: str) -> List[str]:
    """发现所有新闻分类目录"""
    p = Path(data_dir)
    categories = []
    for item in sorted(p.iterdir()):
        if item.is_dir() and not item.name.startswith(".") and not item.name.startswith("__"):
            categories.append(item.name)
    return categories


def count_files(category_path: Path) -> int:
    """统计某类别的文件数"""
    return len(list(category_path.glob("*.txt")))


def read_article(filepath: Path) -> Dict:
    """
    读取一篇新闻文件
    
    返回: {
        "title": 标题,
        "content": 正文,
        "full_text": 标题+正文拼接,
        "char_count": 字符数,
        "file_name": 文件名,
    }
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        
        if not lines:
            return None
        
        title = lines[0]
        content = "\n".join(lines[1:]) if len(lines) > 1 else ""
        full_text = title + ("\n" + content if content else "")
        
        return {
            "title": title,
            "content": content,
            "full_text": full_text.replace("\n", " ").strip(),
            "title_chars": len(title),
            "content_chars": len(content),
            "total_chars": len(full_text),
            "file_name": filepath.name,
        }
    except Exception as e:
        return None


def analyze_dataset(data_dir: str) -> Dict:
    """
    全面分析 THUCNews 数据集
    
    返回包含各项统计指标的分析报告字典
    """
    print("=" * 70)
    print("THUCNews 数据集分析")
    print("=" * 70)

    data_path = Path(data_dir)
    
    # 1. 发现分类
    categories = discover_categories(data_dir)
    print(f"\n[1] 数据集位置: {data_dir}")
    print(f"    分类数: {len(categories)}")

    if not categories:
        print("    [ERROR] 未找到任何分类目录!")
        return None

    # 2. 每类统计
    print(f"\n{'='*70}")
    print(f"[2] 各类别详细统计")
    print(f"{'='*70}")
    print(f"{'类别':<12} {'文件数':>8} {'总字符数':>12} {'平均长度':>10} {'最大长度':>10} {'最小长度':>10}")
    print(f"{'-'*70}")

    total_files = 0
    total_chars = 0
    all_lengths = []
    category_stats = {}
    category_samples = {}  # 每类保存几个样本用于展示

    for cat in categories:
        cat_path = data_path / cat
        files = list(cat_path.glob("*.txt"))
        n_files = len(files)
        
        if n_files == 0:
            continue
        
        cat_total_chars = 0
        cat_lengths = []
        samples = []
        
        for fpath in files[:500]:  # 先采样分析前500个避免太慢
            article = read_article(fpath)
            if article:
                cat_total_chars += article["total_chars"]
                cat_lengths.append(article["total_chars"])
                if len(samples) < 2:
                    samples.append(article)
        
        avg_len = int(sum(cat_lengths) / max(len(cat_lengths), 1))
        max_len = max(cat_lengths) if cat_lengths else 0
        min_len = min(cat_lengths) if cat_lengths else 0
        
        total_files += n_files
        total_chars += cat_total_chars
        all_lengths.extend(cat_lengths)
        
        category_stats[cat] = {
            "file_count": n_files,
            "total_chars": cat_total_chars,
            "avg_length": avg_len,
            "max_length": max_len,
            "min_length": min_len,
        }
        category_samples[cat] = samples
        
        print(f"{cat:<12} {n_files:>8,} {cat_total_chars:>12,} {avg_len:>10,} {max_len:>10,} {min_len:>10,}")

    # 3. 全局统计
    print(f"\n{'='*70}")
    print(f"[3] 数据集全局统计")
    print(f"{'='*70}")
    print(f"    总文件数:     {total_files:,}")
    print(f"    总字符数:     {total_chars:,}")
    print(f"    平均文本长度: {int(sum(all_lengths)/max(len(all_lengths),1)):,} 字符")
    print(f"    最大文本长度: {max(all_lengths):,} 字符" if all_lengths else "")
    print(f"    最小文本长度: {min(all_lengths):,} 字符" if all_lengths else "")

    # 4. 长度分布
    if all_lengths:
        length_bins = [(0,100),(100,200),(200,500),(500,1000),(1000,2000),
                       (2000,5000),(5000,10000),(10000,999999)]
        print(f"\n{'='*70}")
        print(f"[4] 文本长度分布")
        print(f"{'='*70}")
        for lo, hi in length_bins:
            cnt = sum(1 for l in all_lengths if lo <= l < hi)
            pct = cnt / len(all_lengths) * 100
            bar = "#" * int(pct / 2)
            label = f"{lo}-{hi}" if hi < 999999 else f"{lo}+"
            print(f"    {label:>10} : {cnt:>6,} ({pct:>5.1f}%) {bar}")

    # 5. 样例展示
    print(f"\n{'='*70}")
    print(f"[5] 每类样例展示 (标题)")
    print(f"{'='*70}")
    for cat in list(categories)[:10]:
        samples = category_samples.get(cat, [])
        if samples:
            s = samples[0]
            preview = s["title"][:60] + ("..." if len(s["title"]) > 60 else "")
            print(f"    [{cat}] {preview}")
            content_preview = s["content"][:80].replace("\n", " ") + ("..." if len(s["content"]) > 80 else "")
            print(f"           正文预览: {content_preview}")

    report = {
        "categories": categories,
        "category_stats": category_stats,
        "total_files": total_files,
        "total_chars": total_chars,
        "category_samples": category_samples,
    }
    return report


# ============================================================
# 预处理模块 — 转换为审核训练格式
# ============================================================

def preprocess_thucnews(
    data_dir: str,
    output_dir: str,
    max_samples_per_class: Optional[int] = None,
    max_length: int = 512,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    将 THUCNews 新闻数据预处理为内容审核模型的训练格式
    
    策略:
      - THUCNews 的所有新闻都是正规新闻 → 全部标记为 safe (合规)
      - 作为 safe 类的**高质量负样本**，帮助模型学习什么是正常内容
      - 可与违规类样本组合成均衡的训练集
    
    Args:
        data_dir: THUCNews 解压后的根目录
        output_dir: 输出目录
        max_samples_per_class: 每类最多取多少条 (None=全部)
        max_length: 单条文本最大字符长度
        val_ratio: 验证集比例
        seed: 随机种子
    """
    import random
    random.seed(seed)
    
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("[PREPROCESS] THUCNews -> 内容审核训练格式")
    print(f"{'='*70}")
    
    categories = discover_categories(data_dir)
    all_data = []
    
    for cat in categories:
        cat_path = data_path / cat
        files = list(cat_path.glob("*.txt"))
        
        if max_samples_per_class:
            random.shuffle(files)
            files = files[:max_samples_per_class]
        
        cat_count = 0
        for fpath in files:
            article = read_article(fpath)
            if not article:
                continue
            
            text = article["full_text"]
            
            # 清洗
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) < 5 or len(text) > max_length * 2:
                continue
            
            # 截断到 max_length
            text = text[:max_length]
            
            all_data.append({
                "text": text,
                "is_violation": 0,
                "violation_type": "safe",
                "risk_level": "safe",
                "source_category": cat,
                "source_title": article["title"],
            })
            cat_count += 1
        
        print(f"  [{cat:<12}] 处理 {cat_count:,} 条")
    
    # 打乱 & 分割
    random.shuffle(all_data)
    split_idx = max(1, int(len(all_data) * (1.0 - val_ratio)))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    # 写出
    train_file = out_path / "thucnews_safe_train.jsonl"
    val_file = out_path / "thucnews_safe_val.jsonl"
    
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # 统计
    print(f"\n  输出结果:")
    print(f"    训练集: {train_file}  ({len(train_data):,} 条)")
    print(f"    验证集: {val_file}    ({len(val_data):,} 条)")
    print(f"    总计:   {len(all_data):,} 条")
    
    return str(train_file), str(val_file), all_data


# ============================================================
# 训练价值分析
# ============================================================

def analyze_training_value(report: Dict, processed_data: List[Dict]):
    """
    分析 THUCNews 对内容审核模型的训练价值
    """
    print(f"\n{'='*70}")
    print("[ANALYSIS] THUCNews 对内容审核模型的训练价值")
    print(f"{'='*70}")

    total = report.get("total_files", 0)
    cats = report.get("categories", [])
    
    print("""
  ┌──────────────────────────────────────────────────────────────┐
  │  一、THUCNews 数据集概况                                      │
  ├──────────────────────────────────────────────────────────────┤
""" + f"""│  来源:   清华大学 NLP 实验室 (THUCTC)                          │
│  规模:   {total:,} 篇中文新闻                                       │
│  类别数: {len(cats)} 个新闻分类                                        │
│  格式:   每篇一个 .txt 文件 (标题 + 正文)                         │
│  特点:   正规新闻语料，语言规范、无违规内容                        │
  └──────────────────────────────────────────────────────────────┘
""")

    print("""
  ┌──────────────────────────────────────────────────────────────┐
  │  二、对本模型的核心价值                                         │
  ├──────────────────────────────────────────────────────────────┤
  │                                                              │
  │  价值 1: 高质量 SAFE 负样本                                    │
  │  ─────────────────────────────────────────                   │
  │  THUCNews 全部是正规新闻 → 天然适合作为 safe (合规) 类样本。    │
  │  相比合成数据，真实新闻具有:                                   │
  │    - 更丰富的词汇和句式变化                                    │
  │    - 更自然的语言模式                                         │
  │    - 覆盖多个领域 (财经/体育/娱乐/科技等)                      │
  │                                                              │
  │  价值 2: 提升模型区分能力                                      │
  │  ─────────────────────────────────────────                   │
  │  大量 safe 样本 + 少量 violation 样本的对比训练，               │
  │  能让模型学会"什么看起来像正常内容"，从而更准确地识别异常。       │
  │                                                              │
  │  价值 3: 多领域泛化                                           │
  │  ─────────────────────────────────────────                   │
  │  新闻覆盖多领域，有助于模型在不同话题上都具备判断力。             │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
""")

    print("""
  ┌──────────────────────────────────────────────────────────────┐
  │  三、推荐使用方式                                             │
  ├──────────────────────────────────────────────────────────────┤
  │                                                              │
  │  方案 A (推荐): 作为 safe 类基座                               │
  │    safe 样本:    THUCNews (~{safe_n:,} 条)                       │
  │    violation 样本: 合成数据 + 公开违规数据集                     │
  │    比例建议:      safe 占 50~60%                               │
  │                                                              │
  │  方案 B: 与现有数据合并                                        │
  │    python prepare_data.py --mode full                          │
  │    生成的 train.jsonl (含 safe+violation)                      │
  │    + thucnews_safe_train.jsonl (纯 safe)                       │
  │    → 合并后训练                                               │
  │                                                              │
  │  方案 C: 数据增强                                             │
  │    取 THUCNews 的标题作为短文本样本                             │
  │    取全文作为长文本样本                                        │
  │    增加模型对不同长度的鲁棒性                                  │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
""".format(safe_n=len(processed_data)))

    print("""
  ┌──────────────────────────────────────────────────────────────┐
  │  四、注意事项                                                 │
  ├──────────────────────────────────────────────────────────────┤
  │  - THUCNews 只有 safe 类，不含任何 violation 样本              │
  │  - 必须搭配违规样本一起训练，否则模型只会输出 "safe"           │
  │  - 新闻文本偏正式，建议补充口语化文本 (评论/聊天记录)          │
  │  - 数据时间较早 (2013年左右)，部分时效性词汇可能过时            │
  └──────────────────────────────────────────────────────────────┘
""")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="THUCNews 分析与预处理工具")
    parser.add_argument("--data_dir", type=str, default="./data/THUCNews",
                        help="THUCNews 解压后的目录")
    parser.add_argument("--output_dir", type=str, default="./data/thucnews_processed",
                        help="预处理输出目录")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="每类最多取多少条 (默认全部)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="单条文本最大截断长度")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="只分析不预处理")
    args = parser.parse_args()
    
    # Step 1: 分析
    report = analyze_dataset(args.data_dir)
    
    if report and not args.skip_preprocess:
        # Step 2: 预处理
        train_file, val_file, processed = preprocess_thucnews(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_samples_per_class=args.max_samples,
            max_length=args.max_length,
            val_ratio=args.val_ratio,
        )
        
        # Step 3: 训练价值分析
        analyze_training_value(report, processed)
        
        print(f"\n{'='*70}")
        print("[DONE]")
        print(f"{'='*70}")
        print(f"\n预处理文件:")
        print(f"  训练集: {train_file}")
        print(f"  验证集: {val_file}")
        print(f"\n合并到主训练数据的命令提示:")
        print(f"  cat {train_file} >> ./data/train.jsonl")
