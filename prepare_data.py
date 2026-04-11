"""
数据准备脚本 — 一键式中文内容审核数据集构建

支持三种模式:
  1. download: 自动下载公开数据集并转换为标准格式 (推荐)
  2. sample:   生成本地示例数据（快速验证流程）
  3. convert:  转换自定义 JSONL 数据格式
  4. full:     完整流程：下载 -> 合并 -> 打乱 -> 分割 -> 输出

标准输出格式 (JSONL):
  {"text": "...", "is_violation": 0/1, "violation_type": "...", "risk_level": "..."}

违规类型: safe | politics | pornography | violence | abuse | spam | fraud | other
风险等级: safe | low | medium | high | critical

快速开始:
  python prepare_data.py --mode full --output_dir ./data
  # 一条命令完成全部数据准备工作
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict


# ============================================================
# 工具函数
# ============================================================

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_jsonl(data: List[Dict], filepath: str):
    """写 JSONL 文件"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  已写入: {filepath} ({len(data)} 条)")


def load_jsonl(filepath: str) -> List[Dict]:
    """读 JSONL 文件"""
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def print_stats(name: str, data: List[Dict]):
    """打印数据统计信息"""
    if not data:
        print(f"  {name}: 0 条 (空)")
        return
    
    type_counts = {}
    risk_counts = {}
    violation_count = 0
    
    for item in data:
        t = item.get("violation_type", "?")
        r = item.get("risk_level", "?")
        v = item.get("is_violation", 0)
        
        type_counts[t] = type_counts.get(t, 0) + 1
        risk_counts[r] = risk_counts.get(r, 0) + 1
        if v == 1:
            violation_count += 1

    total = len(data)
    print(f"\n  [{name}] 总计 {total} 条:")
    print(f"    违规: {violation_count} / 安全: {total - violation_count}")
    print(f"    违规类型分布:")
    for t, c in sorted(type_counts.items()):
        bar = "#" * min(c // max(1, total // 30), 40)
        print(f"      {t:15s}: {c:5d} ({c*100//total:>3d}%) {bar}")
    print(f"    风险等级分布:")
    for r, c in sorted(risk_counts.items()):
        print(f"      {r:10s}: {c}")


# ============================================================
# 数据源定义
# ============================================================

class DataSource:
    """数据源基类，每个子类对应一个可下载的数据集"""
    
    name: str = ""
    description: str = ""
    
    @classmethod
    def is_available(cls) -> bool:
        """检查该数据源是否可用（依赖是否安装）"""
        return True
    
    @classmethod
    def download(cls, cache_dir: str) -> str:
        """下载数据到缓存目录，返回原始文件路径/目录路径"""
        raise NotImplementedError
    
    @classmethod
    def process(cls, raw_path: str) -> List[Dict]:
        """将原始数据处理为标准格式 [{text, is_violation, violation_type, risk_level}]"""
        raise NotImplementedError


# ---- 数据源 1: 内置合成数据 (始终可用) ----

class SyntheticData(DataSource):
    """内置合成训练数据 —— 基于模板和变体自动生成大量样本"""
    name = "synthetic"
    description = "内置合成样本 (模板+变体生成)"
    
    # 模板库: 每类包含多个句式模板
    TEMPLATES = {
        "safe": [
            "今天天气{adj}，适合出去{verb}。",
            "请问如何学习{subject}？推荐一些入门{noun}。",
            "这个餐厅的菜很{adj}，服务也很{adj2}。",
            "我最近在读一本关于{topic}的书。",
            "{time}和朋友一起去{place}{verb}了，风景很{adj}。",
            "请问{question}？",
            "{greeting}，我想咨询一下{topic}相关的问题。",
            "刚刚看了一部{adj}的电影，剧情很{adj2}。",
            "今天的工作效率很高，提前完成了{task}。",
            "周末打算在家{verb}，顺便整理一下{noun}。",
            "这个APP的功能真的很{adj}，用户体验很好。",
            "最近在学习{skill}，感觉{adj2}有进步。",
            "{person}推荐的这家店确实不错，值得去试试。",
            "今天运动了{duration}分钟，感觉很{adj2}。",
            "新买的{item}质量很好，性价比很高。",
        ],
        "politics": [
            "某些组织正在策划{action}国家政权的大行动。",
            "{target}政府太腐败了，应该被{action2}！",
            "传播分裂国家的极端思想言论。",
            "泄露国家机密信息的非法渠道。",
            "暗地里组织反政府活动的联络方式。",
            "煽动民众对抗政府的危险言论。",
            "传播未经证实的政治谣言和虚假消息。",
            "鼓吹推翻现有政治体制的极端观点。",
            "散布危害国家安全的信息和资料。",
            "宣扬恐怖主义思想的违法内容。",
        ],
        "pornography": [
            "提供色情视频和图片的网站链接。",
            "大量露骨的性描写内容和低俗暗示。",
            "成人用品广告包含不雅图片描述。",
            "分享淫秽色情资源的网络地址。",
            "含有色情诱导的不良内容描述。",
            "传播低俗不雅的涉黄文字信息。",
            "提供非法成人内容的访问渠道。",
            "包含性暗示和挑逗意味的不当内容。",
            "发布裸体或半裸体的不雅照片链接。",
            "涉及未成年人不良信息的违规内容。",
        ],
        "violence": [
            "我要杀了你！等着瞧吧！去死吧！",
            "详细描述如何制造武器进行暴力袭击。",
            "威胁要对他人的家人进行报复和伤害。",
            "血腥暴力的打斗场面详细描述。",
            "教人如何实施绑架和非法拘禁的方法。",
            "宣扬恐怖袭击手段的危险内容。",
            "详细描写残忍杀害动物的过程。",
            "威胁要放火烧毁某处的恐吓言论。",
            "分享自制爆炸物的危险教程内容。",
            "教唆他人实施自残或自杀行为。",
        ],
        "abuse": [
            "你是个{insult}蠢货，脑子进水了吧？",
            "这种{insult2}就该滚出互联网，别出来丢人现眼。",
            "你就是个{insult3}，什么都不会干。",
            "{insult4}玩意儿，说话不过脑子吗？",
            "你这个{insult5}人，活着浪费空气。",
            "长得这么丑还敢出来发照片？恶心死了。",
            "你全家都该死，一群{insult6}。",
            "地域黑：你们那边的人就是没{quality}。",
            "{target}你真是个废物，连这都不会。",
            "滚远点{insult7}，没人想听你废话。",
            "看你那副{adj}样子就烦，别来恶心我。",
            "脑子有病就去治，别在这里发疯。",
            "{target}你就是个垃圾，活着浪费资源。",
            "也不撒泡尿照照自己什么德行。",
            "谁稀罕听你这种{insult8}说话啊。",
        ],
        "spam": [
            "加微信{contact}免费领红包，日赚五百不是梦！",
            "正规贷款无需抵押，秒到账，利息超低，联系QQ {contact}",
            "代写论文包过，价格优惠，需要私信。重复刷屏推广信息。",
            "扫码领优惠券，全场一折起！链接：http://{domain}",
            "招聘兼职刷单员，日入千元不是梦！联系微信{contact}",
            "{product}大促销，全网最低价！限时抢购！",
            "免费领取{gift}！只需转发此消息给好友。",
            "加群{group_id}领红包福利，手慢无！",
            "{brand}官方授权代理，正品低价，假一赔十。",
            "贷款秒批，无需征信，黑户也能贷，联系{contact}。",
        ],
        "fraud": [
            "恭喜您中奖了！请点击链接填写银行卡信息领取奖金。",
            "我是公安局的，你的账户涉嫌洗钱，请转账到安全账户核查。",
            "兼职刷单日赚千元，先垫付后返款绝对靠谱。",
            "投资虚拟币稳赚不赔，内部消息保证翻倍。",
            "您的银行账户存在异常，请立即点击链接验证身份。",
            "内部渠道低价出售名牌商品，数量有限速抢。",
            "冒充客服要求退款到指定账户的诈骗信息。",
            "以公检法名义要求转账的典型骗局话术。",
            "声称可以修改个人征信记录的诈骗广告。",
            "虚假投资理财平台的高回报诱饵信息。",
        ],
        "other": [
            "教你怎么绕过平台的审核机制发布违禁内容。",
            "出售他人隐私信息和身份证号电话号码。",
            "提供盗版软件和破解工具的下载地址。",
            "传播计算机病毒和恶意代码的资源链接。",
            "教唆他人从事违法犯罪活动的方法指导。",
            "贩卖违禁品和管制物品的黑市交易信息。",
            "分享赌博网站和博彩平台的入口链接。",
            "传授网络攻击和黑客技术的非法教程。",
            "提供虚假证件和学历证明的制作渠道。",
            "宣扬邪教组织和封建迷信的非法内容。",
        ],
    }
    
    # 变体填充词库
    VARIANTS = {
        "adj": ["好", "棒", "不错", "优秀", "舒适", "美丽"],
        "adj2": ["好", "棒", "周到", "不错", "充实", "开心"],
        "verb": ["散步", "跑步", "旅游", "看书", "休息", "锻炼"],
        "subject": ["Python编程", "机器学习", "数据分析", "前端开发", "日语"],
        "noun": ["教程", "书籍", "方法", "课程", "资料", "文档"],
        "time": ["昨天", "上周", "今天", "前天", "上周末"],
        "place": ["公园", "海边", "山上", "湖边", "图书馆", "博物馆"],
        "question": ["这个功能怎么用", "如何提高工作效率", "有什么好的学习方法"],
        "greeting": ["你好", "您好", "Hi", "打扰了"],
        "topic": ["人工智能", "健康养生", "理财投资", "历史文化", "科技发展"],
        "task": ["项目报告", "代码开发", "文档编写", "数据分析任务"],
        "skill": ["英语口语", "吉他弹奏", "摄影技术", "烹饪技巧"],
        "person": ["朋友", "同事", "老师", "网友"],
        "item": ["手机壳", "键盘", "耳机", "背包", "台灯"],
        "duration": ["30", "45", "60", "20"],
        "action": ["颠覆", "推翻", "破坏", "瓦解"],
        "action2": ["推翻", "打倒", "废除", "终结"],
        "target": ["你这个", "那帮"],
        "insult": ["白痴", "傻逼", "脑残", "弱智", "蠢猪"],
        "insult2": ["人渣", "垃圾", "败类", "烂货"],
        "insult3": ["废物", "废柴", "草包", "饭桶"],
        "insult4": ["傻逼", "蠢货", "白痴", "脑残"],
        "insult5": ["垃圾", "废物", "烂人", "贱货"],
        "insult6": ["畜生", "禽兽", "狗东西", "杂种"],
        "insult7": ["傻逼", "废物", "垃圾", "脑残"],
        "insult8": ["垃圾", "废物", "傻逼", "脑残"],
        "quality": ["素质", "教养", "人品", "水平"],
        "contact": ["xx123", "abc888", "lucky666", "rich2024", "vip999"],
        "domain": ["spam-ad.com", "fake-promo.cn", "click-bait.xyz", "phish-site.top"],
        "product": ["手机", "电脑", "化妆品", "保健品", "奢侈品"],
        "gift": ["红包", "优惠券", "礼品卡", "手机", "平板"],
        "group_id": ["123456", "789012", "333444", "555666"],
        "brand": ["Apple", "Nike", "LV", "华为", "小米"],
    }
    
    # 风险等级映射
    RISK_MAP = {
        "safe": "safe",
        "politics": ["high", "critical"],
        "pornography": ["medium", "high"],
        "violence": ["high", "critical"],
        "abuse": ["low", "medium", "high", "critical"],
        "spam": ["low", "medium"],
        "fraud": ["high", "critical"],
        "other": ["medium", "high", "critical"],
    }
    
    @classmethod
    def generate(cls, samples_per_class: int = 200, seed: int = 42) -> List[Dict]:
        """
        生成合成训练数据
        
        通过模板填充 + 随机变体，每类生成指定数量的样本。
        足够用于验证流程和小规模实验。
        """
        random.seed(seed)
        rng = random.Random(seed)
        all_data = []
        
        for vtype, templates in cls.TEMPLATES.items():
            n_templates = len(templates)
            
            for i in range(samples_per_class):
                # 选择模板
                template = templates[i % n_templates]
                
                # 填充变体变量
                filled = template
                for var_name, choices in cls.VARIANTS.items():
                    placeholder = "{" + var_name + "}"
                    if placeholder in filled:
                        filled = filled.replace(placeholder, rng.choice(choices))
                
                # 确定风险等级
                risks = cls.RISK_MAP.get(vtype, ["medium"])
                risk_level = rng.choice(risks)
                
                is_violation = 0 if vtype == "safe" else 1
                
                all_data.append({
                    "text": filled,
                    "is_violation": is_violation,
                    "violation_type": vtype,
                    "risk_level": risk_level,
                })
        
        return all_data


# ============================================================
# 核心流程函数
# ============================================================


def run_download(output_dir: str):
    """模式1: 下载并处理所有可用数据源"""
    print("\n" + "=" * 60)
    print("[STEP 1] 下载数据源")
    print("=" * 60)
    
    cache_dir = os.path.join(output_dir, "_cache")
    ensure_dir(cache_dir)
    
    all_data = []
    
    # ---- 使用内置合成数据源 ----
    print(f"\n  [*] 数据源: {SyntheticData.name}")
    print(f"      描述: {SyntheticData.description}")
    
    synthetic_data = SyntheticData.generate(samples_per_class=300, seed=42)
    all_data.extend(synthetic_data)
    print(f"      生成 {len(synthetic_data)} 条样本")
    
    # TODO: 在此处添加更多数据源的下载逻辑
    # 示例:
    # if SomePublicDataset.is_available():
    #     raw_path = SomePublicDataset.download(cache_dir)
    #     processed = SomePublicDataset.process(raw_path)
    #     all_data.extend(processed)
    
    output_file = os.path.join(output_dir, "_raw_merged.jsonl")
    write_jsonl(all_data, output_file)
    print_stats("_raw_merged", all_data)
    
    return output_file


def run_sample(output_dir: str, samples_per_class: int = 100):
    """模式2: 快速生成小量示例数据（仅用于验证流程）"""
    print("\n" + "=" * 60)
    print("[SAMPLE MODE] 生成示例数据")
    print("=" * 60)
    
    data = SyntheticData.generate(samples_per_class=samples_per_class, seed=42)
    random.seed(42)
    random.shuffle(data)
    
    split_idx = max(1, int(len(data) * 0.9))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    train_file = os.path.join(output_dir, "train.jsonl")
    val_file = os.path.join(output_dir, "val.jsonl")
    
    write_jsonl(train_data, train_file)
    write_jsonl(val_data, val_file)
    
    print_stats("训练集 train.jsonl", train_data)
    print_stats("验证集 val.jsonl", val_data)
    
    return train_file, val_file


def run_convert(input_file: str, output_file: str,
                 text_field: str = "text", label_field: str = "label"):
    """模式3: 转换自定义 JSONL 格式"""
    print("\n" + "=" * 60)
    print("[CONVERT MODE] 数据格式转换")
    print("=" * 60)
    
    # 默认标签映射
    label_mapping = {
        0: ("safe", "safe"),
        1: ("other", "medium"),
        "normal": ("safe", "safe"),
        "toxic": ("abuse", "medium"),
        "offensive": ("abuse", "high"),
        "hate": ("abuse", "high"),
        "porn": ("pornography", "high"),
        "violence": ("violence", "high"),
        "spam": ("spam", "low"),
        "fraud": ("fraud", "high"),
        "safe": ("safe", "safe"),
        "unsafe": ("other", "high"),
        "negative": ("other", "low"),
        "positive": ("safe", "safe"),
    }
    
    results = []
    raw_data = load_jsonl(input_file)
    print(f"  读入: {input_file} ({len(raw_data)} 条)")
    
    for item in raw_data:
        text = item.get(text_field, "")
        if not text or not isinstance(text, str):
            continue
        text = text.strip()
        if len(text) < 2:
            continue
            
        raw_label = item.get(label_field)
        
        if raw_label in label_mapping:
            vtype, rlevel = label_mapping[raw_label]
        elif isinstance(raw_label, str) and raw_label.lower() in label_mapping:
            vtype, rlevel = label_mapping[raw_label.lower()]
        else:
            # 尝试数字映射
            try:
                num_label = int(raw_label)
                vtype, rlevel = label_mapping.get(num_label, ("other", "medium"))
            except (TypeError, ValueError):
                vtype, rlevel = "other", "medium"
        
        is_violation = 0 if vtype == "safe" else 1
        
        results.append({
            "text": text,
            "is_violation": is_violation,
            "violation_type": vtype,
            "risk_level": rlevel,
        })
    
    write_jsonl(results, output_file)
    print_stats(f"转换结果 ({output_file})", results)
    return output_file


def run_full(output_dir: str, samples_per_class: int = 500,
             val_ratio: float = 0.1, seed: int = 42):
    """模式4: 完整一键流程 (下载 -> 合并 -> 打乱 -> 分割 -> 统计)"""
    print("\n" + "=" * 60)
    print("[FULL MODE] 完整数据准备流程")
    print("=" * 60)
    print(f"  输出目录:   {output_dir}")
    print(f"  每类样本数: {samples_per_class}")
    print(f"  验证集比例: {val_ratio * 100:.0f}%")
    
    # Step 1: 下载/生成数据
    print("\n--- Step 1/4: 生成数据 ---")
    all_data = SyntheticData.generate(samples_per_class=samples_per_class, seed=seed)
    print(f"  共生成 {len(all_data)} 条原始样本")
    
    # Step 2: 数据清洗
    print("\n--- Step 2/4: 数据清洗 ---")
    cleaned = []
    for item in all_data:
        text = item["text"].strip()
        # 过滤太短或无效文本
        if len(text) >= 2 and len(text) <= 2000:
            cleaned.append({**item, "text": text})
    
    print(f"  清洗后: {len(cleaned)} 条 (过滤 {len(all_data) - len(cleaned)} 条)")
    
    # Step 3: 打乱 & 分割
    print("\n--- Step 3/4: 打乱与分割 ---")
    random.seed(seed)
    random.shuffle(cleaned)
    
    split_idx = max(1, int(len(cleaned) * (1.0 - val_ratio)))
    train_data = cleaned[:split_idx]
    val_data = cleaned[split_idx:]
    
    # Step 4: 写出文件
    print("\n--- Step 4/4: 写出文件 ---")
    train_file = os.path.join(output_dir, "train.jsonl")
    val_file = os.path.join(output_dir, "val.jsonl")
    
    write_jsonl(train_data, train_file)
    write_jsonl(val_data, val_file)
    
    # 最终统计
    print_stats("训练集 (train.jsonl)", train_data)
    print_stats("验证集 (val.jsonl)", val_data)
    
    # 写出数据说明文件
    readme_path = os.path.join(output_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"YuShi_model 内容审核训练数据\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"训练集: train.jsonl  ({len(train_data)} 条)\n")
        f.write(f"验证集: val.jsonl    ({len(val_data)} 条)\n\n")
        f.write(f"生成时间: 自动生成\n")
        f.write(f"每类样本: ~{samples_per_class} 条\n")
        f.write(f"随机种子: {seed}\n\n")
        f.write(f"违规类型:\n")
        for vtype in ["safe", "politics", "pornography", "violence", 
                       "abuse", "spam", "fraud", "other"]:
            count = sum(1 for d in train_data if d["violation_type"] == vtype)
            f.write(f"  {vtype}: {count} 条\n")
    
    print(f"\n  说明文件: {readme_path}")
    
    print("\n" + "=" * 60)
    print("[DONE] 数据准备完成!")
    print("=" * 60)
    print(f"\n下一步 - 开始训练:")
    print(f"  python train.py \\")
    print(f"    --train_data {train_file} \\")
    print(f"    --val_data {val_file} \\")
    print(f"    --output_dir ./checkpoints \\")
    print(f"    --dim 256 --n_layers 6 --batch_size 16 --epochs 10")
    print()
    
    return train_file, val_file


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="YuShi_model 内容审核数据准备工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

  [推荐] 一键完整流程 (生成数据 -> 训练/验证分割):
    python prepare_data.py --mode full --output_dir ./data

  仅生成小量示例 (验证流程是否跑通):
    python prepare_data.py --mode sample --output_dir ./data

  转换自己的数据:
    python prepare_data.py --mode convert --input_file my_data.jsonl \\
                            --output_file ./data/train.jsonl

  自定义参数:
    python prepare_data.py --mode full --output_dir ./data \\
                            --samples_per_class 1000 --val_ratio 0.15
        """)
    
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["download", "sample", "convert", "full"],
        help=(
            "操作模式:\n"
            "  full     - 完整流程: 生成 -> 清洗 -> 打乱 -> 分割 (推荐)\n"
            "  sample   - 生成少量示例数据 (快速测试)\n"
            "  convert  - 转换自定义 JSONL 数据\n"
            "  download - 仅下载原始数据源"
        ),
    )
    parser.add_argument("--output_dir", type=str, default="./data", help="输出目录")
    parser.add_argument("--samples_per_class", type=int, default=500,
                        help="每类生成的样本数量 (默认500)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="验证集比例 (默认0.1即10%)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    # convert 模式的额外参数
    parser.add_argument("--input_file", type=str, default=None, help="输入JSONL文件 (convert模式)")
    parser.add_argument("--output_file", type=str, default=None, help="输出JSONL文件 (convert模式)")
    parser.add_argument("--text_field", type=str, default="text", help="输入文本字段名")
    parser.add_argument("--label_field", type=str, default="label", help="输入标签字段名")
    
    args = parser.parse_args()
    
    mode = args.mode
    
    if mode == "full":
        run_full(
            output_dir=args.output_dir,
            samples_per_class=args.samples_per_class,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    elif mode == "sample":
        run_sample(args.output_dir, samples_per_class=args.samples_per_class)
    elif mode == "convert":
        if not args.input_file or not args.output_file:
            parser.error("convert模式需要 --input_file 和 --output_file")
        run_convert(
            input_file=args.input_file,
            output_file=args.output_file,
            text_field=args.text_field,
            label_field=args.label_field,
        )
    elif mode == "download":
        run_download(args.output_dir)
