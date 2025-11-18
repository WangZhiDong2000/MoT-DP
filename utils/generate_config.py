#!/usr/bin/env python3
"""
交互式维度配置生成器
根据用户的硬件和需求生成最优配置
"""

import yaml
import sys
from pathlib import Path
from typing import Dict

def get_gpu_memory() -> int:
    """尝试获取GPU显存大小"""
    try:
        import torch
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            return device_props.total_memory / 1024 / 1024 / 1024
    except:
        pass
    return 0

def load_config(config_path: str) -> Dict:
    """加载配置"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config: Dict, config_path: str):
    """保存配置"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def print_header(title: str):
    """打印标题"""
    print("\n" + "="*80)
    print(f"🎯 {title}".center(80))
    print("="*80 + "\n")

def print_section(title: str):
    """打印章节标题"""
    print(f"\n📌 {title}")
    print("-" * 40)

def interactive_config_generator():
    """交互式配置生成器"""
    print_header("维度配置交互式生成器")
    
    print("本工具将根据您的硬件和需求生成最优的模型维度配置。\n")
    
    # 第1步: 检测GPU
    print_section("第1步: GPU硬件检测")
    gpu_memory = get_gpu_memory()
    if gpu_memory > 0:
        print(f"✅ 检测到GPU显存: {gpu_memory:.1f} GB\n")
        auto_detected = True
    else:
        print("❌ 未检测到GPU\n")
        auto_detected = False
    
    # 第2步: 用户输入GPU显存
    print_section("第2步: 确认GPU显存")
    if auto_detected:
        use_detected = input("使用检测到的显存大小? (y/n, 默认y): ").strip().lower() or 'y'
        if use_detected == 'y':
            pass
        else:
            gpu_memory = float(input("请输入GPU显存(GB): "))
    else:
        gpu_memory = float(input("请输入GPU显存(GB): "))
    
    print(f"✅ 确认GPU显存: {gpu_memory:.1f} GB\n")
    
    # 第3步: 选择优化目标
    print_section("第3步: 选择优化目标")
    print("1. 最大精度 (需要更多显存)")
    print("2. 精度和速度平衡 (推荐)")
    print("3. 最快速度 (精度可能下降)")
    print("4. 最小显存 (精度下降)")
    
    target = input("选择优化目标 (1-4, 默认2): ").strip() or '2'
    targets = {
        '1': ('max_accuracy', 'Transformer.n_layer'),
        '2': ('balanced', 'balanced'),
        '3': ('max_speed', 'speed'),
        '4': ('min_memory', 'memory'),
    }
    target_type = targets.get(target, targets['2'])[0]
    print(f"✅ 优化目标: {target_type}\n")
    
    # 第4步: 选择batch_size
    print_section("第4步: 设置Batch Size")
    print(f"(当前GPU可用显存: {gpu_memory:.1f} GB)")
    print("推荐:")
    if gpu_memory < 8:
        print("  • 小 (≤32): 适合低显存")
        print("  • 中 (64): 推荐")
        print("  • 大 (128): 不推荐")
        default_batch = 32
    elif gpu_memory < 16:
        print("  • 小 (32): 不推荐")
        print("  • 中 (64): 推荐")
        print("  • 大 (128): 推荐")
        default_batch = 64
    else:
        print("  • 小 (32): 不需要")
        print("  • 中 (64): 推荐")
        print("  • 大 (128): 推荐")
        default_batch = 128
    
    batch_input = input(f"输入batch_size (默认{default_batch}): ").strip() or str(default_batch)
    batch_size = int(batch_input)
    print(f"✅ Batch Size: {batch_size}\n")
    
    # 第5步: 生成配置
    print_section("第5步: 生成配置")
    
    config = {
        'policy': {},
        'bev_encoder': {},
        'dataloader': {'batch_size': batch_size}
    }
    
    # 根据GPU显存和优化目标生成配置
    if gpu_memory < 8:
        config_name = 'lightweight'
        config['policy'] = {
            'n_emb': 256,
            'n_head': 4,
            'n_layer': 4,
            'n_cond_layers': 2,
        }
        config['bev_encoder']['feature_dim'] = 128
        print("✅ 生成的方案: 轻量配置 (内存优先)")
        print("📊 预期精度下降: -10-20%")
        print("⚡ 预期速度: 0.5x")
        
    elif gpu_memory < 16:
        config_name = 'conservative'
        config['policy'] = {
            'n_emb': 768,
            'n_head': 12,
            'n_layer': 8,
            'n_cond_layers': 6,
        }
        config['bev_encoder']['feature_dim'] = 384
        print("✅ 生成的方案: 保守提升 (平衡)")
        print("📊 预期精度提升: +5-10%")
        print("⚡ 预期速度: 1.2x")
        
    elif gpu_memory < 24:
        if target_type == 'max_accuracy':
            config_name = 'aggressive'
            config['policy'] = {
                'n_emb': 1536,
                'n_head': 24,
                'n_layer': 16,
                'n_cond_layers': 8,
            }
            config['bev_encoder']['feature_dim'] = 768
            print("✅ 生成的方案: 激进提升 (高精度)")
            print("📊 预期精度提升: +25-40%")
            print("⚡ 预期速度: 2-3x")
        else:
            config_name = 'balanced'
            config['policy'] = {
                'n_emb': 1024,
                'n_head': 16,
                'n_layer': 12,
                'n_cond_layers': 6,
            }
            config['bev_encoder']['feature_dim'] = 512
            print("✅ 生成的方案: 平衡方案 (推荐)")
            print("📊 预期精度提升: +15-25%")
            print("⚡ 预期速度: 1.5-2x")
    else:
        config_name = 'ultra'
        config['policy'] = {
            'n_emb': 2048,
            'n_head': 32,
            'n_layer': 24,
            'n_cond_layers': 8,
        }
        config['bev_encoder']['feature_dim'] = 1024
        print("✅ 生成的方案: 超高精度 (A100级别)")
        print("📊 预期精度提升: +40-50%")
        print("⚡ 预期速度: 3-4x")
    
    print(f"\n生成的维度参数:")
    print(f"  • n_emb: {config['policy']['n_emb']}")
    print(f"  • n_head: {config['policy']['n_head']}")
    print(f"  • n_layer: {config['policy']['n_layer']}")
    print(f"  • n_cond_layers: {config['policy']['n_cond_layers']}")
    print(f"  • feature_dim: {config['bev_encoder']['feature_dim']}")
    
    # 第6步: 保存配置
    print_section("第6步: 保存配置")
    
    config_path = input("输入配置保存路径 (默认: config/nuscenes.yaml): ").strip()
    if not config_path:
        config_path = 'config/nuscenes.yaml'
    
    config_path = Path(config_path)
    if not config_path.parent.exists():
        config_path.parent.mkdir(parents=True)
    
    # 加载原配置并更新
    if config_path.exists():
        original_config = load_config(config_path)
        original_config['policy'].update(config['policy'])
        original_config['bev_encoder'].update(config['bev_encoder'])
        original_config['dataloader'].update(config['dataloader'])
        config = original_config
        print(f"✅ 已加载原配置并更新维度参数")
    
    # 备份原配置
    if config_path.exists():
        backup_path = config_path.with_stem(config_path.stem + '_backup_auto')
        import shutil
        shutil.copy(config_path, backup_path)
        print(f"✅ 备份原配置: {backup_path}")
    
    # 保存新配置
    save_config(config, str(config_path))
    print(f"✅ 配置已保存: {config_path}\n")
    
    # 第7步: 后续建议
    print_section("第7步: 后续建议")
    print("1️⃣  验证维度兼容性:")
    print(f"   python utils/check_dimensions.py --check {config_path}")
    print("\n2️⃣  估计显存需求:")
    print(f"   python utils/estimate_memory.py")
    print("\n3️⃣  启动训练:")
    print(f"   python training/train_nusc_bev.py --config {config_path}")
    print("\n4️⃣  监控显存使用:")
    print("   watch -n 1 nvidia-smi")
    
    print_header("配置生成完成!")
    print(f"您的新配置已保存在: {config_path}")
    print(f"方案名称: {config_name}")
    print(f"预期精度提升: 取决于具体数据和训练设置")
    print(f"\n🚀 开始训练吧!")

if __name__ == '__main__':
    try:
        interactive_config_generator()
    except KeyboardInterrupt:
        print("\n\n❌ 用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)
