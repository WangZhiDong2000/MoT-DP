#!/usr/bin/env python3
"""
GPU显存估计工具
根据模型配置估计GPU显存需求
"""

import yaml
import sys
from pathlib import Path
from typing import Dict, Tuple

def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def estimate_model_memory(config: Dict) -> Dict:
    """
    详细估计模型显存占用 (单位: GB)
    基于参数量和中间激活
    """
    policy = config.get('policy', {})
    bev_encoder = config.get('bev_encoder', {})
    dataloader = config.get('dataloader', {})
    
    n_emb = policy.get('n_emb', 512)
    n_head = policy.get('n_head', 8)
    n_layer = policy.get('n_layer', 8)
    n_cond_layers = policy.get('n_cond_layers', 4)
    feature_dim = bev_encoder.get('feature_dim', 256)
    state_dim = bev_encoder.get('state_dim', 15)
    batch_size = dataloader.get('batch_size', 128)
    
    action_dim = policy.get('input_dim', 2)
    horizon = policy.get('horizon', 6)
    
    # 1. 模型参数 (单精度)
    # Transformer主网络
    head_dim = n_emb // n_head
    params_per_layer = (
        4 * n_emb * n_emb +  # 自注意力
        3 * n_emb * head_dim * n_head +  # QKV投影
        4 * n_emb * n_emb  # FFN
    )
    transformer_params = params_per_layer * n_layer / 1e9  # 转换为GB
    
    # 条件编码器
    cond_params = (
        feature_dim * n_emb +  # 输入投影
        4 * n_emb * n_emb * n_cond_layers  # 条件编码层
    ) / 1e9
    
    # BEV编码器 (粗略估计)
    bev_params = 100 / 1e9  # 假设100M参数
    
    total_params = transformer_params + cond_params + bev_params
    param_memory = total_params * 4 / 1024  # 单精度为4字节，转换为GB
    
    # 2. 优化器状态 (AdamW: 参数 + 动量 + 方差)
    optimizer_memory = param_memory * 2 * 4 / 4  # 4倍参数量
    
    # 3. 梯度内存
    gradient_memory = param_memory
    
    # 4. 激活函数内存 (主要贡献)
    # Batch中间激活：B * seq_len * n_emb * n_layer
    seq_len = horizon + 4 * 4  # action_horizon + obs_horizon * n_obs_steps
    activation_per_sample = seq_len * n_emb * n_layer * 4 / 1024 / 1024  # MB
    activations_memory = batch_size * activation_per_sample * 4 / 1024  # 转换为GB
    
    # 5. BEV特征和输入数据
    # BEV: B * feature_dim * 448 * 448
    bev_data_memory = batch_size * feature_dim * 448 * 448 * 4 / 1024 / 1024 / 1024
    
    # 6. 其他开销
    misc_memory = 2.0  # GPU驱动、缓冲等
    
    # 计算总显存
    total_memory = (
        param_memory + 
        optimizer_memory + 
        gradient_memory + 
        activations_memory + 
        bev_data_memory + 
        misc_memory
    )
    
    return {
        'param_memory': param_memory,
        'optimizer_memory': optimizer_memory,
        'gradient_memory': gradient_memory,
        'activation_memory': activations_memory,
        'bev_data_memory': bev_data_memory,
        'misc_memory': misc_memory,
        'total_memory': total_memory,
        'breakdown': {
            'parameters': f"{param_memory:.1f}GB",
            'optimizer': f"{optimizer_memory:.1f}GB",
            'gradients': f"{gradient_memory:.1f}GB",
            'activations': f"{activations_memory:.1f}GB",
            'bev_features': f"{bev_data_memory:.1f}GB",
            'misc': f"{misc_memory:.1f}GB",
        }
    }

def get_gpu_recommendations(total_memory: float) -> Tuple[list, str]:
    """根据显存需求推荐GPU"""
    recommendations = []
    
    if total_memory < 8:
        recommendations = [
            'NVIDIA RTX 3050 (6GB)',
            'NVIDIA RTX 2060 (6GB)',
            'NVIDIA T4 (16GB)',
        ]
        level = '低显存'
    elif total_memory < 12:
        recommendations = [
            'NVIDIA RTX 3060 (12GB)',
            'NVIDIA RTX 4060 (8GB)',
            'NVIDIA A10 (24GB)',
        ]
        level = '中等显存'
    elif total_memory < 16:
        recommendations = [
            'NVIDIA RTX 3080 (10GB)',
            'NVIDIA RTX 4080 (12GB)',
            'NVIDIA A100 (40GB)',
        ]
        level = '高显存'
    elif total_memory < 24:
        recommendations = [
            'NVIDIA RTX 3090 (24GB)',
            'NVIDIA RTX 4090 (24GB)',
            'NVIDIA A100 (40GB)',
        ]
        level = '高显存'
    else:
        recommendations = [
            'NVIDIA A100 (40GB)',
            'NVIDIA H100 (80GB)',
            'NVIDIA A6000 (48GB)',
            '多GPU训练',
        ]
        level = '超高显存'
    
    return recommendations, level

def print_memory_report(config_path: str):
    """打印详细的显存报告"""
    config = load_config(config_path)
    memory = estimate_model_memory(config)
    
    policy = config.get('policy', {})
    bev_encoder = config.get('bev_encoder', {})
    dataloader = config.get('dataloader', {})
    
    recommendations, level = get_gpu_recommendations(memory['total_memory'])
    
    print("\n" + "="*80)
    print("📊 GPU显存估计报告".center(80))
    print("="*80)
    
    # 配置总结
    print("\n📝 模型配置:")
    print(f"  • n_emb: {policy.get('n_emb', 512)}")
    print(f"  • n_head: {policy.get('n_head', 8)}")
    print(f"  • n_layer: {policy.get('n_layer', 8)}")
    print(f"  • n_cond_layers: {policy.get('n_cond_layers', 4)}")
    print(f"  • feature_dim: {bev_encoder.get('feature_dim', 256)}")
    print(f"  • batch_size: {dataloader.get('batch_size', 128)}")
    
    # 显存分解
    print(f"\n📈 显存分解 (GPU显存占用):")
    total = memory['total_memory']
    for component, value_str in memory['breakdown'].items():
        value = float(value_str.replace('GB', ''))
        percentage = (value / total) * 100
        bar = "█" * int(percentage / 5)
        print(f"  • {component:.<20} {value_str:>8} ({percentage:>5.1f}%) {bar}")
    
    print(f"\n🎯 总计显存需求: {total:.1f} GB")
    
    # GPU推荐
    print(f"\n💻 推荐GPU (显存等级: {level}):")
    for i, gpu in enumerate(recommendations, 1):
        print(f"  {i}. {gpu}")
    
    # 建议
    print(f"\n💡 建议:")
    if total > 24:
        print(f"  ⚠️  显存需求较大 ({total:.1f}GB)")
        print(f"  💾 可选方案:")
        print(f"    • 减小 batch_size (当前: {dataloader.get('batch_size', 128)})")
        print(f"    • 减小 n_emb 或 n_layer")
        print(f"    • 使用混合精度训练 (FP16)")
        print(f"    • 启用梯度检查点")
    elif total > 16:
        print(f"  ✅ 显存需求适中")
        print(f"  💡 可考虑使用RTX 3090或A100进行训练")
    else:
        print(f"  ✅ 显存需求较小")
        print(f"  💡 可在消费级GPU或普通服务器上训练")
    
    print("\n" + "="*80)

def compare_configs_memory(config1_path: str, config2_path: str):
    """对比两个配置的显存需求"""
    config1 = load_config(config1_path)
    config2 = load_config(config2_path)
    
    memory1 = estimate_model_memory(config1)
    memory2 = estimate_model_memory(config2)
    
    print("\n" + "="*80)
    print("📊 配置显存对比".center(80))
    print("="*80)
    
    print(f"\n配置1: {config1_path}")
    print(f"  总显存: {memory1['total_memory']:.1f} GB")
    
    print(f"\n配置2: {config2_path}")
    print(f"  总显存: {memory2['total_memory']:.1f} GB")
    
    diff = memory2['total_memory'] - memory1['total_memory']
    percent = (diff / memory1['total_memory']) * 100 if memory1['total_memory'] > 0 else 0
    
    print(f"\n📊 差异:")
    if diff > 0:
        print(f"  📈 配置2增加: +{diff:.1f} GB ({percent:+.0f}%)")
    else:
        print(f"  📉 配置2减少: {diff:.1f} GB ({percent:+.0f}%)")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        # 默认检查当前配置
        config_path = '/home/wang/Project/MoT-DP/config/nuscenes.yaml'
        print_memory_report(config_path)
    elif len(sys.argv) == 2:
        if sys.argv[1] == '--default':
            config_path = '/home/wang/Project/MoT-DP/config/nuscenes.yaml'
            print_memory_report(config_path)
        else:
            print_memory_report(sys.argv[1])
    elif len(sys.argv) == 3:
        compare_configs_memory(sys.argv[1], sys.argv[2])
    else:
        print("使用方法:")
        print("  python utils/estimate_memory.py                      # 检查默认配置")
        print("  python utils/estimate_memory.py <config_path>        # 检查指定配置")
        print("  python utils/estimate_memory.py <cfg1> <cfg2>        # 对比两个配置")
