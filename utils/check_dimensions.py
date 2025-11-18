#!/usr/bin/env python3
"""
维度兼容性检查和配置对比工具
用于验证模型维度配置是否合理
"""

import yaml
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple

def load_config(config_path: str) -> Dict:
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config: Dict, config_path: str):
    """保存YAML配置文件"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def check_dimension_compatibility(config: Dict) -> Tuple[bool, list]:
    """
    检查维度配置的兼容性
    返回: (是否兼容, 错误/警告列表)
    """
    issues = []
    policy = config.get('policy', {})
    bev_encoder = config.get('bev_encoder', {})
    
    n_emb = policy.get('n_emb', 512)
    n_head = policy.get('n_head', 8)
    n_layer = policy.get('n_layer', 8)
    n_cond_layers = policy.get('n_cond_layers', 4)
    feature_dim = bev_encoder.get('feature_dim', 256)
    state_dim = bev_encoder.get('state_dim', 15)
    
    # 检查1: n_emb必须能被n_head整除
    if n_emb % n_head != 0:
        issues.append({
            'level': 'ERROR',
            'message': f'n_emb ({n_emb}) 必须能被 n_head ({n_head}) 整除',
            'suggestion': f'建议改为: n_emb={n_head * (n_emb // n_head)} 或 n_head={n_emb // (n_emb // n_head)}'
        })
    
    # 检查2: 每个注意力头的维度应该在64-256之间
    head_dim = n_emb // n_head if n_head > 0 else 0
    if head_dim < 32:
        issues.append({
            'level': 'WARNING',
            'message': f'每个注意力头维度 ({head_dim}) 过小，可能影响性能',
            'suggestion': f'建议增加 n_emb 或减少 n_head'
        })
    elif head_dim > 256:
        issues.append({
            'level': 'WARNING',
            'message': f'每个注意力头维度 ({head_dim}) 过大，可能浪费计算',
            'suggestion': f'建议减少 n_emb 或增加 n_head'
        })
    
    # 检查3: n_emb应该是4的倍数
    if n_emb % 4 != 0:
        issues.append({
            'level': 'ERROR',
            'message': f'n_emb ({n_emb}) 应该是4的倍数',
            'suggestion': f'建议改为: n_emb={(n_emb // 4) * 4}'
        })
    
    # 检查4: feature_dim与n_emb的关系
    ratio = feature_dim / n_emb if n_emb > 0 else 0
    if ratio < 0.25:
        issues.append({
            'level': 'WARNING',
            'message': f'feature_dim ({feature_dim}) 相对于 n_emb ({n_emb}) 太小 (比例 {ratio:.2f})',
            'suggestion': f'建议增加 feature_dim 或减少 n_emb'
        })
    elif ratio > 2:
        issues.append({
            'level': 'WARNING',
            'message': f'feature_dim ({feature_dim}) 相对于 n_emb ({n_emb}) 太大 (比例 {ratio:.2f})',
            'suggestion': f'建议减少 feature_dim 或增加 n_emb'
        })
    
    # 检查5: 层数的合理范围
    if n_layer < 2:
        issues.append({
            'level': 'WARNING',
            'message': f'n_layer ({n_layer}) 过小，可能影响模型容量',
            'suggestion': f'建议至少设置为 4'
        })
    elif n_layer > 32:
        issues.append({
            'level': 'WARNING',
            'message': f'n_layer ({n_layer}) 过大，可能导致训练困难',
            'suggestion': f'建议不超过 16'
        })
    
    if n_cond_layers < 2:
        issues.append({
            'level': 'WARNING',
            'message': f'n_cond_layers ({n_cond_layers}) 过小',
            'suggestion': f'建议至少设置为 2'
        })
    elif n_cond_layers > 16:
        issues.append({
            'level': 'WARNING',
            'message': f'n_cond_layers ({n_cond_layers}) 过大',
            'suggestion': f'建议不超过 8'
        })
    
    return len([i for i in issues if i['level'] == 'ERROR']) == 0, issues

def estimate_memory_and_speed(config: Dict) -> Dict:
    """
    估计显存使用和相对训练速度
    """
    policy = config.get('policy', {})
    bev_encoder = config.get('bev_encoder', {})
    dataloader = config.get('dataloader', {})
    
    n_emb = policy.get('n_emb', 512)
    n_head = policy.get('n_head', 8)
    n_layer = policy.get('n_layer', 8)
    n_cond_layers = policy.get('n_cond_layers', 4)
    feature_dim = bev_encoder.get('feature_dim', 256)
    batch_size = dataloader.get('batch_size', 128)
    
    # 基准配置（512, 8, 8, 4）对应100%
    baseline_n_emb = 512
    baseline_n_head = 8
    baseline_n_layer = 8
    baseline_n_cond_layers = 4
    
    # 计算内存因子
    emb_factor = (n_emb / baseline_n_emb) ** 2
    layer_factor = (n_layer / baseline_n_layer) * 0.5
    cond_factor = (n_cond_layers / baseline_n_cond_layers) * 0.3
    feature_factor = (feature_dim / 256) * 0.2
    
    memory_factor = emb_factor + layer_factor + cond_factor + feature_factor
    
    # 计算速度因子
    speed_factor = (n_emb / baseline_n_emb) * (n_layer / baseline_n_layer) * \
                   (n_cond_layers / baseline_n_cond_layers) * 0.5 + 0.5
    
    return {
        'memory_factor': memory_factor,
        'memory_percentage': f"{memory_factor * 100:.0f}%",
        'speed_factor': speed_factor,
        'speed_relative': f"{speed_factor:.1f}x (与基准配置相比)",
        'estimated_gpu_memory_gb': f"~{memory_factor * 16:.1f} GB (假设基准配置16GB)",
        'batch_size': batch_size,
        'total_params_millions': estimate_params(n_emb, n_layer, n_cond_layers, feature_dim)
    }

def estimate_params(n_emb: int, n_layer: int, n_cond_layers: int, feature_dim: int) -> float:
    """估计模型参数量（百万）"""
    # 简单估计
    transformer_params = (n_emb * n_emb * 4 * n_layer) / 1e6
    cond_encoder_params = (feature_dim * n_emb + n_emb * n_emb * 4 * n_cond_layers) / 1e6
    return transformer_params + cond_encoder_params

def print_compatibility_report(config_path: str):
    """打印完整的兼容性报告"""
    config = load_config(config_path)
    is_compatible, issues = check_dimension_compatibility(config)
    resources = estimate_memory_and_speed(config)
    
    print("\n" + "="*80)
    print("📊 维度兼容性检查报告".center(80))
    print("="*80)
    
    # 配置信息
    print("\n📝 当前配置:")
    policy = config.get('policy', {})
    bev_encoder = config.get('bev_encoder', {})
    print(f"  • n_emb: {policy.get('n_emb', 512)}")
    print(f"  • n_head: {policy.get('n_head', 8)}")
    print(f"  • n_layer: {policy.get('n_layer', 8)}")
    print(f"  • n_cond_layers: {policy.get('n_cond_layers', 4)}")
    print(f"  • feature_dim: {bev_encoder.get('feature_dim', 256)}")
    
    # 计算值
    n_emb = policy.get('n_emb', 512)
    n_head = policy.get('n_head', 8)
    head_dim = n_emb // n_head if n_head > 0 else 0
    print(f"\n🔢 计算值:")
    print(f"  • 每个注意力头维度: {head_dim}")
    print(f"  • feature_dim / n_emb 比例: {bev_encoder.get('feature_dim', 256) / n_emb:.2f}")
    
    # 兼容性检查
    status = "✅ 通过" if is_compatible else "❌ 失败"
    print(f"\n{status} 兼容性检查:")
    if issues:
        for issue in issues:
            icon = "🔴" if issue['level'] == 'ERROR' else "🟡"
            print(f"\n  {icon} [{issue['level']}] {issue['message']}")
            print(f"     💡 {issue['suggestion']}")
    else:
        print("  ✅ 所有检查都通过!")
    
    # 资源估计
    print(f"\n📈 资源估计:")
    print(f"  • 相对显存占用: {resources['memory_percentage']}")
    print(f"  • 估计GPU显存: {resources['estimated_gpu_memory_gb']}")
    print(f"  • 相对训练速度: {resources['speed_relative']}")
    print(f"  • 估计参数量: {resources['total_params_millions']:.1f}M")
    
    print("\n" + "="*80)

def compare_configs(config1_path: str, config2_path: str):
    """对比两个配置"""
    config1 = load_config(config1_path)
    config2 = load_config(config2_path)
    
    print("\n" + "="*80)
    print("📊 配置对比报告".center(80))
    print("="*80)
    
    print(f"\n配置1: {config1_path}")
    print_config_summary(config1)
    
    print(f"\n配置2: {config2_path}")
    print_config_summary(config2)
    
    # 差异分析
    print("\n📊 差异分析:")
    policy1 = config1.get('policy', {})
    policy2 = config2.get('policy', {})
    bev1 = config1.get('bev_encoder', {})
    bev2 = config2.get('bev_encoder', {})
    
    diff_items = [
        ('n_emb', policy1.get('n_emb'), policy2.get('n_emb')),
        ('n_head', policy1.get('n_head'), policy2.get('n_head')),
        ('n_layer', policy1.get('n_layer'), policy2.get('n_layer')),
        ('n_cond_layers', policy1.get('n_cond_layers'), policy2.get('n_cond_layers')),
        ('feature_dim', bev1.get('feature_dim'), bev2.get('feature_dim')),
    ]
    
    for name, val1, val2 in diff_items:
        if val1 != val2:
            change = ((val2 - val1) / val1 * 100) if val1 != 0 else 0
            symbol = "⬆️ " if change > 0 else "⬇️ "
            print(f"  {symbol} {name}: {val1} → {val2} ({change:+.0f}%)")
    
    # 资源对比
    res1 = estimate_memory_and_speed(config1)
    res2 = estimate_memory_and_speed(config2)
    
    print(f"\n📈 资源对比:")
    print(f"  内存占用: {res1['memory_percentage']} → {res2['memory_percentage']}")
    print(f"  训练速度: {res1['speed_relative']} → {res2['speed_relative']}")
    
    print("\n" + "="*80)

def print_config_summary(config: Dict):
    """打印配置摘要"""
    policy = config.get('policy', {})
    bev_encoder = config.get('bev_encoder', {})
    resources = estimate_memory_and_speed(config)
    
    print(f"  • n_emb: {policy.get('n_emb', 512)}")
    print(f"  • n_head: {policy.get('n_head', 8)}")
    print(f"  • n_layer: {policy.get('n_layer', 8)}")
    print(f"  • n_cond_layers: {policy.get('n_cond_layers', 4)}")
    print(f"  • feature_dim: {bev_encoder.get('feature_dim', 256)}")
    print(f"  • 显存: {resources['memory_percentage']}")
    print(f"  • 速度: {resources['speed_relative']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='维度配置检查工具')
    parser.add_argument('--check', type=str, help='检查配置文件的兼容性')
    parser.add_argument('--compare', type=str, nargs=2, help='对比两个配置文件')
    parser.add_argument('--default', action='store_true', help='检查默认配置')
    
    args = parser.parse_args()
    
    if args.default:
        config_path = '/home/wang/Project/MoT-DP/config/nuscenes.yaml'
        print_compatibility_report(config_path)
    elif args.check:
        print_compatibility_report(args.check)
    elif args.compare:
        compare_configs(args.compare[0], args.compare[1])
    else:
        parser.print_help()
