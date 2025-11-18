#!/usr/bin/env python3
"""
快速维度调整脚本
提供预定义的配置方案，快速切换不同的维度设置
"""

import yaml
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / 'config'

# 预定义的维度方案
DIMENSION_SCHEMES = {
    'baseline': {
        'name': '基准配置 (原始)',
        'policy': {
            'n_emb': 512,
            'n_head': 8,
            'n_layer': 8,
            'n_cond_layers': 4,
        },
        'bev_encoder': {
            'feature_dim': 256,
        },
        'notes': '当前生产配置',
    },
    
    'conservative': {
        'name': '保守提升 (低资源压力)',
        'policy': {
            'n_emb': 768,
            'n_head': 12,
            'n_layer': 8,
            'n_cond_layers': 6,
        },
        'bev_encoder': {
            'feature_dim': 384,
        },
        'notes': '精度提升: ~5-10% | 内存增加: ~30% | 速度: 1.2x',
    },
    
    'balanced': {
        'name': '平衡方案 (推荐)',
        'policy': {
            'n_emb': 1024,
            'n_head': 16,
            'n_layer': 12,
            'n_cond_layers': 6,
        },
        'bev_encoder': {
            'feature_dim': 512,
        },
        'notes': '精度提升: ~15-25% | 内存增加: ~100% | 速度: 1.5-2x',
    },
    
    'aggressive': {
        'name': '激进提升 (高精度)',
        'policy': {
            'n_emb': 1536,
            'n_head': 24,
            'n_layer': 16,
            'n_cond_layers': 8,
        },
        'bev_encoder': {
            'feature_dim': 768,
        },
        'notes': '精度提升: ~25-40% | 内存增加: ~200%+ | 速度: 2-3x',
    },
    
    'ultra': {
        'name': '超高精度 (GPU显存充足)',
        'policy': {
            'n_emb': 2048,
            'n_head': 32,
            'n_layer': 24,
            'n_cond_layers': 8,
        },
        'bev_encoder': {
            'feature_dim': 1024,
        },
        'notes': '精度提升: ~40-50% | 内存: 300%+ | 速度: 3-4x',
    },

    'lightweight': {
        'name': '轻量配置 (GPU显存不足)',
        'policy': {
            'n_emb': 256,
            'n_head': 4,
            'n_layer': 4,
            'n_cond_layers': 2,
        },
        'bev_encoder': {
            'feature_dim': 128,
        },
        'notes': '精度降低但速度快 | 内存减少: ~50% | 速度: 0.5x',
    },
}

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config, config_path):
    """保存配置文件"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def apply_scheme(config, scheme):
    """应用维度方案"""
    if 'policy' in scheme:
        for key, value in scheme['policy'].items():
            config['policy'][key] = value
    
    if 'bev_encoder' in scheme:
        for key, value in scheme['bev_encoder'].items():
            config['bev_encoder'][key] = value
    
    return config

def list_schemes():
    """列出所有可用的方案"""
    print("\n" + "="*80)
    print("📋 可用的维度调整方案".center(80))
    print("="*80)
    
    for idx, (name, scheme) in enumerate(DIMENSION_SCHEMES.items(), 1):
        print(f"\n{idx}️⃣  {scheme['name']}")
        print(f"   ID: {name}")
        print(f"   n_emb: {scheme['policy']['n_emb']:<6} | "
              f"n_head: {scheme['policy']['n_head']:<6} | "
              f"n_layer: {scheme['policy']['n_layer']:<6} | "
              f"n_cond_layers: {scheme['policy']['n_cond_layers']:<6} | "
              f"feature_dim: {scheme['bev_encoder']['feature_dim']}")
        print(f"   💡 {scheme['notes']}")
    
    print("\n" + "="*80)

def apply_scheme_interactive():
    """交互式应用方案"""
    list_schemes()
    
    while True:
        choice = input("\n请选择方案 (输入ID或序号, q=退出): ").strip().lower()
        
        if choice == 'q':
            print("已退出")
            return
        
        # 尝试通过序号选择
        try:
            idx = int(choice) - 1
            scheme_names = list(DIMENSION_SCHEMES.keys())
            if 0 <= idx < len(scheme_names):
                choice = scheme_names[idx]
            else:
                print(f"❌ 序号范围1-{len(scheme_names)}")
                continue
        except ValueError:
            pass
        
        if choice not in DIMENSION_SCHEMES:
            print(f"❌ 无效的选择: {choice}")
            continue
        
        scheme = DIMENSION_SCHEMES[choice]
        
        print(f"\n✅ 选中: {scheme['name']}")
        
        # 询问保存位置
        config_path = input("配置文件路径 (默认: config/nuscenes.yaml): ").strip()
        if not config_path:
            config_path = 'config/nuscenes.yaml'
        
        config_path = PROJECT_ROOT / config_path
        
        if not config_path.exists():
            print(f"❌ 文件不存在: {config_path}")
            continue
        
        # 备份原配置
        backup_path = config_path.with_stem(config_path.stem + '_backup')
        import shutil
        shutil.copy(config_path, backup_path)
        print(f"✅ 备份原配置: {backup_path}")
        
        # 加载并应用方案
        config = load_config(config_path)
        config = apply_scheme(config, scheme)
        
        # 保存
        save_config(config, config_path)
        print(f"✅ 配置已保存: {config_path}")
        
        # 打印新配置摘要
        print(f"\n📊 新配置摘要:")
        print(f"  • n_emb: {config['policy']['n_emb']}")
        print(f"  • n_head: {config['policy']['n_head']}")
        print(f"  • n_layer: {config['policy']['n_layer']}")
        print(f"  • n_cond_layers: {config['policy']['n_cond_layers']}")
        print(f"  • feature_dim: {config['bev_encoder']['feature_dim']}")
        
        again = input("\n继续调整其他配置? (y/n): ").strip().lower()
        if again != 'y':
            break

def apply_scheme_cli(scheme_name, config_path):
    """命令行应用方案"""
    if scheme_name not in DIMENSION_SCHEMES:
        print(f"❌ 未知的方案: {scheme_name}")
        print(f"   可用方案: {', '.join(DIMENSION_SCHEMES.keys())}")
        return False
    
    scheme = DIMENSION_SCHEMES[scheme_name]
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"❌ 文件不存在: {config_path}")
        return False
    
    # 备份
    backup_path = config_path.with_stem(config_path.stem + '_backup')
    if not backup_path.exists():
        import shutil
        shutil.copy(config_path, backup_path)
        print(f"✅ 备份原配置: {backup_path}")
    
    # 应用方案
    config = load_config(config_path)
    config = apply_scheme(config, scheme)
    save_config(config, config_path)
    
    print(f"✅ 已应用方案: {scheme['name']}")
    print(f"✅ 配置已保存: {config_path}")
    print(f"   notes: {scheme['notes']}")
    return True

if __name__ == '__main__':
    if len(sys.argv) == 1:
        # 交互模式
        apply_scheme_interactive()
    elif len(sys.argv) == 2:
        if sys.argv[1] == '--list':
            list_schemes()
        else:
            print("使用方法:")
            print("  python utils/adjust_dimensions.py                          # 交互模式")
            print("  python utils/adjust_dimensions.py --list                   # 列出所有方案")
            print("  python utils/adjust_dimensions.py <scheme> <config_path>   # CLI模式")
            print("\n例子:")
            print("  python utils/adjust_dimensions.py balanced config/nuscenes.yaml")
            print("  python utils/adjust_dimensions.py conservative config/nuscenes.yaml")
    elif len(sys.argv) == 3:
        # CLI模式
        scheme_name = sys.argv[1]
        config_path = sys.argv[2]
        apply_scheme_cli(scheme_name, config_path)
    else:
        print("参数错误")
        print("使用 --list 查看帮助")
