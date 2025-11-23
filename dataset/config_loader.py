import os
import yaml
from types import SimpleNamespace


def find_project_root(start_path=None):
    if start_path is None:
        start_path = os.path.dirname(os.path.abspath(__file__))
    cur = start_path
    for _ in range(6):
        if os.path.exists(os.path.join(cur, 'config')):
            return cur
        cur = os.path.dirname(cur)
    return start_path


def load_dataset_yaml(yaml_path=None):
    if yaml_path is None:
        project_root = find_project_root()
        yaml_path = os.path.join(project_root, 'config', 'dataset_mini.yaml')

    if not os.path.exists(yaml_path):
        return {}

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f) or {}
    return data


def merge_dicts(base, override):
    # shallow merge
    result = dict(base or {})
    for k, v in (override or {}).items():
        if v is not None:
            result[k] = v
    return result


def load_config(section, cli_kwargs=None, yaml_path=None, defaults=None):
    """
    Load configuration for a given section from dataset.yaml, then overlay defaults and CLI kwargs.

    Order of precedence (low -> high): defaults < yaml < cli_kwargs (when not None)

    Returns a dict.
    """
    defaults = defaults or {}
    yaml_data = load_dataset_yaml(yaml_path)
    yaml_section = yaml_data.get(section, {}) if isinstance(yaml_data, dict) else {}

    cfg = dict(defaults)
    cfg.update(yaml_section or {})

    if cli_kwargs:
        for k, v in cli_kwargs.items():
            if v is not None:
                cfg[k] = v

    return cfg


def to_namespace(d):
    return SimpleNamespace(**(d or {}))
