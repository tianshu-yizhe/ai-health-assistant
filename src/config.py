"""
配置加载器
==========
从 YAML 文件读取应用配置（场景名、提示词等），
不是放 Key（Key 在 .env 里）。
"""

import os
import yaml

# 配置文件目录
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")


def load_config(scene: str) -> dict:
    """加载指定场景的配置"""
    file_path = os.path.join(CONFIG_DIR, f"{scene}.yml")
    with open(file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config[scene]
