"""
Prompt 管理器
============
负责：读配置 → 取模板 → Jinja2 渲染 → 返回最终 Prompt 文本
为什么独立成文件：以后可能加多种 Prompt 构建策略（拼 RAG 结果、拼历史对话）
"""

from jinja2 import Template
from .config import load_config


def build_system_prompt(scene: str = "medical_assistant") -> str:
    """
    从 YAML 配置构建最终的 System Prompt 文本。
    """
    cfg = load_config(scene)

    # Jinja2 渲染：把 {{ name }} 等占位符替换成实际值
    template = Template(cfg["system_template"])
    return template.render(
        name=cfg.get("name", ""),
        role=cfg.get("role", ""),
        max_words=cfg.get("max_words", 200),
        refuse_msg=cfg.get("refuse_msg", "抱歉，我无法回答这个问题。"),
        emergency_msg=cfg.get("emergency_msg", "请立即就医。"),
        unclear_msg=cfg.get("unclear_msg", "抱歉，我没太理解您的问题。"),
    )


def get_model(scene: str = "medical_assistant") -> str:
    """获取场景对应的模型名"""
    cfg = load_config(scene)
    return cfg.get("model", "qwen-turbo")
