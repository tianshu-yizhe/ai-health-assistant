"""
日志模块
========
统一日志出口，控制台 + 文件双写，按天滚动。

用法:
  from src.logger import logger
  logger.info("用户提问: %s", question)
  logger.warning("安全拦截: %s", question)
  logger.error("LLM调用失败: %s", e)

日志文件: logs/app.log（按天自动滚动 logs/app-2026-08-01.log）
"""

import logging
import os
from logging.handlers import TimedRotatingFileHandler

# 日志目录
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 统一格式: 时间 | 级别 | 模块 | 消息
FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 创建 logger（名字固定，避免重复创建）
logger = logging.getLogger("ai-agent")
logger.setLevel(logging.INFO)
if not logger.handlers:  # 防止重复添加 handler
    # 控制台输出（systemd 日志/终端）
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
    logger.addHandler(console)

    # 文件输出，每天滚动，保留 7 天
    file_handler = TimedRotatingFileHandler(
        os.path.join(LOG_DIR, "app.log"),
        when="midnight",
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
    logger.addHandler(file_handler)
