# 一折 · AI 健康助手（健康知识检索问答系统）

面向健康科普场景的 RAG 检索问答服务，支持智能导诊、医学科普、健康预防三类问答。
已上线公网运行，本目录为完整源码（与线上版本同步），可用于本地运行学习。

## 项目结构

```
openai/
├── main.py                  # FastAPI 入口：路由、IP 限流中间件、流式问答、健康检查、用量统计
├── .env                     # DASHSCOPE_API_KEY（不提交 Git）
├── prompts/
│   └── medical_assistant.yml  # Prompt 配置：系统提示词、模型名（qwen-turbo）、能力边界
├── src/
│   ├── llm_client.py        # 通义千问客户端（同步+异步）、token 用量按天统计
│   ├── rag_engine.py        # RAG：文档切分 → Embedding → ChromaDB 增量索引 → 相似度检索（带来源标注）
│   ├── safety.py            # 6 层输入安全过滤 + 输出剂量过滤 + 流式增量过滤
│   ├── memory.py            # 多轮对话记忆（Redis + 24h TTL + 最多 10 轮裁剪）
│   ├── cache.py             # 问答缓存（问题归一化 MD5 key，15min TTL）
│   ├── prompt_manager.py    # YAML 配置加载 + Jinja2 模板渲染
│   ├── config.py            # 配置加载器
│   └── logger.py            # 日志（控制台 + 按天滚动文件，保留 7 天）
├── knowledge/               # 健康科普资料（txt，放入后重启自动增量索引）
├── static/                  # 前端静态资源（favicon 等）
├── tests/                   # 接口测试
└── chroma_db/               # 向量库持久化目录（自动生成）
```

## 本地运行

```bash
# 1. 安装依赖（建议用 .venv）
pip install -r requirements.txt

# 2. 确保 .env 里有 DASHSCOPE_API_KEY

# 3. 确保 Redis 可用（默认连 192.168.150.128，可用环境变量 REDIS_HOST 覆盖）

# 4. 启动
python main.py
# 或：uvicorn main:app --host 127.0.0.1 --port 8000

# 5. 打开 http://127.0.0.1:8000（聊天界面），接口文档见 /docs
```

## 核心接口

| 接口 | 说明 |
|------|------|
| POST /api/chat/rag/stream | RAG 流式问答（body: question, session_id 可选） |
| POST /api/chat/rag | RAG 问答（debug=true 返回检索来源） |
| POST /api/chat/memory | 多轮对话（带记忆） |
| POST /api/chat | 无记忆单轮问答 |
| GET /healthz | 健康检查 |
| GET /api/stats | 今日 token 用量 + 缓存命中率（公网已被 nginx 拦截，仅服务器本地可查） |
| POST /api/admin/login | 管理后台登录（ADMIN_USER/ADMIN_PASSWORD 在 .env，5 次失败锁 15 分钟） |
| GET /api/admin/snapshot | 运营数据快照（需 token）：请求/拦截/错误/限流计数、token 用量、7 天趋势、最近事件 |
| GET /admin | 管理后台页面（隐藏入口，无公开链接） |

## 关键设计（面试可讲）

1. **RAG 链路**：文档 300 字切分（50 重叠）→ text-embedding-v4 向量化 → ChromaDB Top-K 检索 → 拼入 Prompt → 通义千问生成，回答标注【来源：文件名】
2. **增量索引**：index_meta.json 记录文件修改时间，只重建新增/变更的文档
3. **多轮对话**：历史存 Redis（JSON + 24h TTL + 10 轮裁剪），多轮模式不走问答缓存（答案依赖上下文）
4. **安全分层**：输入 6 层拦截（心理危机/无效输入/绕过话术/求药/剂量/违禁）+ 输出剂量过滤 + 流式 15 字符尾部缓冲增量过滤
5. **降级设计**：Redis 挂→无缓存/无记忆继续服务；检索挂→退化为普通问答；LLM 超时→固定话术，核心接口不 500
6. **限流**：Redis 固定窗口 30 次/分钟/IP，nginx 反代下取 X-Real-IP 真实 IP
7. **可观测性**：metrics.py 按天埋点（请求/拦截/LLM 错误/限流）→ Redis，管理后台 /admin 登录后查看仪表盘；公网不暴露内部接口（/api/stats、debug=true 均被 nginx 404）
