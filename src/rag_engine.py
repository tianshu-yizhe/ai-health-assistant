"""
RAG 检索引擎
============
离线: 加载 knowledge/ 目录下的 txt → 切分 → 向量化 → 存入 Chroma（增量索引）
在线: 用户提问 → 向量化 → 搜 Chroma Top-K → 返回带来源标注的文档片段

向量模型: text-embedding-v4（阿里云百炼，0.0005元/千Token，有免费额度）
向量库: Chroma（本地轻量，自动持久化到 ./chroma_db）
"""

import json
import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
from src.llm_client import client, async_client  # 同步用于启动建库，异步用于在线检索

# ── 配置 ──
KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge")
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
EMBEDDING_MODEL = "text-embedding-v4"  # 当前最便宜，有免费额度
CHUNK_SIZE = 300   # 每段 300 字
CHUNK_OVERLAP = 50  # 段与段之间重叠 50 字，避免截断关键信息

# 检索相似度阈值（2026-08-17 实测校准）：
# 库内问题相似度最低 0.64，库外问题最高 0.51，取 0.6 卡在中间零误伤。
# 低于阈值的片段视为"知识库没有相关内容"，返回空列表 → 降级为普通问答，
# 避免模型参考不相关内容硬答（如用感冒资料回答鼻炎问题）。
SIM_THRESHOLD = 0.6

# 增量索引状态文件：{文件名: 修改时间}，只有新增/修改过的文件才重新索引
INDEX_META_PATH = os.path.join(CHROMA_DIR, "index_meta.json")

# 创建 Chroma 客户端（数据自动持久化到 chroma_db 目录；关闭匿名遥测）
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False),
)
collection = chroma_client.get_or_create_collection(name="medical_knowledge")


def _clean_knowledge_text(text: str) -> str:
    """清洗 Markdown 残留：行首 `* ` 列表项、行尾 `*` 脚注星号。
    源文件/存量索引都可能带（LLM 会把这些星号复制进回答），入库和检索返回两处都用。"""
    text = re.sub(r"^\s*\*\s*", "", text, flags=re.M)
    text = re.sub(r"\*\s*$", "", text, flags=re.M)
    return text


def _get_embedding(text: str) -> list[float]:
    """调通义 Embedding API，把文本转成向量"""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def build_knowledge_base():
    """
    增量建库（启动时执行）：
      - 新增的 txt → 切分索引
      - 修改过的 txt → 删除旧片段后重建
      - 未变动的文件 → 跳过（幂等、省 Embedding 费用）
    """
    docs = {}
    for filename in os.listdir(KNOWLEDGE_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(KNOWLEDGE_DIR, filename)
            docs[filename] = os.path.getmtime(filepath)

    if not docs:
        print("[RAG] knowledge/ 目录为空，请先放入 txt 文件。")
        return

    meta = {}
    if os.path.exists(INDEX_META_PATH):
        try:
            with open(INDEX_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    to_index = {f: t for f, t in docs.items() if meta.get(f) != t}
    if not to_index:
        existing = collection.get()
        print(f"[RAG] 索引已是最新，共 {len(existing['ids'])} 个片段。跳过建库。")
        return

    existing_ids = collection.get()["ids"]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )

    for filename, mtime in to_index.items():
        # 内容变更过的文件：先删旧片段再重建
        old_ids = [i for i in existing_ids if i.rsplit("_", 1)[0] == filename]
        if old_ids:
            collection.delete(ids=old_ids)
            print(f"[RAG] 删除旧索引: {filename} ({len(old_ids)} 个片段)")

        with open(os.path.join(KNOWLEDGE_DIR, filename), "r", encoding="utf-8") as f:
            content = f.read()
        chunks = splitter.split_text(_clean_knowledge_text(content))
        if not chunks:
            continue
        embeddings = [_get_embedding(chunk) for chunk in chunks]
        collection.add(
            ids=[f"{filename}_{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
        )
        meta[filename] = mtime
        print(f"[RAG] 已索引: {filename} → {len(chunks)} 个片段")

    with open(INDEX_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"[RAG] 增量建库完成，本次处理 {len(to_index)} 个文件。")


async def search(question: str, top_k: int = 3) -> list[dict]:
    """
    在线检索：把用户问题向量化，在 Chroma 中找最相似的 top_k 个文档片段，
    并用相似度阈值过滤无关片段。
    返回 [{"source": 来源文件名, "text": 片段内容}, ...]，供 Prompt 引用标注。
    知识库没有相关内容时返回空列表 → 上层降级为普通问答。
    使用异步客户端，不阻塞事件循环。
    异常降级：Chroma 或 Embedding API 挂了 → 返回空列表，走普通聊天，不 500。
    """
    try:
        existing = collection.get()
        if not existing["ids"]:
            return []

        # 异步 Embedding，不阻塞
        resp = await async_client.embeddings.create(model=EMBEDDING_MODEL, input=question)
        q_embedding = resp.data[0].embedding
        results = collection.query(query_embeddings=[q_embedding], n_results=top_k)
        docs = results["documents"][0]
        ids = results["ids"][0]
        distances = results["distances"][0]

        kept = []
        for d, i, dist in zip(docs, ids, distances):
            sim = 1.0 / (1.0 + dist)   # L2 距离 → 0~1 相似度
            if sim >= SIM_THRESHOLD:
                kept.append({"source": i.rsplit("_", 1)[0].removesuffix(".txt"), "text": _clean_knowledge_text(d)})
        return kept
    except Exception:
        return []  # 检索挂了 → 当没有知识库，降级为普通问答
