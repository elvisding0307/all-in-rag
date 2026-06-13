import os
import sys

from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 配置全局嵌入模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

persist_path = "./llamaindex_index_store"


def build_index():
    """创建示例文档并构建索引"""
    texts = [
        "张三是法外狂徒",
        "LlamaIndex是一个用于构建和查询私有或领域特定数据的框架。",
        "它提供了数据连接、索引和查询接口等工具。"
    ]
    docs = [Document(text=t) for t in texts]

    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=persist_path)
    print(f"LlamaIndex 索引已保存至: {persist_path}")
    return index


def load_index():
    """从本地加载已保存的索引"""
    storage_context = StorageContext.from_defaults(persist_dir=persist_path)
    index = load_index_from_storage(storage_context)
    print(f"已从 {persist_path} 加载索引")
    return index


def get_or_build_index():
    """如果存在已保存的索引则加载，否则创建新索引"""
    if os.path.exists(persist_path):
        return load_index()
    return build_index()


def search(index, query: str, top_k: int = 2):
    """执行相似性搜索，返回最相关的文档"""
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return nodes


def main():
    index = get_or_build_index()

    # 命令行参数模式: python 03_llamaindex_vector.py "查询文本"
    if len(sys.argv) > 1:
        query = sys.argv[1]
        print(f"\n查询: {query}")
        results = search(index, query)
        print(f"\n返回 {len(results)} 条结果:\n")
        for i, node in enumerate(results, 1):
            print(f"[{i}] 相似度: {node.score:.4f}")
            print(f"    内容: {node.text}")
            print()
        return

    # 交互式模式
    print("\n=== LlamaIndex 向量搜索 ===")
    print("输入查询内容（或输入 'quit' 退出）:\n")
    while True:
        query = input("> ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        results = search(index, query)
        print(f"\n返回 {len(results)} 条结果:\n")
        for i, node in enumerate(results, 1):
            print(f"[{i}] 相似度: {node.score:.4f}")
            print(f"    内容: {node.text}")
            print()


if __name__ == "__main__":
    main()
