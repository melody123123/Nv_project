import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.schema import Document
from ....xzs_config import API_KEY
from typing import List, Optional, Dict, Any
class RAGVectorDB:
    def __init__(self, persist_directory: str = "./vectordb"):
        """
        初始化RAG向量数据库
        
        参数:
            persist_directory: 向量数据库持久化目录
        """
        self.persist_directory = persist_directory
        self.embedding_function = self._setup_aliyun_embedding()
        self.vectorstore = None
        
    def _setup_aliyun_embedding(self):
        """
        设置阿里百炼Embedding模型
        """
        return DashScopeEmbeddings(
            model="text-embedding-v2",  # 阿里百炼的Embedding模型
            dashscope_api_key=API_KEY
            )
    
    def create_vectorstore(self, documents: Optional[List[Document]] = None):
        """
        创建或加载向量数据库
        
        参数:
            documents: 文档列表，如果为None则加载现有数据库
        """
        if documents and len(documents) > 0:
            # 创建新的向量数据库
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_function,
                persist_directory=self.persist_directory
            )
            print(f"创建向量数据库，添加了 {len(documents)} 个文档")
        else:
            # 加载现有数据库
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )
            print("加载现有向量数据库")
    
    def add_single_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        向向量数据库添加单条文档
        
        参数:
            text: 文本内容
            metadata: 元数据字典
            
        返回:
            文档ID
        """
        if not self.vectorstore:
            self.create_vectorstore()
        
        if metadata is None:
            metadata = {}
            
        document = Document(page_content=text, metadata=metadata)
        doc_ids = self.vectorstore.add_documents([document])
        self.vectorstore.persist()
        
        print(f"成功添加单条文档，ID: {doc_ids[0]}")
        return doc_ids[0]
    
    def add_multiple_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        向向量数据库批量添加文档
        
        参数:
            texts: 文本内容列表
            metadatas: 元数据字典列表
            
        返回:
            文档ID列表
        """
        if not self.vectorstore:
            self.create_vectorstore()
        
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]
        elif len(metadatas) != len(texts):
            raise ValueError("metadatas长度必须与texts相同")
        
        documents = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(texts, metadatas)
        ]
        
        doc_ids = self.vectorstore.add_documents(documents)
        self.vectorstore.persist()
        
        print(f"成功批量添加 {len(doc_ids)} 个文档")
        return doc_ids
    
    def similarity_search(self, query: str, k: int = 3, filter_dict: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        相似性搜索
        
        参数:
            query: 查询文本
            k: 返回结果数量
            filter_dict: 过滤条件
            
        返回:
            相似文档列表
        """
        if not self.vectorstore:
            raise ValueError("向量数据库未初始化")
        
        if filter_dict:
            results = self.vectorstore.similarity_search(
                query=query, 
                k=k, 
                filter=filter_dict
            )
        else:
            results = self.vectorstore.similarity_search(query=query, k=k)
        
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 3) -> List[tuple]:
        """
        带相似度得分的搜索
        
        参数:
            query: 查询文本
            k: 返回结果数量
            
        返回:
            (文档, 相似度得分) 元组列表
        """
        if not self.vectorstore:
            raise ValueError("向量数据库未初始化")
        
        return self.vectorstore.similarity_search_with_score(query=query, k=k)
    
    def get_document_count(self) -> int:
        """
        获取向量数据库中的文档数量
        
        返回:
            文档数量
        """
        if not self.vectorstore:
            return 0
        
        return self.vectorstore._collection.count()
    
    def delete_document(self, doc_id: str):
        """
        删除指定文档
        
        参数:
            doc_id: 文档ID
        """
        if not self.vectorstore:
            raise ValueError("向量数据库未初始化")
        
        self.vectorstore.delete(ids=[doc_id])
        self.vectorstore.persist()
        print(f"已删除文档 ID: {doc_id}")
    
    def delete_documents_by_filter(self, filter_dict: Dict[str, Any]):
        """
        根据过滤条件删除文档
        
        参数:
            filter_dict: 过滤条件字典
        """
        if not self.vectorstore:
            raise ValueError("向量数据库未初始化")
        
        self.vectorstore.delete(filter=filter_dict)
        self.vectorstore.persist()
        print(f"已根据条件 {filter_dict} 删除文档")


# 使用示例
def main():
    # 初始化环境变量（实际使用时请设置正确的API密钥）
    os.environ["ALIYUN_API_KEY"] = "your_aliyun_api_key"
    os.environ["ALIYUN_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    # 创建RAG向量数据库实例
    rag_db = RAGVectorDB(persist_directory="./my_vectordb")
    
    # 示例文档
    
    # 按条添加文档
    doc_id = rag_db.add_single_document(
        text="NULL",                                                                                                                                                                                                                                                                                                                                                                                                                       
        metadata={"source": "local", "type": "agent"}
    )
    
    
    # 执行相似性搜索
    print("=== 相似性搜索示例 ===")
    query = "什么是LangChain？"
    results = rag_db.similarity_search(query, k=2)
    
    
    print(f"查询: {query}")
    for i, doc in enumerate(results):
        print(f"结果 {i+1}: {doc.page_content}")
        print(f"元数据: {doc.metadata}")
        print("-" * 50)
    
    # 带过滤的搜索
    print("\n=== 带过滤的搜索示例 ===")
    filtered_results = rag_db.similarity_search(
        query="数据库", 
        k=2, 
        filter_dict={"type": "database"}
    )
    
    print("过滤条件: type=database")
    for i, doc in enumerate(filtered_results):
        print(f"结果 {i+1}: {doc.page_content}")
        print(f"元数据: {doc.metadata}")
        print("-" * 50)
    
    # 带得分的搜索
    print("\n=== 带相似度得分的搜索 ===")
    scored_results = rag_db.similarity_search_with_score("向量检索", k=2)
    
    for i, (doc, score) in enumerate(scored_results):
        print(f"结果 {i+1} (相似度: {score:.4f}): {doc.page_content}")
        print("-" * 50)
    
    # 查看文档数量
    print(f"\n向量数据库中的文档总数: {rag_db.get_document_count()}")
    
    # 删除文档示例
    # rag_db.delete_document(doc_id)
    # rag_db.delete_documents_by_filter({"source": "batch"})

if __name__ == "__main__":
    main()