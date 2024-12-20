from his_simi_str import HistoryAwareRAG_simi
from his_md_str import HistoryAwareRAG_md
from langchain_community.chat_message_histories import ChatMessageHistory

class CombinedRAG:
    def __init__(self,
                 openai_base_url = "https://xiaoai.plus/v1",
                 openai_api_key= "sk-dsOPRRFZZFjdLtL1IfiRfZZ8cGv125eP6YHetH6JGQAL9Alx"
                 ):
        """
        初始化组合RAG系统
        """
        self.openai_base_url = openai_base_url
        self.openai_api_key = openai_api_key
        self.simi_rag = None
        self.md_rag = None
        # 将聊天历史存储在主类中统一管理
        self.store = {}

    def _sync_history(self, session_id: str):
        """同步历史记录到两个RAG实例"""
        if session_id in self.store:
            if self.simi_rag:
                self.simi_rag.store[session_id] = self.store[session_id]
            if self.md_rag:
                self.md_rag.store[session_id] = self.store[session_id]
    
    def _update_history(self, session_id: str, question: str, answer: str):
        """更新主存储中的历史记录"""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        self.store[session_id].add_user_message(question)
        self.store[session_id].add_ai_message(answer)

    def query(self, question: str, session_id: str = "default"):
        """修改后的查询函数"""
        # 同步历史记录到两个RAG实例
        self._sync_history(session_id)
        
        # 收集完整的回答
        full_answer = ""
        
        # 尝试使用相似度匹配RAG
        found_none = False
        if not self.simi_rag:
            self.simi_rag = HistoryAwareRAG_simi(openai_api_key=self.openai_api_key)
            
        for chunk in self.simi_rag.query(question, session_id=session_id):
            if chunk == "none":
                found_none = True
                break
            full_answer += chunk
            yield chunk
            
        if found_none:
            full_answer = ""  # 重置答案
            print("QA数据库里没有匹配答案，使用MD文档匹配...\n")
            
            if not self.md_rag:
                self.md_rag = HistoryAwareRAG_md(openai_api_key=self.openai_api_key)
            
            for chunk in self.md_rag.query(question, session_id=session_id):
                full_answer += chunk
                yield chunk
        
        # 更新主存储中的历史记录
        if full_answer:
            self._update_history(session_id, question, full_answer)

    def get_chat_history(self, session_id: str = "default"):
        """
        获取指定会话的聊天历史
        
        Args:
            session_id: 会话ID
            
        Returns:
            list: 聊天历史消息列表
        """
        # 优先从已使用的RAG实例获取历史记录
        if self.simi_rag and session_id in self.simi_rag.store:
            return self.simi_rag.get_chat_history(session_id)
        elif self.md_rag and session_id in self.md_rag.store:
            return self.md_rag.get_chat_history(session_id)
        return []

# 使用示例
if __name__ == "__main__":
    rag = CombinedRAG()
    session_id = "test_session"
    
    # 第一个问题
    question1 = "你怎么看待大模型的超级节点角色？"
    print("问题1:", question1)
    print("正在生成回答...\n")
    for chunk in rag.query(question1, session_id=session_id):
        print(chunk, end='', flush=True)
    print("\n")
    
    # 第二个问题（基于上下文）
    question2 = "它对中美关系有什么影响？"
    print("问题2:", question2)
    print("正在生成回答...\n")
    for chunk in rag.query(question2, session_id=session_id):
        print(chunk, end='', flush=True)
    print("\n")