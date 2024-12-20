from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
import json
from langchain_core.documents import Document

class HistoryAwareRAG_simi:
    def __init__(self, openai_api_key):
        self.openai_api_base = "https://xiaoai.plus/v1"
        self.openai_api_key = openai_api_key
        self.persist_directory = "./faiss_index"
        self.json_file_path = "./database/all_qa.json"
        
        # 初始化 LLM
        self.llm = ChatOpenAI(
            base_url=self.openai_api_base,
            model="gpt-4o",
            api_key=self.openai_api_key,
            temperature=0,
            streaming=True
        )
        
        # 初始化 embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_base=self.openai_api_base,
            openai_api_key=self.openai_api_key
        )
        
        # 初始化聊天历史存储
        self.store = {}
    
    def _init_vectorstore(self):
        """初始化向量数据库"""

        self.vectorstore = FAISS.load_local(
            self.persist_directory,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def _setup_chain(self):
        """设置RAG chain"""
        if not hasattr(self, 'conversational_rag_chain'):
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 8}
            )
            
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", """
                基于聊天历史和最新的问题，生成一个独立的搜索查询。
                如果问题提到了"刚刚"、"之前"等词，请查看历史记录并将相关上下文整合到查询中。
                如果是全新的问题，直接返回原问题即可。
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])
            
            self.history_aware_retriever = create_history_aware_retriever(
                self.llm, retriever, contextualize_q_prompt
            )
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", """
                #假设你是张洪忠老师，下面是一些你过往的研究资料，你可以从这些资料里提取有用的部分来回答提问者的提问。
                资料库是{context}。
                #在回答时，你需要注意以下注意事项：
                1.在回答时，你需要完整读取研究资料，分析里面的逻辑关系，结构要清晰，逻辑要严谨，不能出现遗漏的情况。
                2.参考资料都是QA对，但是它们的存放顺序不一定按照逻辑顺序排列。你在回答时要注意这一点，不能出现框架上有遗漏的地方，要完整读取所有资料，并按照逻辑关系组织答案。
                3.在回答时，如果材料里没有提到提问者的问题,一定不能自己编造，要严格按照你检索的结果来回答。
                4.在回答时，你的语言风格要尽可能地忠于原文，但需要确保逻辑通顺。
                5.请直接输出处理后的回答内容，不要包含任何原始文档的元数据信息。
                6.请你输出时，保持文本生成的连贯性，不能有断断续续的情况，前后逻辑要清晰。
                7.请注意输出时，不要出现多余的标点符号。尤其是两个本不应该连续的标点符号连续出现的情况，不可以出现：,。：这样明显违反标点符号使用规范的情况。
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])
            
            question_answer_chain = create_stuff_documents_chain(
                self.llm, 
                qa_prompt,
            )
            
            rag_chain = create_retrieval_chain(
                self.history_aware_retriever, 
                question_answer_chain,
            )
            
            self.conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
    
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """获取或创建会话历史"""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    def get_chat_history(self, session_id: str = "default"):
        """获取指定会话的聊天历史"""
        if session_id in self.store:
            return self.store[session_id].messages
        return []
    
    def query(self, question: str, session_id: str = "default"):
        """流式查询接口"""
        try:
            # 确保向量数据库已初始化
            self._init_vectorstore()
            
            # 首先检查相似度
            results = self.vectorstore.similarity_search_with_score(question, k=8)
            
            # 检查最相关的结果是否高于阈值
            if results and results[0][1] > 0.3:
                yield "none"
                return
            
            # 过滤出相似度小于阈值的结果
            filtered_results = [
                doc for doc, score in results 
                if score < 0.3
            ]
            
            if not filtered_results:
                yield "none"
                return
            
            print("符合条件的检索结果：")
            for doc, score in results:
                print(f"相似度分数: {score}")
                print(f"找到的内容: {doc.page_content}\n")
            
            # 如果相似度检查通过，继续处理
            if not hasattr(self, 'conversational_rag_chain'):
                self._setup_chain()
                
            response_stream = self.conversational_rag_chain.stream(
                {"input": question},
                config={
                    "configurable": {"session_id": session_id}
                }
            )
            
            for chunk in response_stream:
                if "answer" in chunk:
                    if hasattr(chunk["answer"], 'content'):
                        yield chunk["answer"].content
                    else:
                        yield chunk["answer"]
                        
        except Exception as e:
            print(f"查询出错: {str(e)}")
            yield "抱歉，处理您的问题时出现错误。"
    

    

# 使用示例
if __name__ == "__main__":
    rag = HistoryAwareRAG_simi(openai_api_key="sk-dsOPRRFZZFjdLtL1IfiRfZZ8cGv125eP6YHetH6JGQAL9Alx")
    
    # 使用同一个session_id来维持对话连续性

    session_id = "test_session"
    
    Q1="请介绍一下你自己"
    print(Q1)
    for chunk in rag.query(Q1, session_id=session_id):
        print(chunk, end="", flush=True)
    print("\n")
    
    # 第二个问题

    # Q2="它对信息传播有什么影响？"
    # print(Q2)
    # for chunk in rag.query(Q2, session_id=session_id):
    #     print(chunk, end="", flush=True)
    # print("\n")