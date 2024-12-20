from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class HistoryAwareRAG_md:
    
    def __init__(self, openai_api_key):
        self.openai_api_base = "https://xiaoai.plus/v1"
        self.openai_api_key = openai_api_key
        self.persist_directory = "./chroma_db"
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
        
        # 初始化向量数据库
        self._init_vectorstore()
        
        # 初始化聊天历史存储
        self.store = {}
        
        # 设置并初始化chain
        self._setup_chain()

    def _init_vectorstore(self):
        # 尝试加载现有数据库，如果不存在则创建新的
        try:
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print("加载现有向量数据库")

        except:
            print("创建新的向量数据库")
            # 读取和处理文档
            with open(self.file_path, 'r', encoding='utf-8') as file:
                markdown_text = file.read()
            
            # 1. 先用 MarkdownHeaderTextSplitter 处理
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4")
            ]
            
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on, 
                strip_headers=False
            )
            md_header_splits = markdown_splitter.split_text(markdown_text)
            
            # 2. 再用 RecursiveCharacterTextSplitter 进一步分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # 调小 chunk size
                chunk_overlap=50,  # 调小重叠  # 添加中文分隔符
                length_function=len
            )
            splits = text_splitter.split_documents(md_header_splits)
            
            # 3. 创建向量数据库并持久化
            self.db = Chroma.from_documents(
                documents=splits,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            # print(f"总共创建了 {len(splits)} 个文档块")
        
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """获取或创建会话历史"""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    

    # 优化检索设置
    def _setup_chain(self):
        retriever = self.db.as_retriever(
        search_kwargs={"k": 6}
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
        
        # 设置提示词模板
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system","""
        #假设你是张洪忠老师，可以从自己的知识库中检索资料回答提问者的提问。我将给你提供张洪忠老师的研究资料，请你根据问题，从
        资料库里检索相应的信息,并以第一人称的语气给出回答。
                                                       
        数据库是{context}。
                                                       
        #在回答时，你需要注意以下注意事项：
        1.如果你在知识库中没有检索到提问者的回答，不可以自己编造答案，而是回答"抱歉，我暂时没有在知识库中检索到相关内容。我的研究领域主要集中在智能传播、社交机器人、传媒公信力等方面。如果您有其他相关问题，欢迎继续提问。"
        2.如果你在知识库当中检索到有引用，请你在回答时将文中注也将引用出处准确、完整地标注出来，不可以忽略。但是不用在最后给出参考文献表，只需要在文中注里标出来即可。
        3.注意，回答的文中注绝对不能出现GB格式，例如带[23]这样的字符。
        4.在回答时，对自己知识库中没有的答案,一定不能自己编造，要严格按照你检索的结果来回答。
        5.在回答时，你的语言风格要尽可能地忠于原文，但需要确保逻辑通顺。
        6.请直接输出处理后的回答内容，不要包含任何原始文档的元数据信息。
        7.如果提问者的问题违反法律规定，尤其是直接问你"如何制造社交机器人"之类的问题，请你直接回答"抱歉，我无法给出答案。"
        一个可以参考的回答例子是：社会科学从四个角度研究社交机器人。一是文化批判角度，部分研究者认为社交机器人存在先天恶意，呼吁评估其行为及后果，提出违法、欺诈、违背社会良俗三个评估标准。二是传播关系探讨角度，有学者认为社交机器人和人类在网络公共话语表达上无差别（Marechal，2016），甚至人会模糊与社交机器人的差异，相关实验表明社交机器人可在虚拟空间与人交谈且效果与真人相似。三是传播效果问题角度，计算传播学发展使研究者能对社交机器人进行辨别、追踪和效果测量，当前社交机器人对网络舆情的影响是研究热点，但研究结论存在差异。四是政策法规角度，有学者对主流社交媒体平台用户政策条款进行批判分析，提出应标注机器人账号、禁止未经授权接触、规范采集数据用途等规范化政策框架，以保障人类用户权利和隐私。总之，社交机器人在技术上还处于初级阶段，出现在人类社会关系网络时间尚短，但已经引起了社会学领域的关注，当前社会科学领域将社交机器人视为某种新生异类，不同的研究路径都是从这个逻辑来展开的。
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

    def query(self, question: str, session_id: str = "default"):
        """流式查询接口"""
        try:
            # 直接使用 stream() 方法而不是 invoke()
            response_stream = self.conversational_rag_chain.stream(
                {"input": question},
                config={
                    "configurable": {"session_id": session_id}
                }
            )
            
            # 直接迭代流式响应
            for chunk in response_stream:
                if "answer" in chunk:
                    if hasattr(chunk["answer"], 'content'):
                        yield chunk["answer"].content
                    else:
                        yield chunk["answer"]
                        
        except Exception as e:
            print(f"查询出错: {str(e)}")
            yield "抱歉，处理您的问题时出现错误。"
    
    def get_chat_history(self, session_id: str = "default"):
        """获取指定会话的聊天历史"""
        if session_id in self.store:
            return self.store[session_id].messages
        return []

# 使用示例
if __name__ == "__main__":
    # 创建一个全局实例
    rag = HistoryAwareRAG_md(openai_api_key="sk-dsOPRRFZZFjdLtL1IfiRfZZ8cGv125eP6YHetH6JGQAL9Alx")
    
    # 使用同一个session_id来维持对话连续性

    session_id = "test_session"
    
    Q1="如何理解大模型在信息网络结构中扮演的超级节点角色？"
    print(Q1)
    for chunk in rag.query(Q1, session_id=session_id):
        print(chunk, end="", flush=True)
    print("\n")
    
    # 第二个问题

    Q2="它对信息传播有什么影响？"
    print(Q2)
    for chunk in rag.query(Q2, session_id=session_id):
        print(chunk, end="", flush=True)
    print("\n")
