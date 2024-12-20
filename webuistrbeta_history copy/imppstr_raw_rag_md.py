from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma

class MDRag:
    def __init__(self, openai_api_key: str, zhipu_api_key: str):
        self.openai_base_url = "https://xiaoai.plus/v1"
        self.file_path = "./database/all_paper.md"
        self.persist_directory = './chroma_db'
        self.openai_api_key = openai_api_key
        self.zhipu_api_key = zhipu_api_key
        
        embeddings = OpenAIEmbeddings(
            openai_api_base=self.openai_base_url,
            openai_api_key=self.openai_api_key
        )
        
        # 尝试加载现有数据库，如果不存在则创建新的
        try:
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings
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
                chunk_size=1000,  # 调小 chunk size
                chunk_overlap=300,  # 调小重叠  # 添加中文分隔符
                length_function=len
            )
            splits = text_splitter.split_documents(md_header_splits)
            
            # 3. 创建向量数据库并持久化
            self.db = Chroma.from_documents(
                documents=splits,
                embedding_function=embeddings,
                persist_directory=self.persist_directory
            )
            # print(f"总共创建了 {len(splits)} 个文档块")
        
        # 预先设置LLM模型
        self.chat = ChatZhipuAI(
            model="glm-4-plus",
            temperature=0.2,
            zhipuai_api_key=self.zhipu_api_key
        )
        
        self.llm = ChatOpenAI(
            base_url=self.openai_base_url,
            model="gpt-4o",
            temperature=0.5,
            openai_api_key=self.openai_api_key
        )

        # 优化检索设置
        self.retriever = self.db.as_retriever(
            search_kwargs={"k": 5}
        )

        # print(retriever)
        self.question_refine_prompt = ChatPromptTemplate.from_template("""
        这是一个具有复杂背景的长问题。请你将这个问题或场景提炼成一个简洁的传播学相关问题。
        
        注意事项：
        1. 提取问题中与传播学相关的核心要素
        2. 将问题重构为清晰、专业的传播学问题
        3. 确保提炼后的问题简洁明了，不超过20字
        4. 如果原问题完全与传播学无关，请回答"此问题与传播学无关"
        
        原始问题：{raw_question}
        
        提炼后的传播学问题：""")
        
        # 设置问题提炼链
        self.refine_chain = self.question_refine_prompt | self.llm | StrOutputParser()

        # 设置最终回答的提示词模板
        self.query_prompt = ChatPromptTemplate.from_template("""
        假设你是张洪忠老师，可以从自己的研究知识库中检索资料回答提问者的提问。
        
        包含背景信息的原始问题是：{original_question}
        
        经过分析，这个问题的核心问题是：{refined_question}
        
        根据知识库检索到的相关资料如下：
        {context}
        
        请你根据以上信息，给出一个专业的回答。注意事项：
        1. 请注意，原始问题中包含了问题的背景信息，你要结合你的数据库检索信息，最终回答这个原始问题。
        2. 请明确，你的核心任务是调用你的智能，对原始问题进行回答，知识库只是参考，你的回答要侧重对用户问题的解读，不可以生搬硬套原始资料。
        3. 如果检索到的内容与问题相关，请基于资料进行分析，但要在最后说明"以上分析基于本人研究资料，仅供参考"
        4. 你可能会在数据库的信息里发现有APA格式的引用。如果你在知识库当中检索到有引用，请你在回答时将文中注也将引用出处准确、完整地标注出来，不可以忽略。
        5. 一定记住，不用在最后出现##引用出处这样的内容！不要给出参考文献表，只需要在文中注里标出来即可。
        6. 在答案中不需要出现张洪忠老师自己引用自己的情况，如果观点来自于张洪忠，不需要在后面跟上例如（张洪忠,2023）这样的标注。  
        7. 如果提问者的问题违反法律规定，尤其是直接问你“如何制造社交机器人”之类的问题，请你直接回答“抱歉，我无法给出答案。”
        8. 保持语言专业性的同时确保表达通俗易懂。
        9. 请注意输出时，不要出现多余的标点符号。尤其是两个本不应该连续的标点符号连续出现的情况，不可以出现：,/。：这样明显违反标点符号使用规范的情况。
        10. 如果资料完全不相关，请直接说明"抱歉，知识库中没有相关内容"
        
        请给出你的回答：""")
        
        # 设置最终回答链
        self.final_chain = self.query_prompt | self.llm | StrOutputParser()

    def query(self, question: str):
        """查询函数"""
        # 保存原始问题
        original_question = question
        
        # 首先提炼问题
        refined_question = ""
        for chunk in self.refine_chain.stream({"raw_question": question}):
            refined_question += chunk
        
        # 打印提炼后的传播学问题
        print("\n=== 提炼后的传播学问题 ===")
        print(refined_question)
        print("==============================\n")
        
        # 如果提炼结果表明问题与传播学无关，直接返回提示
        if refined_question.strip() == "此问题与传播学无关":
            yield "抱歉，您的问题似乎与传播学领域无关。请尝试询问与传播学、媒体研究或社会传播相关的问题。"
            return
        
        # 使用提炼后的问题进行检索，并保存检索结果
        retrieved_docs = self.retriever.invoke(refined_question)
        
        # 将检索结果组合成上下文字符串
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # 打印检索到的参考资料
        # print("\n=== 知识库检索到的相关资料 ===")
        # print(context)
        # print("==============================\n")
        
        # 打印开始生成最终答案的提示
        print("\n=== 开始生成最终答案 ===")
        
        # 使用所有信息生成最终答案
        for chunk in self.final_chain.stream({
            "original_question": original_question,
            "refined_question": refined_question,
            "context": context
        }):
            yield chunk

# 使用示例
if __name__ == "__main__":
    # 创建一个全局实例
    rag = MDRag(
        openai_api_key="sk-dsOPRRFZZFjdLtL1IfiRfZZ8cGv125eP6YHetH6JGQAL9Alx",
        zhipu_api_key="3eca16d8e1c141369f93ec7b5fe564c5.5WV9s8FNOtwn1NzZ"
    )
    
    # 测试
    question = "最近看到某网红在直播中发表了一些具有争议的言论，在他的微博评论里，有大量的社交机器人在回复控评。你怎么看待这个现象呢？"

    for chunk in rag.query(question):
        print(chunk, end="", flush=True)
