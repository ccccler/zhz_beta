from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from his_combine_str import CombinedRAG
from imppstr_raw_rag_md import MDRag
import os
import json
import random

# 获取当前文件所在的目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__,
    static_folder='static',  # 不需要包含 webui
    static_url_path='/static'
)

CORS(app)

# 可以添加一个调试输出
print(f"Static folder path: {app.static_folder}")

# 初始化RAG系统
rag = CombinedRAG(
    openai_base_url="https://xiaoai.plus/v1",
    openai_api_key="sk-dsOPRRFZZFjdLtL1IfiRfZZ8cGv125eP6YHetH6JGQAL9Alx",
)

# 初始化 MDRag 实例
md_rag = MDRag(
    openai_api_key="sk-dsOPRRFZZFjdLtL1IfiRfZZ8cGv125eP6YHetH6JGQAL9Alx",
    zhipu_api_key="3eca16d8e1c141369f93ec7b5fe564c5.5WV9s8FNOtwn1NzZ"
)

# 根路由
@app.route('/')
def index():
    return send_from_directory('templates', "index.html")

@app.route('/api/query', methods=['GET', 'POST'])
def query():
    try:
        if request.method == 'POST':
            data = request.json
            question = data.get('question')
            session_id = data.get('session_id', 'default')
        else:
            question = request.args.get('question')
            session_id = request.args.get('session_id', 'default')
            
        if not question:
            return jsonify({"error": "No question provided"}), 400

        print(f"Received question: {question} for session: {session_id}")
        
        def generate():
            try:
                for chunk in rag.query(question, session_id=session_id):
                    print(f"Sending chunk: {chunk}")
                    yield f"data: {chunk}\n\n"
            except Exception as e:
                print(f"Error in generate(): {str(e)}")
                yield f"data: Error: {str(e)}\n\n"
        
        response = Response(generate(), mimetype='text/event-stream')
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Connection'] = 'keep-alive'
        response.headers['X-Accel-Buffering'] = 'no'
        return response

    except Exception as e:
        print(f"Error in /api/query: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/case-analysis', methods=['GET'])
def case_analysis():
    try:
        question = request.args.get('question')
        session_id = request.args.get('session_id', 'default')  # 获取session_id
        
        if not question:
            return jsonify({"error": "No question provided"}), 400

        print(f"Received case analysis question: {question} for session: {session_id}")
        
        def generate():
            try:
                for chunk in md_rag.query(question, session_id=session_id):  # 传入session_id
                    print(f"Sending chunk: {chunk}")
                    yield f"data: {chunk}\n\n"
            except Exception as e:
                print(f"Error in generate(): {str(e)}")
                yield f"data: Error: {str(e)}\n\n"
        
        response = Response(generate(), mimetype='text/event-stream')
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Connection'] = 'keep-alive'
        response.headers['X-Accel-Buffering'] = 'no'
        return response

    except Exception as e:
        print(f"Error in /api/case-analysis: {str(e)}")
        return jsonify({"error": str(e)}), 500

# 新增：获取聊天历史的接口
@app.route('/api/chat-history', methods=['GET'])
def get_chat_history():
    try:
        session_id = request.args.get('session_id', 'default')
        history = rag.get_chat_history(session_id)
        return jsonify({"history": [{"role": msg.type, "content": msg.content} for msg in history]})
    except Exception as e:
        print(f"Error in /api/chat-history: {str(e)}")
        return jsonify({"error": str(e)}), 500

# 添加一个测试路由来检查静态文件是否可访问
@app.route('/test-static')
def test_static():
    static_path = app.static_folder
    files = os.listdir(static_path)
    return jsonify({
        'static_folder': static_path,
        'files': files,
        'css_exists': os.path.exists(os.path.join(static_path, 'css', 'style.css'))
    })

@app.route('/get_random_questions')
def get_random_questions():
    # 读取所有问题
    with open('database/00q.json', 'r', encoding='utf-8') as f:
        questions = json.load(f)
    # 随机选择6个问题
    random_questions = random.sample(questions, 6)
    return jsonify(random_questions)

if __name__ == '__main__':
    print(f"Server running. Static folder: {current_dir}")
    app.run(debug=True, port=5000)
