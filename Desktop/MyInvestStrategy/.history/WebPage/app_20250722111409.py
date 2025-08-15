# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS  # 允许跨域请求

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

@app.route('/greet', methods=['POST'])
def greet():
    data = request.get_json(force=True)  # 强制解析 JSON
    name = data.get('name', '游客')
    return jsonify({"greeting": f"你好，{name}！欢迎来到本站！"})

@app.route('/', methods=['GET'])
def home():
    return "后端已启动，请通过前端页面访问 /greet 接口。"

if __name__ == '__main__':
    app.run(port=5000)
