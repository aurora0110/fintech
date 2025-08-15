from flask import Flask, request, jsonify
from flask_cors import CORS  # 跨域支持

app = Flask(__name__)
CORS(app)  # 允许跨域请求

@app.route('/', methods=['POST'])
def greet():
    data = request.get_json()
    name = data.get('name', '游客')
    return jsonify({"greeting": f"你好，{name}！欢迎来到本站！"})

if __name__ == '__main__':
    app.run(port=5000)
