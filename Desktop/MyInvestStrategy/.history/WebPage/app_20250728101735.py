from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允许所有来源跨域请求

@app.route('/')
def index():
    return render_template('index.html')  # 返回 HTML 页面

@app.route('/greet', methods=['POST'])
def greet():
    data = request.get_json(force=True)
    name = data.get('name', '游客')
    return jsonify({"greeting": f"你好，{name}！欢迎来到本站！"})
if __name__ == '__main__':
    app.run(port=5004)
