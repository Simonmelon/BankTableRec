import os
import sys

from werkzeug.utils import secure_filename

from core.json_test import main_pipeline

curr_path = os.path.abspath(os.path.dirname(__file__))
project_dir = os.path.join(curr_path, "../../")
print("Base Project Dir", project_dir)
sys.path.append(project_dir)
import time
from flask import Flask, request, jsonify

app = Flask(__name__)

# 配置上传参数
UPLOAD_FOLDER = '/root/cell_rec/uploads/'  # 上传文件保存目录
ALLOWED_EXTENSIONS = {'pdf'}  # 允许的文件类型
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传大小为16MB

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "请求中没带有file"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "没有文件"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "文件格式不支持"}), 400
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    filename = secure_filename(file.filename)
    upload_folder = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(file_path)
    file.save(file_path)
    result = main_pipeline(file_path)
    return result

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=3333)