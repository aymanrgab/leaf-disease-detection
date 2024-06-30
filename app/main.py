import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from src.predict import predict
import base64

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                predicted_class, confidence = predict(filepath, 'models/saved_models/final_model.pth')
                
                # Read the visualization image
                with open('prediction_visualization.png', 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                
                return jsonify({
                    'prediction': predicted_class,
                    'confidence': f'{confidence:.2f}',
                    'visualization': img_data
                })
            except Exception as e:
                return jsonify({'error': str(e)})
        else:
            return jsonify({'error': 'File type not allowed'})
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
