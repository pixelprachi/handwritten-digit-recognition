from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64

app = Flask(__name__)

print('Loading model...')
model = tf.keras.models.load_model('mnist_model.h5')
print('Model loaded successfully!')

total_params = model.count_params()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image']
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)

        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        img = img.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img).astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        predictions = model.predict(img_array, verbose=0)[0]

        probabilities = [
            {'digit': i, 'probability': float(predictions[i]) * 100}
            for i in range(10)
        ]
        probabilities.sort(key=lambda x: x['probability'], reverse=True)

        predicted_digit = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100

        return jsonify({
            'success': True,
            'digit': predicted_digit,
            'confidence': round(confidence, 2),
            'probabilities': probabilities
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        img = Image.open(file.stream).convert('L')
        img = img.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img).astype('float32') / 255.0
        img_array = 1.0 - img_array  # Invert: white bg + dark digit → black bg + white digit
        img_array = img_array.reshape(1, 28, 28, 1)

        predictions = model.predict(img_array, verbose=0)[0]

        probabilities = [
            {'digit': i, 'probability': float(predictions[i]) * 100}
            for i in range(10)
        ]
        probabilities.sort(key=lambda x: x['probability'], reverse=True)

        predicted_digit = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100

        return jsonify({
            'success': True,
            'digit': predicted_digit,
            'confidence': round(confidence, 2),
            'probabilities': probabilities
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    return jsonify({
        'total_params': total_params,
        'training_samples': 60000,
        'test_samples': 10000,
        'architecture': 'CNN (3 Conv + 2 Dense layers)',
        'dataset': 'MNIST',
        'accuracy': '99.2%'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)