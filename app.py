from flask import Flask, render_template, request, jsonify
from src.generate.simple_model import SimpleModel
from src.generate.model_with_kv_cache import ModelWithKVCache
# from src.generate.batching import Batching
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Initialize models
simple_model = SimpleModel('gpt2')
model_with_kv_cache = ModelWithKVCache('gpt2')
# batching_model = Batching('gpt2')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    user_input = request.form['user-input']
    parameter = int(request.form['parameter-select'])
    method = request.form['method-select']

    print(f"User input: {user_input}, Parameter: {parameter}, Method: {method}")

    if method == 'normal':
        output, plot_data = simple_model(user_input, max_length=parameter)
    elif method == 'batching':
        output = batching_model(user_input, max_length=parameter)
        time_taken = batching_model.last_generation_time
    elif method == 'caching':  # caching
        output, plot_data = model_with_kv_cache(user_input, max_length=parameter)

    return jsonify({
        'output': output,
        'graph': plot_data
    })

if __name__ == '__main__':
    app.run(debug=True)
