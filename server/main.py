from flask import Flask, request, jsonify
from textToImage import generateImage

app = Flask(__name__)

@app.route('/', methods=['POST'])
def generate():
    args = request.json
    prompt = args['prompt']
    seed = args['seed']
    width = args['width']
    height = args['height']
    steps = args['steps']

    images, seed = generateImage(prompt, seed, width, height, steps)

    return jsonify(
        images=images,
        seed=seed
    )

app.run(host='0.0.0.0', port=3001, debug=True)