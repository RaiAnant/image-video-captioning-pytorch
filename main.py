from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/caption', methods=['POST'])
def captionize():
    return jsonify({'result': 1})
