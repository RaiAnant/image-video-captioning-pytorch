from flask import Flask, request, jsonify
from predictor import Predictor

app = Flask(__name__)

predictor = Predictor()


@app.route('/caption', methods=['POST'])
def caption():
    return jsonify({'result': predictor.predict()})




