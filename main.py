from flask import Flask, request, jsonify
from predictor import Predictor
from threading import Thread
import datetime
import os

app = Flask(__name__)
epoch = datetime.datetime.utcfromtimestamp(0)


def unix_time_millis(dt):
    return (dt - epoch).total_seconds() * 1000.0


class Worker:
    def __init__(self):
        self.thread = None
        self.jobs = []

    def add_job(self, job):
        self.jobs.append(job)

    def work(self):
        while self.jobs:
            print(self.jobs[0], ":", predictor.predict())
            self.jobs = self.jobs[1:]

    def start(self):
        if self.thread is None or not self.thread.is_alive():
            print("new thread started")
            self.thread = Thread(target=self.work)
            self.thread.start()


worker = Worker()

predictor = Predictor()


@app.route('/caption', methods=['POST'])
def caption():
    id = datetime.datetime.now().timestamp()
    worker.add_job(id)
    worker.start()
    return jsonify({'id': id})


@app.route('/upload', methods=['POST'])
def upload():
    id = datetime.datetime.now().timestamp()
    f = request.files['file']
    basepath = os.path.dirname(__file__)
    os.mkdir(basepath+"/videos/"+str(id))
    file_path = os.path.join(basepath+"/videos/"+str(id), str(id)+"."+str(f.filename).split(".")[-1])
    f.save(file_path)
    return jsonify({'id': id})
