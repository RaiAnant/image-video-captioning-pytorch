from flask import Flask, request, jsonify
from predictor import Predictor
from threading import Thread
import datetime
import os
import cv2
import subprocess
import random

app = Flask(__name__)
epoch = datetime.datetime.utcfromtimestamp(0)


def unix_time_millis(dt):
    return (dt - epoch).total_seconds() * 1000.0


def extract_key_frames(path):
    print("Extracting frames")
    os.system("ffmpeg -skip_frame nokey -i " + path + " -vsync 0 -r 30 -f image2 " + "/".join(
        path.split("/")[:-1]) + "/thumbnails-%02d.jpeg")


def get_frame_types(video_fn):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video_fn]).decode()
    frame_types = out.replace('pict_type=', '').split()
    return zip(range(len(frame_types)), frame_types)


def save_i_keyframes(video_fn):
    frame_types = get_frame_types(video_fn)
    print("Extracting frames")
    i_frames = [x[0] for x in frame_types if x[1] == 'I']
    if i_frames:
        basename = os.path.splitext(os.path.basename(video_fn))[0]
        cap = cv2.VideoCapture(video_fn)
        for frame_no in i_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            frmno = str(frame_no)
            n = len(str(i_frames[-1]))
            outname = "/".join(video_fn.split("/")[:-1])+"/"+"0"*(n-len(frmno)) +frmno + '.jpg'
            print("saved: ")
            cv2.imwrite(outname, frame)
            print('Saved: ' + outname)
        cap.release()
    else:
        print('No I-frames in ' + video_fn)


class Worker:
    def __init__(self):
        self.thread = None
        self.jobs = []
        self.basepath = os.path.dirname(__file__)

    def add_job(self, job):
        self.jobs.append(job)

    def work(self):
        while self.jobs:
            print("doing work")
            dir = self.basepath + "/videos/" + str(self.jobs[0])
            save_i_keyframes(dir + "/" + str(self.jobs[0]) + ".mp4")
            for filename in os.listdir(dir):
                if filename.endswith(".jpg"):

            # print(self.jobs[0], ":", predictor.predict())
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
    id = random.randint(0, 99999)+datetime.datetime.now().timestamp()
    f = request.files['file']
    # model = request.args.get('username')
    basepath = os.path.dirname(__file__)
    os.mkdir(basepath + "/videos/" + str(id))
    file_path = os.path.join(basepath + "/videos/" + str(id), str(id) + "." + str(f.filename).split(".")[-1])
    f.save(file_path)

    worker.add_job(id)
    worker.start()
    return jsonify({'id': id})
