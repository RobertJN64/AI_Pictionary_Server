import flask
from flask import request, send_file
from io import BytesIO
from PIL import Image
import base64
import re

imlist = []

app = flask.Flask(__name__)

@app.route('/')
def home():
    return send_file('index.html')


@app.route('/upload_image', methods=['POST'])
def save_canvas():
    image_data = re.sub('^data:image/.+;base64,', '', request.json['imageBase64'])
    im = Image.open(BytesIO(base64.b64decode(image_data)))
    if len(imlist) == 0:
        imlist.append(im)
    else:
        imlist[0] = im
    return "Image OK"


def run(_imlist):
    global imlist
    imlist = _imlist
    app.run('0.0.0.0', 80)