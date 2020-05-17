from flask import Flask
from flask import send_file

import base64
import re
from io import BytesIO

import numpy as np
from PIL import Image

# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2

from PIL import Image


import os
import numpy as np
import time
from flask import render_template
from matplotlib import image


def get_session():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)
# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
tf.compat.v1.keras.backend.set_session(get_session())

app = Flask(__name__)

model_path = "resnet50_csv_12_inference.h5"
model = models.load_model(model_path, backbone_name='resnet50')
labels_to_names = {0: 'Biker', 1: 'Car', 2: 'Bus', 3: 'Cart', 4: 'Skater', 5: 'Pedestrian'}

#@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

#@app.route('/predict', methods=['GET', 'POST'])
@app.route('/')
def hello_world():

    path = 'C:/Users/Sankalp Thakur/Capstone/aerial_pedestrian_detection-master/examples/little_video0_550.jpg'

    #image = Image.open(path)
    #img = preprocess_image(image)
    return run_detection_image(path)
    #return render_template("index.html")

def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image

def np_to_base64(img_np):
    """
    Convert numpy image (RGB) to base64 string
    """
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")


def run_detection_image(path):

    image = cv2.imread(path)
# image base 64   retimag = np_to_base64(image)
#    print(base64_to_pil(retimag))
    # # copy to draw on

    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    #
    # # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)
    #
    # # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)
    #
    # # correct for image scale
    boxes /= scale

    # # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
         # scores are sorted so we can break
         if score < 0.3:
             break

         color = label_color(label)

         b = box.astype(int)
         draw_box(draw, b, color=color)

         caption = "{} {:.3f}".format(labels_to_names[label], score)
         draw_caption(draw, b, caption)

     #plt.figure(figsize=(30, 30))
     #plt.axis('off')
    # abcd = plt.imshow(draw)
    # #plt.show()
    #
    # #file, ext = os.path.splitext(filepath)
    # #image_name = file.split('/')[-1] + ext
    # #output_path = os.path.join('examples1/results/', image_name)
    #
    # draw_conv = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    # #cv2.imwrite(output_path, draw_conv)
    # return image
    retimag = np_to_base64(draw)
    strimg = base64_to_pil(retimag)
    strimg.save("testimg.png")
    print(strimg)
    return send_file('testimg.png',mimetype="image/png")
