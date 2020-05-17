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

    #path = 'C:/Users/Sankalp Thakur/Capstone/aerial_pedestrian_detection-master/examples/little_video0_550.jpg'
    video_path = 'D:/Capstone/videos/bookstore/video3/trimdemo.avi'
    output_path = 'D:/Capstone/videos/bookstore/video3/pred2.avi'
    fps = 15

    vcapture = cv2.VideoCapture(video_path)

    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))  # uses given video width and height
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vwriter = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))  #

    #image = Image.open(path)
    #img = preprocess_image(image)
    run_detection_image(video_path,vwriter,output_path)
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


def play_video():
    cap = cv2.VideoCapture(output_path)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video file")

    # Read until video is completed
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def run_detection_image(video_path,vwriter, output_path):
    vcapture = cv2.VideoCapture(video_path)
    count = 0
    success = True
    start = time.time()
    while success:
        #if count % 100 == 0:
        print("frame: ", count)
        count += 1  # see what frames you are at
        # Read next image
        success, image = vcapture.read()

        if success:
            draw = image.copy()
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

            # # preprocess image for network
            image = preprocess_image(image)
            image, scale = resize_image(image)

            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

            boxes /= scale

            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # scores are sorted so we can break
                if score < 0.3:
                    break

                color = label_color(label)

                b = box.astype(int)
                draw_box(draw, b, color=color)

                caption = "{} {:.3f}".format(labels_to_names[label], score)
                draw_caption(draw, b, caption)

            vwriter.write(draw)
    vcapture.release()
    vwriter.release()  #
    end = time.time()
    #play_video(output_path)
    print("Total Time: ", end - start)
    return send_file(output_path)