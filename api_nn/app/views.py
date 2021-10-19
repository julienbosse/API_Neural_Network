from flask import render_template, request, abort, redirect, url_for, Flask, Response
import tensorflow as tf
import numpy as np
import cv2
from app import app

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    ext = file.filename.split('.')[1]
    if file.filename != '':
        file.save("app/static/img/img."+ext)

        img = cv2.imread("app/static/img/img."+ext)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dim=(28,28)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite("app/static/img/resized.jpg",img)
        img = np.resize(img,(28,28,1)).reshape(1,28,28,1)

        model = tf.keras.models.load_model("app/static/model")

        pred = np.argmax(model.predict(img))

    return render_template('result.html', prediction = pred)
