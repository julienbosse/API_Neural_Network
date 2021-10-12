from flask import Flask
from flask_bootstrap import Bootstrap
import tensorflow as tf

model = tf.keras.models.load_model("app/static/model")
app = Flask(__name__)

from app import views
# from app import models
Bootstrap(app)