from flask import Flask
from flask_bootstrap import Bootstrap
import tensorflow as tf

app = Flask(__name__)

from app import views
# from app import models
Bootstrap(app)