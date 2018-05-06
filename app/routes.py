from app import app
from flask import jsonify

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"

@app.route('/scenarios', methods = ['POST'])
def scenarios():
	return jsonify(overdose=True)