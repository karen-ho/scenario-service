from app import app
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"
