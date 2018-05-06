from app import app
from flask import jsonify
from flask import request
from tempfile import TemporaryFile

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv

from scipy.io import loadmat
from scipy.optimize import minimize

from datetime import datetime

@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"


def sigmoid(z):
	return (1 / (1 + np.exp(-z)))

def lrcostFunctionReg(theta, X, y, reg, return_grad=False):
	m = y.size
	h = sigmoid(X.dot(theta))
	
	J = -1*(1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y)) + (reg/(2*m))*np.sum(np.square(theta[1:]))
	
	if np.isnan(J[0]):
		return(np.inf)
	return (J[0])

def cost(theta, X, y, learningRate):  
	theta = np.matrix(theta)
	X = np.matrix(X)
	y = np.matrix(y)
	first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
	second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
	reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
	return np.sum(first - second) / (len(X)) + reg

def gradient_with_loop(theta, X, y, learningRate):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])

        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:,i])

    return grad

def gradient(theta, X, y, learningRate):  
	theta = np.matrix(theta)
	X = np.matrix(X)
	y = np.matrix(y)

	parameters = int(theta.ravel().shape[1])
	error = sigmoid(X * theta.T) - y

	grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)

	# intercept gradient is not regularized
	grad[0, 0] = np.sum(np.multiply(error, X[:,0])) / len(X)

	result = np.array(grad).ravel()

	print np.shape(result)

	return result

def oneVsAll(X, y, num_labels, lambda_reg):
	m, n = X.shape
	all_theta = np.zeros((num_labels, n + 1))
	X = np.column_stack((np.ones((m,1)), X))

	for c in xrange(num_labels):
		# initial theta for c/class
		initial_theta = np.zeros((n + 1, 1))
		print("Training {:d} out of {:d} categories...".format(c+1, num_labels))
		myargs = (X, (y%10==c).astype(int), lambda_reg, True)
		theta = minimize(lrcostFunctionReg, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':13}, method="Newton-CG", jac=True)
		all_theta[c,:] = theta["x"]

	return all_theta

def one_vs_all(features, classes, n_labels, reg):
	rows = features.shape[0]
	params = features.shape[1]
	all_theta = np.zeros((n_labels, params + 1))
	X = np.insert(features, 0, values=np.ones(rows), axis=1) 

	for i in range(1, n_labels + 1):
		theta = np.zeros(params + 1)
		y_i = np.array([1 if label == i else 0 for label in classes])
		y_i = np.reshape(y_i, (rows, 1))
		fmin = minimize(fun=cost, x0=theta, args=(X, y_i, reg), method='TNC', jac=gradient_with_loop)
		all_theta[i-1,:] = fmin.x

	return all_theta

def predict_all(X, all_theta):  
	rows = X.shape[0]
	params = X.shape[1]
	num_labels = all_theta.shape[0]

	X = np.insert(X, 0, values=np.ones(rows), axis=1)
	X = np.matrix(X)
	all_theta = np.matrix(all_theta)

	h = sigmoid(X * all_theta.T)

	h_argmax = np.argmax(h, axis=1)
	h_argmax = h_argmax + 1

	return h_argmax

def getY():
	with open('training-results.csv', 'rb') as csvfile:
		spamreader = csv.reader(csvfile)
		result = []
		for row in spamreader:
			scenario = int(row[0])
			result.append([scenario])

	return np.asarray(result)

epoch = datetime.utcfromtimestamp(0)
def unix_time_millis(dt):
	return (dt - epoch).total_seconds() * 1000.0

def getX():
	with open('heartbeat-training.csv', 'rb') as csvfile:
		spamreader = csv.reader(csvfile)
		result = []
		for row in spamreader:
			data_row = []
			for idx, val in enumerate(row):
				if idx == 4:
					data_row.append(float(row[idx]))
				elif idx == 5 or idx == 6:
					times = datetime.strptime(row[idx], '%Y-%m-%d %H:%M:%S')
					data_row.append(unix_time_millis(times))
				elif idx >= 7 and idx <= 16:
					data_row.append(float(row[idx]))
				else:
					data_row.append(float(row[idx]))

			result.append(data_row)

	data = np.array(result, dtype=np.float128)

	return np.c_[np.ones((data.shape[0],1)), data]


# defined scenarios
# 0 - resting
# 1 - sleeping
# 3 - exercising
# 4 - overdose
@app.route('/scenarios', methods = ['POST'])
def scenarios():
	y = getY()
	X = getX()

	# Add constant for intercept
	X = np.c_[np.ones((X.shape[0],1)), X]

	print('X: {} (with intercept)'.format(X.shape))
	print('y: {}'.format(y.shape))

	all_theta = one_vs_all(X, y, 4, 1)

	print all_theta, np.shape(all_theta)

	test_x = req.data
	test_x = np.array(test_x, dtype=np.float128)
	new_x = np.c_[np.ones((test_x.shape[0],1)), test_x]
	new_x = np.c_[np.ones((new_x.shape[0],1)), new_x]

	prediction =predict_all(new_x, all_theta)

	return jsonify(scenario=prediction)

def getAccuracy():
	y = getY()
	X = getX()
	X = np.c_[np.ones((X.shape[0],1)), X]

	all_theta = one_vs_all(X, y, 3, 1)  

	y_pred = predict_all(X, all_theta)  
	correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]  
	accuracy = (sum(map(int, correct)) / float(len(correct)))  

	print 'accuracy = {0}%'.format(accuracy * 100)