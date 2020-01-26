import os
import sys

from multiprocessing import Process, Pool
from numpy import *
from numpy import random as rd
from sklearn import svm
from models import *
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import math
import random
import statistics
import datetime


def bootstrap_the_svm(trainings_data, prediction_data, kernel, C, gamma = "auto", degree = 3, processes = 1, replications = 10):
    
    input_parameters = SVM_Input(trainings_data, prediction_data, kernel, C, gamma, degree)

    data = input_parameters
    real_svm = do_svm(data)
    
    ### Do bootstrapping
    PROCESSES = processes
    REPLICATIONS = replications
    pool = Pool(processes = PROCESSES)
    results = pool.map(single_sample_and_svm, [data] * REPLICATIONS)
        
    ### Calculate the Variance of the Support Vector Machine
    
    points_information = Points_Information(results)
    
    variance_of_svm_probabilites = calculate_variance_of_svm(points_information.probabilites)
    variance_of_svm_distance_to_hyperplane = calculate_variance_of_svm(points_information.distances)
    
        
    return(Bootstrap_Result(real_svm[0], real_svm[1], real_svm[2], variance_of_svm_probabilites, variance_of_svm_distance_to_hyperplane, real_svm[0].n_support_))
    
def random_sample_with_replacement(population, sample_size):
    "Chooses k random elements (with replacement) from a population"
    n = len(population)
    _random, _int = random.random, int  # speed hack
    return [_int(_random() * n) for i in range(sample_size)]

def random_sample_with_replacement_of_dataset(data, sample_size):
    y, x = data
    sorted_list = sorted(random_sample_with_replacement(range(len(y)), sample_size))
    y = [y[i] for i in sorted_list]
    x = [x[i] for i in sorted_list]
    return(y, x)
        

def single_sample_and_svm(input_parameters):
    "New version of using skilearn"
    
    #training_data, prediction_data = input_parameters
    
    training_data = input_parameters.training_data
    prediction_data = input_parameters.prediction_data
    kernel = input_parameters.kernel
    
    # Set seed for each Thread new as otherwise each process starts with same Seed -> ugly
    SEED = datetime.datetime.now().time().microsecond
    random.seed(SEED)
    
    y, X = random_sample_with_replacement_of_dataset(training_data, len(training_data[0]))
    
    clf = svm.SVC(probability = True, kernel = kernel, C = input_parameters.C, gamma = input_parameters.gamma, degree = input_parameters.degree )
    fit = clf.fit(X, y)  
    
    yy = 0
    
    distance_to_hyperplane = clf.decision_function(prediction_data[1])
    probabilities = clf.predict_proba(prediction_data[1])
    probabilities = list(zip(*probabilities))
    probabilities = probabilities[1]
    
    
    result = SVM_Result(probabilities, distance_to_hyperplane, yy, clf.n_support_, 0)
    
    return(result)
    
    
def do_svm(input_parameters):
    
    training_data = input_parameters.training_data
    prediction_data = input_parameters.prediction_data
    kernel = input_parameters.kernel
    
    y, X = training_data
    
    clf = svm.SVC(probability = True, kernel = kernel, C = input_parameters.C, gamma = input_parameters.gamma, degree = input_parameters.degree )
    fit = clf.fit(X, y)  
    
    
    score = clf.score(prediction_data[1],prediction_data[0])
    prediction = clf.predict(prediction_data[1])
    y = prediction_data[0]
    
    y_count_1 = (y.size + sum(y)) / 2
    prediction_count_1 = (prediction.size + sum(prediction)) / 2
    
    return(clf, score, [[y_count_1, y.size-y_count_1], [prediction_count_1, prediction.size-prediction_count_1]])
    

def calculate_variance_of_svm(results):
    
    # Creating an empty array with size of points
    standard_deviations = [None] * len(results)
    
    # For each point the standard deviation of the different predicted values will be calculated
    for index in range(len(results)):
        std = statistics.stdev(results[index])
        standard_deviations[index] = std
        
    return(statistics.mean(standard_deviations))
    
    
def transpose(matrix):
    return(list(zip(*matrix)))
    
def sort_multiple_array(x, y):
    multiple = [x,y]
    multiple = transpose(multiple)
    multiple = sorted(multiple)
    multiple = transpose(multiple)
    return multiple

def dataSimulation(coefs, errorCoef, intercept, size):
	inputs = []
	error = errorCoef*rd.standard_normal(size)
	y = error + intercept
	for i in range(len(coefs)):
		 inputs = inputs + [rd.standard_normal(size)]		 
		 y = y+coefs[i]*inputs[i]
	y = sign(y)
	inputs = list(zip(*inputs))
	return([y,inputs])

def get_time_stamp():
    return(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def create_text_file(name, X, Y):
    # Save data to csv
    txt = open(name + '.txt', 'w+')
    txt.write(" ".join(str(x) for x in X))
    txt.write("\n")
    txt.write(" ".join(str(x) for x in Y))
    txt.closed
    print("File", name + '.txt', "has been created.")

#Funktion zur Simulierung von Daten mit y wert abhaengig vom Abstand von Zentren
def centroidSimulation(coefs, locations, errorCoef, size, intercept, distance, xdistribution = "normal", par1 = 0, par2 = 1):
	X = []
	dimension = len(list(zip(*locations)))
	for i in range(dimension):
		if(xdistribution == "normal"):		
			X = X + [rd.normal(par1, par2, size)]
		elif(xdistribution == "uniform"):
			X = X + [rd.uniform(par1, par2, size)]
		else:
			print("Please choose supported Distribution")
			return None	
	X = list(zip(*X))
	distances = []
	error = errorCoef*rd.standard_normal(size)
	y = error + intercept
	for i in range(size):	
		newDistance = pdist([X[i]]+locations, distance)
		newDistance = newDistance[:len(locations)]
		inverseDistance = power(newDistance, -1)
		y[i] = y[i] + dot(coefs, inverseDistance)
		distances = distances + [newDistance]
	y = sign(y)
	#distances = list(zip(*distances))
	return([y,X])	
	
def do_quadratic_regression(x, y, x_start, x_end):
    x = array(x)
    y = array(y)
    x2 = x * x
    X = array([ones(len(x)), x, x2])
    coefficients = linalg.lstsq(X.T,y)[0]
    
    xx = linspace(x_start, x_end)
    yy = coefficients[0] + coefficients[1] * xx + coefficients[2] * (xx * xx)
    plt.plot(xx, yy, '-k', color = "green")    
	