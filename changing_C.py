import os
import sys
from functions import *
from numpy import *
import math
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    ### 1. Part: SIMULATION OF DATA
    
    # Seed
    random.seed(1109992)
    
    # Parameters
    n = 100
    processes = 1
    replications = 1000
    number_of_datasets = 10
    kernel = 'rbf'
    ### important if you want to know in the filename which distribution the data hase
    simulation_function = 'dataSimulation([0.8, 0.7, 0.9, -0.3, -0.7, 0.8, 12], 1, 0, n)'
    
    # List of Datasets
    trainings_data = list()
    prediction_data = list()
    
    # Simulation of the data
    for i in range(number_of_datasets):
        trainings_data.append(dataSimulation([0.8, 0.7, 0.9, -0.3], 1, 0, n))
        prediction_data.append(dataSimulation([0.8, 0.7, 0.9, -0.3], 1, 0, n))
    
    ### 2. Part: CALCULATION
    
    # values being plotted
    c_s = list()
    mean_of_variances = list()
    
    for i in range(20):
        
        # Set C parameter dynamically
        C = (i+1) / 10
        
        # List of variances of this C parameter and of the different data sets
        variances_for_this_iteraction = list()

        print("Iteration:", i)
        for j in range(len(trainings_data)):
            # do Bootstrap
            result = bootstrap_the_svm(trainings_data[j], prediction_data[j], kernel, C, "auto", 1, processes, replications)
            # print the result
            result.view() 
            print("Iteration", i, ".", j)
            # add variance to list
            variances_for_this_iteraction.append(result.var_distance)
        
        # add values to the lists
        c_s.append(C)
        mean_of_variances.append(mean(variances_for_this_iteraction))
        
    # Do the plotting
    plt.plot(c_s, mean_of_variances, c = 'green')
    plt.scatter(c_s, mean_of_variances, c = 'green')
    plt.xlabel('C')
    plt.ylabel('Variance of distances')
        
    # Save the plot
    filename = 'output/change_C_' + get_time_stamp() + '_n=' + str(n) + '_replication=' + str(replications) + '_numver_of_datasets=' + str(number_of_datasets) + '_kernel=' + kernel + '_data= ' + simulation_function
    plt.savefig(filename + '.png')
    print("File", filename + '.png', "has been created.")
    
    create_text_file(filename, c_s, mean_of_variances)