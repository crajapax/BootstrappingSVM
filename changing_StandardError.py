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
    n = 1000
    C = 1
    processes = 10
    replications = 500
    kernel = 'linear'
    ### important if you want to know in the filename which distribution the data hase
    simulation_function = 'dataSimulation([0.8, 0.7, 0.9, -0.3], 1, 0, n)'
    
    ### 2. Part: CALCULATION
    
    # values being plotted
    n_support_vectors = list()
    variances = list()
    
    for i in range(40):
        
        # Set error dynamically
        error = (i+1) / 20
        
        # generate data
        trainings_data = dataSimulation([0.8, 0.7, 0.9, -0.3], error, 0, n)
        prediction_data = dataSimulation([0.8, 0.7, 0.9, -0.3], error, 0, n)
        
        # do Bootstrap
        result = bootstrap_the_svm(trainings_data, prediction_data, kernel, C, "auto", 1, processes, replications)

        # Print outs the result
        print("Iteration:", i)
        result.view() 
        
        # add values to the lists
        n_support_vectors.append(result.n_support[0] + result.n_support[1])
        variances.append(result.var_distance)

        
    ### 3. Part: PLOTTING
    
    # Sort array
    n_support_vectors, variances = sort_multiple_array(n_support_vectors, variances)
    
    # Do the plotting
    #plt.plot(n_support_vectors, variances, c = 'green')
    do_quadratic_regression(n_support_vectors, variances, 0, n)
    plt.scatter(n_support_vectors, variances, c = 'green')
    plt.xlabel('Number of support vectors')
    plt.ylabel('Variance of distances')

    # Save the plot
    filename = 'output/change_Support_Vectors_' + get_time_stamp() + '_n=' + str(n) + '_replication=' + str(replications) + '_C=' + str(C) + '_kernel=' + kernel + '_data= ' + simulation_function
    plt.savefig(filename + '.png')
    print("File", filename + '.png', "has been created.")
    
    create_text_file(filename, n_support_vectors, variances)