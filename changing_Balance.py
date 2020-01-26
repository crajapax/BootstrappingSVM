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
    n = 500
    C = 1
    processes = 10
    replications = 1000
    kernel = 'linear'
    ### important if you want to know in the filename which distribution the data hase
    simulation_function = 'dataSimulation([0.8, 0.7, 0.9, -0.3], 1, intercept, n)'
    
    ### 2. Part: CALCULATION
    
    # values being plotted
    balances = list()
    variances = list()
    
    for i in range(52):
        
        # Set intercept dynamically
        intercept = ((i+1) / 13) - 2
        
        # generate data
        trainings_data = dataSimulation([0.8, 0.7, 0.9, -0.3], 1, intercept, n)
        prediction_data = dataSimulation([0.8, 0.7, 0.9, -0.3], 1, intercept, n)
        
        # do Bootstrap
        result = bootstrap_the_svm(trainings_data, prediction_data, kernel, C, "auto", 1, processes, replications)

        # Print outs the result
        print("Iteration:", i)
        result.view() 
        
        # add values to the lists
        balances.append(result.classification[0][0]/n)
        variances.append(result.var_distance)
        
    ### 3. Part: PLOTTING
    
    # Sort array
    balances, variances = sort_multiple_array(balances, variances)
    
    # Do the plotting
    #plt.plot(balances, variances, c = 'green')
    do_quadratic_regression(balances, variances, 0, 1)
    plt.scatter(balances, variances, c = 'green')
    plt.xlabel('Balances')
    plt.ylabel('Variance of distances')
    
    # Save the plot
    filename = 'output/change_Balances_' + get_time_stamp() + '_n=' + str(n) + '_replication=' + str(replications) + '_C=' + str(C) + '_kernel=' + kernel + '_data= ' + simulation_function
    plt.savefig(filename + '.png')
    print("File", filename + '.png', "has been created.")
    
    create_text_file(filename, balances, variances)
    

