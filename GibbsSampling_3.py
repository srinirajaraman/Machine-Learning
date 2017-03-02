'''
Reference: Algorithm 10.1 from Gibbs sampling for a discrete undirected model
Book : Simon J.D Prince

Notation details:
vector - vec
matrix - mat

Author:Srinivas R
Dept of EECE 
Computational Ocularscience Lab
Date: 22/9/2016


'''

#Import packages
import numpy as np
import matplotlib.pyplot as plt
import math as m
import os as os


def function_readimage(given_image):
    return plt.imread(given_image)

def function_plot(given_image, color_option):
	plt.imshow(given_image, cmap = color_option, vmin = 0, vmax = 1, interpolation = 'none')
	plt.show()

def function_get_right_coord(i, j):
    return i, j + 1

def function_get_left_coord(i, j):
    return i, j - 1

def function_get_top_coord(i, j):
    return i - 1, j

def function_get_bottom_coord(i, j):
    return i + 1, j

def function_get_current_coord(i, j):
    return i, j


'''
If pixel values are same return theta_01 which has highest probability or if pixel values are different return theta_00 which has least 
probability
'''
def binary_phi_calc(w1, w2, theta_00, theta_01):
    return theta_00 if w1 == w2 else theta_01 
    #if w1 == w2:
    #    return theta_01
    #else:
    #    return theta_00

#Compute cliques based on the node index
def compute_clique(node_index, row_size, col_size): 
    #
    s_list = []
    r = node_index[0]
    c = node_index[1]

    #Identify all valid cliques containing node_index
    if( r - 1 >= 0):
        s_list.append(function_get_top_coord(r, c))

    if( r + 1 < row_size):
        s_list.append(function_get_bottom_coord(r, c))

    if( c - 1 >= 0):
        s_list.append(function_get_left_coord(r, c))
    
    if( c + 1 < col_size):
        s_list.append(function_get_right_coord(r, c))

    return s_list

#Helper function to get index with the value 'val'
def function_get_index(num_array, val):
    #

    for ind in range(0, len(num_array)):
        if(num_array[ind] == val):
            break;

    return ind

#Perform gibbs sampling for a binary image
def gibbs_sampling_categorical_2d(x_mat_t0, sample_space_vec):
    #x_mat_t0: Current state of the undirected graph for Gibb's sampling; will return the next state x_mat_t1
    #sample_space_vec: contains the values that each of the x_d variable in x_mat sample from. e.g. [0, 1] for binary sample space
    #

    #Dimension D in Algorithm 10.1 has been split into row and columns (i.e. into a 2D vector)
    row_size, col_size = np.shape(x_mat_t0);

    #Initialize the next state of the undirected graph with the previous state
    x_mat_t1 = x_mat_t0;

    #Potential function paramteres
    #Note: these are defined for binary MRF and should be updated when there is change in the potential function form or change from binary to categorical variable
    theta_00 = 0.1;
    theta_01 = 1.0;
    

    #For each dimension
    for row in range(0, row_size):
        for col in range(0, col_size):
            #Compute the values that a variable can take here {0, 1}
            lambda_vec = np.ones(np.shape(sample_space_vec)); #Initialize the lambda parameter of the categorical distribution for the d'th location (or [row, col] location)
        
            #Get the cliques corresponding to row and col indcies
            clique_list = compute_clique([row, col], row_size, col_size);
        
            for k in sample_space_vec:
                #Set the current location's state to k, i.e. now working with p(x_d = k | x_{1...D}\d at t0)
                x_mat_t1[row, col] = k

                #Compute the unnormalized marginal probability
                for c in range(0, len(clique_list), 1): 
                    lambda_vec[k] *= binary_phi_calc(x_mat_t1[row, col], x_mat_t1[clique_list[c]], theta_00, theta_01)
                   
            #Normalize the probabilities
            lambda_vec = lambda_vec / np.sum(lambda_vec)

            #Sample from categorical distribution
            curr_no_of_samples = 1;
            sample_k_array = np.random.multinomial(curr_no_of_samples, lambda_vec) #returned value contains number of sample in each of the k categories
            
            #Assign the index associated with the value 1 to current row and column
            x_mat_t1[row, col] = function_get_index(sample_k_array, curr_no_of_samples)

    return x_mat_t1;

#Compute gibbs sampling for a binary image
def wrapper_gibbs_sampling_categorical_2d(no_of_time_samples, noisy_image, burn_in_count, sample_space_vec, row_size, col_size):
    #

    debug_flag = 0;

    #Plotting variables
    color_option = 'gray'
    #
    fig_no_of_rows = np.floor(np.sqrt(no_of_time_samples)); #Identifying subplot layout to show all samples simultaneously
    fig_no_of_cols = np.ceil(no_of_time_samples / fig_no_of_rows);

    #Initialize the state vector (containing the states of the nodes in the undirected graph)
    x_mat_all_samples = np.zeros([no_of_time_samples, row_size, col_size])

    #Specify the initial state of the state vector to begin Gibb's sampling
    #x_mat_t0 = np.zeros([row_size, col_size])
    x_mat_t0 = noisy_image
    
    mu = 0;
    sigma = 1
    
    #Debugging
    if (debug_flag == 1):
        plt.imshow(x_mat_t0, cmap = color_option, interpolation = 'none');
        plt.draw();
        plt.pause(0.01);

    for t in range(1, no_of_time_samples + burn_in_count):
        print 'Sample #', t;
        x_mat_t1 = gibbs_sampling_categorical_2d(x_mat_t0, sample_space_vec)
        
        #Start capturing the samples after burn-in
        if t >= burn_in_count:
            x_mat_all_samples[t - burn_in_count, :, :] = x_mat_t1

            #Debugging
            if (debug_flag == 1):
                plt.subplot(fig_no_of_rows, fig_no_of_cols, t - burn_in_count + 1);
                plt.imshow(x_mat_t1, cmap = color_option);
                plt.draw()
                plt.pause(0.01)

        #Current state becomes the initial state for the next Gibb's sampling
        x_mat_t0 = x_mat_t1;
    return x_mat_all_samples[no_of_time_samples - 1, :, :]
        
                 
if __name__ == '__main__':


       #Get the dataset from the path:
    currDir = os.getcwd()
    dataset_loc = '/Datasets/'
    dataset_name = 'test_binary_image_01.png'
    dataset_file_loc = currDir + dataset_loc + dataset_name
    
    #Time sample and no of variables on a subset Sc
    no_of_time_samples = 10

    #Gibbs sampling parameter
    burn_in_count = 20;

    #Dimension d
    row_size = 100;
    col_size = 100; #In original notation, d -> (row_size x col_size); e.g. 100 x 100 matrix equivalent to D = 10,000
       
    noisy_image = plt.imread(dataset_file_loc)

    #Parameters to generate prior
    row_size, col_size = noisy_image.shape
    #Sample space of x_d (where x_d is the d'th element of x_vec)
    sample_space_vec = np.array([0, 1]);
          
    #Generate
    x_mat_all_samples = wrapper_gibbs_sampling_categorical_2d(no_of_time_samples, noisy_image, burn_in_count, sample_space_vec, row_size, col_size)

   