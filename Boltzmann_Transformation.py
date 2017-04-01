'''
This python code is to train the ML model to recognise affine transformation based on Probabilistic graph model
'''
import threading
from multiprocessing import Process, Pipe, Lock
import scipy as sci
import os 
from os import listdir
import sys
import numpy as np
#MB April 27, 2016
#from utils import *
from ActivationFunctions  import function_logistic
from random import randint
from collections import namedtuple
#MB April 27, 2016
#from Plot import get_value
from PIL import Image as Image
from visual import * # must import visual or vis first
from visual.graph import *	# import graphing features 
from matplotlib import pylab as mplt
import math as m
from sympy.physics.quantum import TensorProduct
#MB April 27, 2016--begin
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import time
#MB ends

#MB April 27, 2016-begins
def generate_image_small_square(row, col, patch_size, patch_loc_row, patch_loc_col, debugFlag):
    '''
        Generates a small image of size "row x col" with a patch of size "patch_size x patch_size" at index (patch_loc_row, patch_loc_col)
    '''
    test_image = np.zeros([row, col])
    test_image[patch_loc_row : patch_loc_row + patch_size, patch_loc_col : patch_loc_col + patch_size] = 1.0

    if debugFlag == 1:
        fig_handle = plt.figure
        plt.imshow(test_image)
        plt.title('Generated image')
        plt.show()

    return test_image

def apply_affine_transform(input_image, theta, debugFlag):
    '''
        Currently we will only do simple constant in place rotation. But we can generalize this to create 
        a complete affine transformation including translation, rotation, stretching, etc. in addition to translation
    '''
    #Initialize
    rows, cols = np.shape(input_image)

    #Transformation matrix (for rotation only)
    #angle_radians = np.radians(theta)
    #c, s = np.cos(angle_radians), np.sin(angle_radians)
    #transformation_matrix = np.array([[c, -s], [s, c]])
    #transformation_matrix = np.array([[c, -s], [s, c]])
    #
    #Need to use the transformation matrix provided by opencv for using warpAffine
    rotation_center = (cols/2, rows/2)
    scale = 1
    transformation_matrix = cv2.getRotationMatrix2D(rotation_center, theta, 1)

    #Apply the transformation matrix to the input image
    output_image = cv2.warpAffine(input_image, transformation_matrix, np.shape(input_image))

    #Debugging
    if debugFlag == 1:
        fig = plt.figure
        #input image
        plt.subplot(121)
        plt.imshow(input_image)
        plt.title("Input image")
        plt.savefig("inutimage.png")
        #Output transformed image
        plt.subplot(122)
        plt.imshow(output_image)
        plt.title("Output transformed image")
        plt.draw()
        #plt.show(block=False)
        #plt.draw()
        #Add a little pause to show the image
        #time.sleep(5)
        #Close the figure window
        plt.pause(0.001)

    return input_image, output_image

def generate_rotating_training_data():
    '''
        Generates a training data that containing a rotating square in the middle
        This idea can be extended to generate objects that translate, stretch, etc.
    '''
    debugFlag = 0;
    #Image size
    row = 12
    col = 12
    square_size = 1
    #Initialize output
    train_set = np.zeros([36, row*col])

    #Generate an initial image
    train_index = 0;
    input_image = generate_image_small_square(row, col, 8, square_size , square_size, debugFlag)
    
    train_set[train_index, :] = np.reshape(input_image, (row*col))
    prev_image = input_image
    timage = np.zeros([row, col])
    #Affine transform of the image
    for theta in range(10, 359, 10):
        train_index = train_index + 1
        if (train_index == 1):
            test_image, output_image = apply_affine_transform(prev_image, theta, debugFlag)
            timage =test_image
        else:
            input_image, output_image = apply_affine_transform(prev_image, theta, debugFlag)
        print(train_index)
        train_set[train_index, :] = np.reshape(output_image, (row*col))
        prev_image = output_image

    return timage,  train_set

def function_reshape_arr(test_arr, val1, val2, val3):
    if(val2 == 0 & val3 == 0):
        return np.reshape(test_arr, (val1))
    elif(val3 == 0):
        return np.reshape(test_arr, (val1, val2))
    else:
        return np.reshape(test_arr, (val1, val2, val3))

#Initialize the network
def function_initialise(visible_nodes, hidden_nodes, output_nodes, learning_rate):

    rbm_struct = namedtuple("rbm_struct", "v_val h_val o_val visible_nodes hidden_states output_nodes weights learning_rate rng bj ci ")

    rbm_struct.learning_rate = learning_rate
    rbm_struct.v_val = visible_nodes
    rbm_struct.h_val = hidden_nodes 
    rbm_struct.o_val = output_nodes
    rbm_struct.visible_states = np.zeros([visible_nodes, 1]) 
    rbm_struct.hidden_states = np.zeros([hidden_nodes, 1]) 
    rbm_struct.output_nodes = np.zeros([output_nodes, 1]) 

    rbm_struct.weights = np.random.uniform(0, 2, size =(visible_nodes, output_nodes, hidden_nodes))
    rbm_struct.rng = np.random.RandomState(1234)

    #bias 
    rbm_struct.bj = np.zeros([visible_nodes, 1]) 
    rbm_struct.ci = np.zeros([hidden_nodes, 1]) 

    #rbm_struct.plt = gdots(color=color.red)	# a graphics curve
    return rbm_struct

def rbm_update(rbm_struct, inputimage, output_image):

    rbm_struct.visible_states = inputimage

    rbm_struct.output_nodes = output_image

	#Compute the probability of hidden states
    p_h_c_v_zero, sampled_p_h_c_v_zero = get_h_given_v(rbm_struct, rbm_struct.visible_states, rbm_struct.output_nodes)

	#Gibbs sampling and learning is achieved through contrastive divergence (K = 1 )
    [p_v_c_h_k, sampled_p_v_c_h_k, p_h_c_v_k, sampled_p_h_c_v_k] = gibbs_sampling(rbm_struct, rbm_struct.visible_states, sampled_p_h_c_v_zero, rbm_struct.output_nodes)

    temp = rbm_struct.visible_states * output_image.T 

    a = function_reshape_arr(temp, rbm_struct.v_val, rbm_struct.o_val, 1)

    b = function_reshape_arr(p_h_c_v_zero, rbm_struct.h_val, 1, 0)

    c = a * b.T

    temp1 = output_image * sampled_p_v_c_h_k.T
    
    a1 = function_reshape_arr(temp1, rbm_struct.v_val, rbm_struct.o_val, 1)

    b1 = function_reshape_arr(p_h_c_v_k , rbm_struct.h_val, 1, 0)

    c1 = a1 * b1.T    

	#Parameters updation
    rbm_struct.weights += rbm_struct.learning_rate * (c - c1)
    rbm_struct.bj += rbm_struct.learning_rate * np.mean(rbm_struct.visible_states - sampled_p_v_c_h_k, axis=0)
    rbm_struct.ci += rbm_struct.learning_rate * np.mean(p_h_c_v_zero - p_h_c_v_k, axis=0)
    
#Compute the probability of hidden states

def get_h_given_v(rbm_struct, x_i_input_image, y_j_output_image):
    
    a = np.tensordot(np.tensordot(x_i_input_image.T, rbm_struct.weights[:, :, :], 1).T, y_j_output_image) 
    a = function_reshape_arr(a, rbm_struct.h_val, 1, 0)
    p_h_c_v_k = function_logistic(a + rbm_struct.ci)
    p_h_c_v_k = function_reshape_arr(p_h_c_v_k, rbm_struct.h_val, 0, 0)
    sampled_p_h_c_v_k = rbm_struct.rng.binomial(size=p_h_c_v_k.shape, n=1, p=p_h_c_v_k)

    return [p_h_c_v_k, sampled_p_h_c_v_k]

	
	#Compute the probability of output states
def get_y_given_h(rbm_struct, x_i_input_image, h0_sample):
    
    a = np.tensordot(x_i_input_image.T, rbm_struct.weights[:, :, :], 1).T
    b = np.tensordot(h0_sample, a, 1)
    p_v_c_h_k = function_logistic(b + rbm_struct.bj)
    sampled_p_v_c_h_k = rbm_struct.rng.binomial(size=p_v_c_h_k.shape, n=1, p=p_v_c_h_k)
        
    return [p_v_c_h_k, sampled_p_v_c_h_k]
   

def gibbs_sampling( rbm_struct, x_i_input_image, h0_sample, y_j_output_image):
    p_v_c_h_k, sampled_p_v_c_h_k = get_y_given_h(rbm_struct, x_i_input_image, h0_sample)
    p_h_c_v_k, sampled_p_h_c_v_k = get_h_given_v(rbm_struct, x_i_input_image, sampled_p_v_c_h_k)

    return [p_v_c_h_k, sampled_p_v_c_h_k, p_h_c_v_k, sampled_p_h_c_v_k]

#Testing the model 	
def reconstruct(rbm_struct, input_images, test_image):
    
    for i in range(0, 1):
        input_image = np.transpose(input_images[[i]])
        expected_label = np.transpose(input_images[[i+1]])
        
    hidden_activation, sampled_p_h_c_x_y = get_h_given_v(rbm_struct, input_image, expected_label)
    
    p_y_h_v, sampled_p_y_h_x = get_y_given_h(rbm_struct, test_image, hidden_activation)
    
    return p_y_h_v

def plot_weight_matrix(figpath, weight_matrix):

    for i in range(0, rbm_struct.h_val):
        a = np.reshape(weight_matrix[i][i][:], (rbm_struct.h_val, 1))
        a = np.reshape(a, (5, 5))
        mplt.imshow(a, cmap = 'gray')
        mplt.draw()
        mplt.pause(0.001)
        mplt.savefig(figpath + '/WeightMatrix/' + 'Weightmatrix_node_' + str(i) + '.png')

#	Training the model 
def rbm_train(rbm_struct, input_images, training_ind):
    for i in range(1, np.shape(input_images)[0] - 1):
        input_image = np.transpose(input_images[[i]])
        expected_label = np.transpose(input_images[[i+1]])
        rbm_update(rbm_struct, input_image, expected_label)
        print 'current training' , i
    return rbm_struct 

def function_run(rbm_struct, training_epoch, train_set):
    print 'inside fnc run'
    for epoch in xrange(training_epoch):
        print ('current epoch:'  + str(epoch))
        rbm_struct = rbm_train(rbm_struct, train_set, epoch)
    return rbm_struct 

def test_rbm():

    file_name = '\RBM_logger.txt'    
    currDir = os.getcwd()
    fig_path = currDir + '\Figures' 
    if not os.path.exists(fig_path ):
        os.makedirs(fig_path )
    weight_path = fig_path + '\WeightMatrix' 
    if not os.path.exists(weight_path ):
        os.makedirs(weight_path )

    #f = listdir(fig_path)
    #test = fig_path + '/' + f[2]
    #im = Image.open(test)
    #for i in range(0, 6):
    #    if( i ==0):
    #        im.rotate(20).save(fig_path + '/' + str(i) + '.jpg')
    #    else:
    #        im = Image.open(fig_path + '/' + str(i - 1) + '.jpg')
    #        im.rotate(20).save(fig_path + '/' + str(i) + '.jpg')

    #MB April 27, 2016
    #train_set, valid_set, test_set = function_import_mnist(currDir, dataset_loc, dataset_name)
    
    
    # construct RBM
    learning_rate = 0.001
    input_nodes = 144
    hidden_nodes = 25
    output_nodes = 144
    training_epochs= 1
    
    global rbm_struct    
    rbm_struct = function_initialise(input_nodes, hidden_nodes, output_nodes, learning_rate) 

    test_image, train_set = generate_rotating_training_data()
    test = np.reshape(test_image, (input_nodes, 1))    
    rbm_struct = function_run(rbm_struct, training_epochs, train_set)
      
# Get process results from the output queue
    plot_weight_matrix(fig_path, rbm_struct.weights)
    res = reconstruct(rbm_struct, train_set, test)
    res = res.T
    res_val = np.reshape(res, (12, 12))
    
    mplt.imshow(res_val, cmap = 'gray')
    mplt.show()
    mplt.savefig(fig_path + '/figure_result.png')
    mplt.close()

    
if __name__ == "__main__":
    test_rbm()  
