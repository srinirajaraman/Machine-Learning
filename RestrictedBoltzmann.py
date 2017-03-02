from ImportData import function_import_mnist
import os
import sys
import numpy as np
from utils import *
from ActivationFunctions  import function_logistic
from random import randint
from collections import namedtuple
from Plot import get_value
from PIL import Image as Image
from visual import * # must import visual or vis first
from visual.graph import *	# import graphing features 
from matplotlib import pylab as mplt
import math as m
from sympy.physics.quantum import TensorProduct


def function_initialise(visible_nodes, hidden_nodes, output_nodes):

    rbm_struct = namedtuple("rbm_struct", "v_val h_val o_val visible_nodes hidden_states output_nodes weights rng bj ci ")

    rbm_struct.v_val = visible_nodes
    rbm_struct.h_val  = hidden_nodes 
    rbm_struct.o_val  = output_nodes
    rbm_struct.visible_states = np.zeros([visible_nodes, 1]) 
    rbm_struct.hidden_states = np.zeros([hidden_nodes, 1]) 
    rbm_struct.output_nodes = np.zeros([output_nodes, 1]) 

    rbm_struct.weights = np.random.uniform(-1, 1, size =(visible_nodes, output_nodes, hidden_nodes))
    rbm_struct.rng = np.random.RandomState(1234)

    #bias 
    rbm_struct.bj = np.zeros([visible_nodes, 1]) 
    rbm_struct.ci = np.zeros([hidden_nodes, 1]) 

    #rbm_struct.plt = gdots(color=color.red)	# a graphics curve
    return rbm_struct

def rbm_update(inputimage, output_image, rbm_struct, learning_rate):

    rbm_struct.visible_states = inputimage
    rbm_struct.output_nodes = output_image

    p_h_c_v_zero, sampled_p_h_c_v_zero = get_h_given_v(rbm_struct, rbm_struct.visible_states, rbm_struct.output_nodes)

    [p_v_c_h_k, sampled_p_v_c_h_k, p_h_c_v_k, sampled_p_h_c_v_k] = gibbs_sampling(rbm_struct, rbm_struct.visible_states, sampled_p_h_c_v_zero, rbm_struct.output_nodes)

    temp = rbm_struct.visible_states * p_v_c_h_k.T 

    a = np.reshape(temp, (rbm_struct.v_val, rbm_struct.o_val, 1))

    b = np.reshape(p_h_c_v_zero, (rbm_struct.h_val, 1))

    c = a * b.T

    temp1 = rbm_struct.visible_states.T * sampled_p_v_c_h_k
    temp1 = temp1.T

    a1 = np.reshape(temp1, (rbm_struct.v_val, rbm_struct.o_val, 1))

    b1 = np.reshape(p_h_c_v_k , (rbm_struct.h_val, 1))

    c1 = a1 * b1.T    

    rbm_struct.weights += learning_rate * (c - c1)
 
    return rbm_struct


def get_h_given_v(rbm_struct, x_i_input_image, y_j_output_image):
    
    a = np.tensordot(np.tensordot(x_i_input_image.T, rbm_struct.weights[:, :, :], 1).T, y_j_output_image) 
    
    p_h_c_v_k = function_logistic(a)
    sampled_p_h_c_v_k = rbm_struct.rng.binomial(size=p_h_c_v_k.shape,   # discrete: binomial
                                    n=1,
                                    p=p_h_c_v_k)

    return [p_h_c_v_k, sampled_p_h_c_v_k]

def get_y_given_h(rbm_struct, x_i_input_image, h0_sample):
    
    a = np.tensordot(x_i_input_image.T, rbm_struct.weights[:, :, :], 1).T
    b = np.tensordot(h0_sample, a, 1)
    p_v_c_h_k = (function_logistic(b))
    sampled_p_v_c_h_k = rbm_struct.rng.binomial(size=p_v_c_h_k.shape,   # discrete: binomial
                                        n=1,
                                        p=p_v_c_h_k)
        
    return [p_v_c_h_k, sampled_p_v_c_h_k]
   
def gibbs_sampling(rbm_struct, x_i_input_image, h0_sample, y_j_output_image):
    p_v_c_h_k, sampled_p_v_c_h_k = get_y_given_h(rbm_struct, x_i_input_image, h0_sample)
    p_h_c_v_k, sampled_p_h_c_v_k = get_h_given_v(rbm_struct, x_i_input_image, sampled_p_v_c_h_k)

    return [p_v_c_h_k, sampled_p_v_c_h_k, p_h_c_v_k, sampled_p_h_c_v_k]
    
def get_reconstruction_cross_entropy(rbm_struct):
    
    pre_sigmoid_activation_h =  np.tensordot(rbm_struct.visible_states.T, rbm_struct.weights[:, :, :], 1).T
    #pre_sigmoid_activation_h =  np.dot(rbm_struct.weights , rbm_struct.visible_states) 
    sigmoid_activation_h = function_logistic(pre_sigmoid_activation_h )

    pre_sigmoid_activation_v = np.dot(rbm_struct.weights.T , sigmoid_activation_h)  + rbm_struct.bj 
    sigmoid_activation_v = function_logistic(pre_sigmoid_activation_v)

    cross_entropy =  - np.mean(
        np.sum(rbm_struct.visible_states * np.log(sigmoid_activation_v) +
        (1 - rbm_struct.visible_states) * np.log(1 - sigmoid_activation_v),
                    axis=1))
        
    return cross_entropy

def reconstruct(rbm_struct, test_image):
    
    pre_sigmoid_activation_h =  np.dot(rbm_struct.weights, test_image) #+  rbm_struct.ci
    sigmoid_activation_h = function_logistic(pre_sigmoid_activation_h )

    pre_sigmoid_activation_v = np.dot(rbm_struct.weights[:,785:795].T , sigmoid_activation_h)  #+ rbm_struct.bj 
    sigmoid_activation_v = function_logistic(pre_sigmoid_activation_v)

    return sigmoid_activation_v

def plot_weight_matrix(rbm_struct, weight_matrix, input_nodes, hidden_nodes):

    #mat_to_plot = np.zeros([n_hidden_nodes, n_visible_nodes-1])
    sub_mat_val = int(m.sqrt(input_nodes))
    total_mat_val = int(m.sqrt(hidden_nodes))
    
    reshaped_mat = np.zeros([hidden_nodes, sub_mat_val, sub_mat_val])
    plot_mat = np.zeros([total_mat_val, total_mat_val,sub_mat_val, sub_mat_val])
    
    for i in range(0, input_nodes):
        a = np.reshape(rbm_struct.weights[i][1][:], (529, 1))
        b = np.reshape(a, (23, 23))
        mplt.imshow(b, cmap = 'gray')
        mplt.draw()
        mplt.pause(0.001)
        a = 'weight_matrix' + str(i)+'.png'
        mplt.savefig(a)
    #for i in range(0, hidden_nodes):
    #    mat_to_plot = weight_matrix[i][1][:input_nodes]
    #    mat_to_plot = np.reshape(mat_to_plot, (rbm_struct.h_val, 1))
    #    reshaped_mat[i][:][:] = np.reshape(mat_to_plot, (sub_mat_val, sub_mat_val))
        
    plot_mat = np.reshape(reshaped_mat, (total_mat_val, total_mat_val, sub_mat_val, sub_mat_val))
    for i in range(0, total_mat_val):
        for j in range(0, total_mat_val):
            mplt.imshow(plot_mat[i][j], cmap = 'gray')
            mplt.show()    
        
    
def rbm_train(input_images, expected_labels, arrayRepNum, learningRate, maxIterations, rbm_struct, logger, log_path, training_ind):
    for i in range(0, maxIterations):
        image_ind = randint(0, len(input_images) - 1)
        input_image = np.transpose(input_images[[image_ind]])
        expected_label = np.transpose(arrayRepNum[expected_labels[[image_ind]]])
        rbm_struct = rbm_update(input_image, expected_label, rbm_struct, learningRate)
        #cost = get_reconstruction_cross_entropy(rbm_struct)
        print ('trainingIndex:'  + str(training_ind) + ' ' + ' ' + 'Current epoch:' + str(i) + '\t' + 'image:' + str(image_ind) + '\t'  ) #+ 'total error: ' +  str(cost))
        logger.write('trainingIndex:'  + str(training_ind) + ' ' + ' ' + 'Current epoch:' + str(i) + '\t' + 'image:' + str(image_ind) + '\t') #  + 'total error: ' +  str(cost))
        logger.write('\n')
        #rbm_struct.plt.plot(pos = (i, cost))
        #rate(100)

    logger = open(log_path, 'r')
    logger.read()
    
    return rbm_struct 


def test_rbm():

    file_name = '\RBM_logger.txt'    
    currDir = os.getcwd()
    log_path = currDir + '\Log' 
    fig_path = currDir + '\Figures' 
    if not os.path.exists(log_path ):
        os.makedirs(log_path )
    if not os.path.exists(fig_path ):
        os.makedirs(fig_path )
    fig_path =  fig_path + '/' + 'firstimage.png'
    dataset_loc = '/Datasets/'
    dataset_name = 'mnist.pkl.gz'
    
    train_set, valid_set, test_set = function_import_mnist(currDir, dataset_loc, dataset_name)

    # construct RBM
    learning_rate = 0.001
    input_nodes = 784
    hidden_nodes = 529
    output_nodes = 10
    training_epochs= 1
    max_iterations = 10000

    rbm_struct = function_initialise(input_nodes, hidden_nodes, output_nodes)

    noOfTrainingExamples = 300

    arrayRepNum = np.identity(10, float)
    log_path = log_path  + file_name
    logger = open(log_path , 'a+')
    logger = open(log_path , 'w')
    
    # train
    for epoch in xrange(training_epochs):
        rbm_struct = rbm_train(train_set[0][0:noOfTrainingExamples], train_set[1][0:noOfTrainingExamples], arrayRepNum, learning_rate, max_iterations, rbm_struct, logger, log_path, epoch)
      
    #test_images = test_set[0][0:noOfTrainingExamples]
    
    #for i in range(0, 4):
    #    test_image = np.transpose(test_images[[i]])
    #    val = reconstruct(rbm_struct, test_image)
    #    logger.write('Total Output' + str(val))
    #    logger.write('\n')

    
    #plot_weight_matrix(rbm_struct.weights, input_nodes-1, hidden_nodes)
    plot_weight_matrix(rbm_struct, rbm_struct.weights, input_nodes, hidden_nodes)

    logger = open(log_path, 'r')
    logger.read()
    logger.close()
    #shape = (28, 28)
    #extent = [-130,130,0,10]
    #mplt.imshow(rbm_struct.weights[0:rbm_struct.n_visible_nodes-1], cmap = 'gray')
    
    #mplt.draw()
    #mplt.pause(0.01)
    #mplt.savefig(fig_path)
    print val
   

if __name__ == "__main__":
    test_rbm()