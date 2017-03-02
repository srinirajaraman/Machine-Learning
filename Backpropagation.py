import gc as gc
import os
import random
import numpy as np
import string

#import matplotlib.pyplot as plt
import math as ma       
from ActivationFunctions import function_d_sigmoid, function_logistic
from collections import namedtuple
from ImportData import function_import_mnist

def function_to_reshape(vec, vec_size):
    return np.reshape(vec, (vec_size, 1))

def function_rand(a, b):
    return ((b - a) * random.random() + a)  

def PlotData(x, y):
    plt.plot(x, y, '*')
    plt.draw()
    plt.pause(0.000001)
    plt.xlim([0, 80])
    plt.ylim([-5, 5])
    plt.xlabel('Per - Epoch')
    plt.ylabel('Error ')

def function_initialise( numberOfInputNodes, numberOfHiddenNodes, numberOfOutputNodes):

    backPropStruct = namedtuple("backProp", "N M O Y_i Y_j Y_k V_i V_j V_k W_ji W_kj")

    backPropStruct.N = numberOfInputNodes + 1
    backPropStruct.M = numberOfHiddenNodes + 1
    backPropStruct.O = numberOfOutputNodes

    #Create  activation nodes
    backPropStruct.Y_i = np.zeros([backPropStruct.N, 1]) 
    backPropStruct.Y_j = np.zeros([backPropStruct.M, 1])
    backPropStruct.Y_k = np.zeros([backPropStruct.O, 1])
     
    backPropStruct.Y_i[0 :] = 1
    backPropStruct.Y_j[0 :] = 1
      
    #Create netinput nodes
    backPropStruct.V_j = np.zeros([backPropStruct.M, 1])
    backPropStruct.V_k =  np.zeros([backPropStruct.O, 1])
    
    #Create weights for the network
    backPropStruct.W_ji = np.zeros(shape = (backPropStruct.M, backPropStruct.N))
    backPropStruct.W_kj = np.zeros(shape = (backPropStruct.O, backPropStruct.M))
    
    backPropStruct.C_ji = np.zeros(shape = (backPropStruct.M, backPropStruct.N))
    backPropStruct.C_kj = np.zeros(shape = (backPropStruct.O, backPropStruct.M))

    
    backPropStruct.W_ji[0][:] = 1
    #Create random weights for inputs and outputs
    for j in range(0, backPropStruct.M):
        for i in range(1, backPropStruct.N):
            backPropStruct.W_ji [j][i] = function_rand(-2, 2)

    backPropStruct.W_kj[0][:] = 1
    for k in range(0, backPropStruct.O):
        for j in range(1, backPropStruct.M):
            backPropStruct.W_kj [k][j] = function_rand(-2, 2)
    
    return backPropStruct

def function_feed_forward( inputData, backPropStruct ):
    #Create activation function for inputNodes
    backPropStruct.Y_i[1 :] = inputData
    
    #Compute hidden activation function from input layer
    backPropStruct.V_j =  np.dot(backPropStruct.W_ji, backPropStruct.Y_i)
    backPropStruct.Y_j[1 :]  = function_logistic(backPropStruct.V_j[1 :])
            
    #Compute output activation function from hidden layer 
    backPropStruct.V_k = np.dot(backPropStruct.W_kj, backPropStruct.Y_j )
    backPropStruct.Y_k = function_logistic(backPropStruct.V_k)

    return backPropStruct

def function_back_propagate( d_k , learningRate, momentumFactor, backPropStruct):

    #Eqn 4.21 from page 133
    e_k = (d_k  - backPropStruct.Y_k) 
        
    #Compute error 
    del_k = np.multiply(e_k, function_d_sigmoid(backPropStruct.Y_k)) 

    del_j = np.multiply(function_d_sigmoid(backPropStruct.Y_j), np.dot(del_k.T,  backPropStruct.W_kj).T)

    #Eqn 4.15 of page 131 to compute delta j to update weights from ouput to hidden layer
    del_W_kj = (  np.dot(backPropStruct.Y_j, del_k.T).T)

    #Eqn 4.26 of page 133 to compute delta j
    delW_ji = ( np.dot(del_j, backPropStruct.Y_i.T))

    #Eqn 4.15 of page 131 to compute delta j to update weights from ouput to hidden layer
    backPropStruct.W_kj[:][:] = backPropStruct.W_kj[:][:] +  learningRate * del_W_kj + momentumFactor * backPropStruct.C_kj[:][:]  
    backPropStruct.C_kj = del_W_kj
    
    #Eqn 4.27 of page 134 to compute delta j to update weights from hidden layer to input layer
    backPropStruct.W_ji[:][:] = backPropStruct.W_ji[:][:] + learningRate * delW_ji + momentumFactor * backPropStruct.C_ji[:][:]  
    backPropStruct.C_ji = delW_ji


    totalErr = np.sum((e_k**2))

    return totalErr, backPropStruct

def function_XOR_train(input_images, expected_labels, arrayRepNum, learningRate, momentumFactor, maxIterations, backPropStruct, logger, log_path, trainingInd):
    
    logger = open(log_path, 'w')

    for dataInd in range(0, len(input_images)):   
         
        inputVec = function_to_reshape(np.transpose(input_images[dataInd]), len(input_images[dataInd]))
        print'Training for the', dataInd, 'image started'
        expected_output = expected_labels[dataInd]

        for epoch in range(0, maxIterations):
            backPropStruct = function_feed_forward(inputVec, backPropStruct)
            totalErr, backPropStruct = function_back_propagate( expected_output , learningRate, momentumFactor, backPropStruct)
            logger.write('image :' + str(dataInd) +  ' ' + 'Current epoch:' + str(epoch) + ' ' + 'total error :' +  str(totalErr))
            logger.write('\n')
            #PlotData(epoch, totalErr)

    logger = open(log_path, 'r')
    logger.read()
    logger.close()
    return backPropStruct 

def function_BP_train(input_images, expected_labels, arrayRepNum, learningRate, momentumFactor, maxIterations, backPropStruct, logger, log_path, trainingInd):
    
    for dataInd in range(0, len(input_images)):   
         
        inputVec = np.transpose(input_images[[dataInd]])
        print'Training for the', dataInd, 'image started'
        expected_output = np.transpose(arrayRepNum[[expected_labels[dataInd]]])

        for epoch in range(0, maxIterations):
            backPropStruct = function_feed_forward(inputVec, backPropStruct)
            totalErr, backPropStruct = function_back_propagate( expected_output , learningRate, momentumFactor, backPropStruct)
            logger.write('trainingIndex:' + '' + str(trainingInd) + ' ' + 'image :' + str(dataInd) +  ' ' + 'Current epoch:' + str(epoch) + ' ' + 'total error :' +  str(totalErr))
            logger.write('\n')
            #PlotData(epoch, totalErr)

    logger = open(log_path, 'r')
    logger.read()
    #logger.close()
    return backPropStruct 

def function_BP_test(patterns, backPropStruct):
  
    inputVec = (function_to_reshape(patterns, patterns.size))
    return function_feed_forward(inputVec, backPropStruct)

def function_XOR_test(patterns, backPropStruct):
  
    inputVec = (function_to_reshape(patterns, len(patterns)))
    return function_feed_forward(inputVec, backPropStruct)
       
#currDir, newpath, dataset_loc,file_name
def function_back_propagation(currDir, path, dataset_loc, dataset_name, file_name):

    train_set, valid_set, test_set = function_import_mnist(currDir, dataset_loc, dataset_name)
    arrayRepNum = np.identity(10, float)

    noOfTrainingset  = 100

    maxIterations = 100
    learningRate = 0.01
    momentumFactor = 0.1
    local_number_input_nodes  = 784
    local_number_hidden_nodes = 100
    local_number_output_nodes = 10
    noOfTrainingExamples = 200


    log_path = path + file_name
    logger = open(log_path , 'a+')
    logger = open(log_path , 'w')

    backPropStruct = function_initialise(local_number_input_nodes , local_number_hidden_nodes , local_number_output_nodes)
    for trainingInd in range(0, noOfTrainingset):
        backPropStruct = function_BP_train(train_set[0][0:noOfTrainingExamples], train_set[1][0:noOfTrainingExamples], arrayRepNum, learningRate, momentumFactor, maxIterations, backPropStruct, logger, log_path, trainingInd)
    backPropStruct = function_BP_test(test_set[0][0], backPropStruct)
    
    print 'Total Output', (backPropStruct.Y_k)
    logger.write('Total Output' + str(backPropStruct.Y_k))
    logger = open(log_path, 'r')
    logger.read()
    logger.close()

def function_XOR(path, file_name):
    patt = [[0,0],[0,1],[1,0],[1,1]]
    expectedRes = [0, 1, 1, 0]
    arrayRepNum = expectedRes
    
    trainIterations = 10 
    maxIterations = 5000
    learningRate = 0.1
    momentumFactor = 0.01
    local_number_input_nodes  = 2
    local_number_hidden_nodes = 2
    local_number_output_nodes = 1
    log_path = path + file_name
    logger = open(log_path , 'a')
    logger = open(log_path , 'w')
    
    backPropStruct = function_initialise(local_number_input_nodes , local_number_hidden_nodes , local_number_output_nodes)
    for i in range(0, trainIterations):
        backPropStruct = function_XOR_train(patt, expectedRes, arrayRepNum, learningRate, momentumFactor, maxIterations, backPropStruct, logger, log_path)
    
    input = [1,1 ]
    backPropStruct = function_XOR_test(input, backPropStruct)
    print input ,'-', backPropStruct.Y_k
    
    input = [0,1 ]
    backPropStruct = function_XOR_test(input, backPropStruct)
    print input ,'-', backPropStruct.Y_k
    
    input = [1,0 ]
    backPropStruct = function_XOR_test(input, backPropStruct)
    print input ,'-', backPropStruct.Y_k

    input = [0, 0 ]
    backPropStruct = function_XOR_test(input, backPropStruct)
    print input ,'-', backPropStruct.Y_k



if __name__ == '__main__':
    file_name = '\BP_logger.txt'    
    currDir = os.getcwd()
    newpath = currDir + '\Log' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    dataset_loc = '/Datasets/'
    dataset_name = 'mnist.pkl.gz'
    function_back_propagation(currDir, newpath, dataset_loc, dataset_name, file_name)
    #function_XOR(path, file_name)