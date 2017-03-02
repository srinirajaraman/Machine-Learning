'''
This script is to test the Recurrent neural network
Source  : deep learning book from MIT - GoodFellow
Chapter : 10
Eqns    : 10.8 to 10.28 from Page 373 - 386
Date    : 2/2/17
Training Method: SGD Min-Batch of size 5
Problem Statement: Train the model with XOR logic and test the same. 
'''

import numpy as np

class Rnn(object):
    """description of class"""
    def __init__(self,ip_nodes, hidden_nodes, op_nodes, learn_rate, timestep, reg_param, momentum_rate):
        self.ip_nodes, self.hidden_nodes, self.output_nodes, self.learning_rate, \
            self.timestep, self.reg_param, self.momentum_rate = ip_nodes, hidden_nodes, op_nodes, learn_rate,\
            timestep, reg_param, momentum_rate
        self.x_t = {}
        self.a_t = {}
        self.h_t = {}
        self.o_t = {}
        self.y_hat = {}
        self.grad_L_H = np.zeros([self.timestep, self.hidden_nodes, 1])
        #MB changes
        #self.b = np.zeros([self.hidden_nodes, 1])
        self.b = np.ones([self.hidden_nodes, 1])
        #MB changes
        #self.c = np.zeros([self.output_nodes, 1])
        self.c = np.zeros([self.output_nodes, 1]);
        self.U = np.random.randn(self.hidden_nodes, self.ip_nodes)  * 0.01 
        self.W = np.random.randn(self.hidden_nodes, self.hidden_nodes)  * 0.01
        #self.W = np.identity(self.hidden_nodes)
        self.V = np.random.randn(self.output_nodes, self.hidden_nodes)  * 0.01
        self.cV, self.cW, self.cU = np.zeros(np.shape(self.V)), np.zeros(np.shape(self.W)), np.zeros(np.shape(self.U))

    #Feed forward process for the neural network
    def feed_forward(self, input, t, epoch_ind):
        self.x_t[t] = input

        #MB changes
        #if epoch_ind == 0 and t -1 == -1:
        #    self.h_t[-1] = np.tile(0.5, [rnn_obj.hidden_nodes, 1])
        #elif epoch_ind > 0:
        #    self.h_t[-1] = self.h_t[len(x_val) - 1]

        #Eqns 10.8; net activation at input layer
        self.a_t[t] = self.b + np.dot(self.W, self.h_t[t -1]) + np.dot(self.U, self.x_t[t])

        #Eqns 10.9; net activation at hidden layer
        self.h_t[t] = np.tanh(self.a_t[t])

        #Eqns 10.10; net activation at o/p layer
        self.o_t[t] = self.c + np.dot(self.V, self.h_t[t])

        #Eqns 10.11; o/p from softmax
        self.y_hat[t] = np.exp(self.o_t[t])/ np.sum(np.exp(self.o_t[t]))
        
        
    #Back propagation
    def back_prop(self, target, t):
        grad_L_O = np.copy(self.y_hat[t])
        #Eqn 10.18
        if target == 0:
            grad_L_O -= np.array([[1], [0]])
        else:
            grad_L_O -= np.array([[0], [1]])
       
        
        h_sq_diag = np.identity(self.hidden_nodes) * (1 - self.h_t[t+1] **2)
        
        #Eqn 10.20
        grad_H = np.dot(np.dot(self.W.T, h_sq_diag), self.grad_L_H[t+1])  + np.dot(self.V.T, grad_L_O)
        self.grad_L_H[t] = grad_H

        #Precomputing the derivative of h(t)
        pre_grad_H = np.identity(self.hidden_nodes) * (1 -self.h_t[t] ** 2)
        
        #Bias updation
        #MB changes    
        #Eqn 10.22
        #self.c += -grad_L_O
        #Eqn 10.23
        #self.b += -np.dot(pre_grad_H , grad_H)

        #Gradient weights computation
        #Eqn 10.24
        grad_L_V = np.dot(grad_L_O, self.h_t[t].T)
        #Eqn 10.26
        grad_L_W = np.dot(pre_grad_H , np.dot(grad_H, self.h_t[t - 1].T))
        #Eqn 10.28
        grad_L_U = np.dot(pre_grad_H , np.dot(grad_H, self.x_t[t].T))

        #Regularization computation
        #delV += self.reg_param * (np.linalg.norm(self.V, ord = 2))
        #delW += self.reg_param * (np.linalg.norm(self.W, ord = 2))
        #delU += self.reg_param * (np.linalg.norm(self.U, ord = 2))

        #Weights updation 
        self.V += -self.learning_rate * grad_L_V # + self.momentum_rate * self.cV
        self.W += -self.learning_rate * grad_L_W #+ self.momentum_rate * self.cW
        self.U += -self.learning_rate * grad_L_U # + self.momentum_rate * self.cU
        #self.cV, self.cW, self.cU = grad_L_V, grad_L_W, grad_L_U
   
    #Feed forward wrapper
    def feed_forward_wrapper(self, x_val, epoch_ind):
        x_len = len(x_val)
        for i in range(0, x_len):
            rnn_obj.feed_forward(x_val[i], i, epoch_ind)
    #Back prop wrapper
    def back_prop_wrapper(self, y_val):
        y_len = len(y_val)
        #MB changes
        #for j in reversed(xrange(y_len, 0)):
        for j in range(y_len-1, 0, -1):
            if j == y_len - 1:
                grad_L_O = np.copy(rnn_obj.y_hat[j])
                if y_val[j] == 0:
                    grad_L_O -= np.array([[1], [0]])
                else:
                    grad_L_O -= np.array([[0], [1]])
                #Eqn 10.19
                rnn_obj.grad_L_H[j] = np.dot(rnn_obj.V.T, grad_L_O)
            else:
                rnn_obj.back_prop(y_val[j], j)
            
    #Testing the network
    def test_xor(self, x_val, yval):
        x_len=len(x_val)
        h_t1 = 0
        b = np.ones([self.hidden_nodes, 1])
        c = np.ones([self.output_nodes, 1])
        
        for i in range(x_len):
            if i - 1 == -1:
                h_t1 = np.tile(0, [self.hidden_nodes, 1])

            #net activation at input layer
            a_t = b + np.dot(self.W, h_t1) + np.dot(self.U, x_val[i])

            #net activation at hidden layer
            h_t = np.tanh(a_t)

            #net activation at o/p layer
            o_t  = c + np.dot(self.V, h_t)

            #o/p from softmax
            
            #y_hat = 1/(1 + np.exp(o_t))
            y_hat = np.exp(o_t)/np.sum(np.exp(o_t))

            #y_hat = np.tanh(o_t)
            h_t1 = h_t
            print 'expected val', yval[i],  'yhat-argmax', np.argmax(y_hat), y_hat
            #print 'yhat max', np.max(y_hat)
            #print 'yhat', y_hat
      

if __name__ == '__main__':
    #Network 
    ip_nodes = 1
    hidden_nodes = 2
    op_nodes = 2
    epoch = 10000  
    train_iter = 10

    #Hyper parameters
    learn_rate = 1e-3
    momentum_rate = learn_rate/10
    reg_param = 0
    #Input data 
    x_val = np.array([0, 0, 1, 1, 0])
    
    x_len = len(x_val)
    y_val = np.array([0,0,1,0,1]) 
    #y_val = np.array([1,0,1, 0, 0])
    timestep = x_len

    #Initialization of the network
    rnn_obj = Rnn(ip_nodes, hidden_nodes, op_nodes, learn_rate, timestep, reg_param, momentum_rate)
    
    
    #Training the network
    #MB changes
    #Initialize the prior input (h^(t-1)) at epoch = 0 and t = 0
    rnn_obj.h_t[-1] = np.tile(0.5, [rnn_obj.hidden_nodes, 1])
    for i in range(train_iter):
        for epoch_ind in range(epoch):

            #MB changes
            if (epoch_ind > 0):
                #Initialize the prior input (h^(t-1)) at epoch > 0 and t = 0
                rnn_obj.h_t[-1] = rnn_obj.h_t[x_len - 1];

            print 'training iteration-', i, 'epoch -', epoch_ind
            #Feed forward 
            rnn_obj.feed_forward_wrapper(x_val, epoch_ind)
            #Back propagation
            rnn_obj.back_prop_wrapper(y_val)
        
    #Testing the network
    #x_val = np.array([1,0,0,1,1,0])
    #x_val = np.array([1,0,1,0,0,0,0,1,1,1,1,0,1,0,1])
    #y_val = np.array([0,1,0,0,0,0,1,1,1,1,0,1,0,1,0])

    print 'o/p layer after testing'
    #MB changes
    x_val_test_seq = np.tile(x_val, 5);
    y_val_test_seq = np.tile(y_val, 5);
    rnn_obj.test_xor(x_val_test_seq, y_val_test_seq)
   
    #MB changes
    #print 'i/p weightmatrix', rnn_obj.U
    #print 'hidden weightmatrix', rnn_obj.W
    #print 'o/p weightmatrix', rnn_obj.V