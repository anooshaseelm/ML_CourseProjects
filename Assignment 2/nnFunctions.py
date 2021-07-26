import numpy as np
from scipy.optimize import minimize
from math import sqrt
import pickle
'''
You need to modify the functions except for initializeWeights() 
'''

def initializeWeights(n_in, n_out):
    '''
    initializeWeights return the random weights for Neural Network given the
    number of node in the input layer and output layer

    Input:
    n_in: number of nodes of the input layer
    n_out: number of nodes of the output layer

    Output:
    W: matrix of random initial weights with size (n_out x (n_in + 1))
    '''
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    '''
    Notice that z can be a scalar, a vector or a matrix
    return the sigmoid of input z (same dimensions as z)
    '''
    # remove the next line and replace it with your code
    s = 1.0 / (1.0 + np.exp(-1.0 * z))
    return s

def nnObjFunction(params, *args):
    '''
    % nnObjFunction computes the value of objective function (cross-entropy
    % with regularization) given the weights and the training data and lambda
    % - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices W1 (weights of connections from
    %     input layer to hidden layer) and W2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of nodes in input layer (not including the bias node)
    % n_hidden: number of nodes in hidden layer (not including the bias node)
    % n_class: number of nodes in output layer (number of classes in
    %     classification problem
    % train_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of the corresponding instance 
    % train_label: the vector of true labels of training instances. Each entry
    %     in the vector represents the truee label of its corresponding training instance.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector (not a matrix) of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.
    '''
    # do not remove the next 5 lines
    n_input, n_hidden, n_class, train_data, train_label, lambdaval = args
    # First reshape 'params' vector into 2 matrices of weights W1 and W2

    W1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    W2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # remove the next two lines and replace them with your code 

    bias = np.ones((len(train_data),1))
    train_data = np.concatenate((train_data, bias), 1) #adding bias to the input data

    net1 = np.dot(train_data, W1.T)
    output_1 = sigmoid(net1)

    bias_hidden = np.ones((len(output_1),1))
    output_1 = np.concatenate((output_1, bias_hidden), 1) #adding bias to the hidden output

    net2 = np.dot(output_1,W2.T)
    output_2 = sigmoid(net2)   #feed forward output

    #one-of-K encoding
    y_l = np.zeros((len(train_label), n_class), dtype=int)
    # print(train_label)
    y_l[np.arange(len(train_label)), train_label.astype(int)] = 1
    
    
    delta_l = output_2 - y_l
    gradient_output = np.dot(delta_l.T, output_1)

    gradient_hidden1 = (1-output_1)*output_1
    gradient_hidden2 = np.dot(delta_l,W2)
    gradient_hidden3 = gradient_hidden1 * gradient_hidden2
    gradient_hidden = np.dot(gradient_hidden3.T, train_data)
    gradient_hidden = gradient_hidden[0:n_hidden,:]

    N = len(train_data)
    obj_val1 = (-1/N)*(np.sum((y_l * np.log(output_2) + ((1-y_l) * np.log(1-output_2)))))

    obj_val = obj_val1 + (lambdaval/(2*N))* (np.sum(W1*W1)+np.sum(W2*W2))

    grad_w2 = (1/N)* (gradient_output + lambdaval*W2)
    grad_w1 = (1/N)* (gradient_hidden + lambdaval*W1)

    obj_grad = np.concatenate((grad_w1.flatten(),grad_w2.flatten()),0)

    return (obj_val,obj_grad)

def nnPredict(W1, W2, data):
    '''
    % nnPredict predicts the label of data given the parameter W1, W2 of Neural
    % Network.

    % Input:
    % W1: matrix of weights for hidden layer units
    % W2: matrix of weights for output layer units
    % data: matrix of data. Each row of this matrix represents the feature vector for the corresponding data instance

    % Output:
    % label: a column vector of predicted labels
    '''
    # remove the next line and replace it with your code
    labels = np.zeros((data.shape[0],1))


    bias = np.ones((len(data),1))
    data = np.concatenate((data, bias), 1) #adding bias to the input data

    net1 = np.dot(data, W1.T)
    output_1 = sigmoid(net1)

    bias_hidden = np.ones((len(output_1),1))
    output_1 = np.concatenate((output_1, bias_hidden), 1) #adding bias to the hidden output

    net2 = np.dot(output_1,W2.T)
    output_2 = sigmoid(net2)   #feed forward output

    for i in range(len(output_2)):
        pred_class = np.argmax(output_2[i])
        labels[i] = pred_class
    # print("labels is {}".format(labels.shape))

    return labels.reshape((labels.shape[0],))
