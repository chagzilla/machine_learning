# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 22:23:57 2018

@author: chagulwi
"""

import numpy as np
import matplotlib.pyplot as mat
from NeuralNetwork_skeleton import neuralNetwork


#testing a neural network
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

#learning rate is 0.3
learning_rate = 0.3

#create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
print("this is the current weight for input to hidden layer ", n.wih)
print("this is the current weight for hidden layer to output ", n.who)
#load the mnist training data CSV file into a list
training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#go through all records in the training data set
for record in training_data_list:
    all_values = record.split(',')
    inputs = np.asfarray(all_values[1:]) / 255.0 * .99 + .01
    targets = np.zeros(output_nodes) + .01
    targets[int(all_values[0])] = .99
    n.train(inputs, targets)
    

# load the mnist test data CSV file into a list
test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the neural network

# scorecard for how well the nerwork performs, initially empty
scorecard = []

# go through all te records in the test data set
for record in test_data_list:
    #split the record by the ',' commas
    all_values = record.split(',')
    # coorect answer is first value
    correct_label = int(all_values[0])
    print(correct_label, "correct label")
    #scale and shift the inputs
    inputs = np.asfarray(all_values[1:]) / 255.0 * .99 + 0.1
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value vorresponds to the label
    label = np.argmax(outputs)
    print(label, "networks's answer")
    # append corret or incorrec to list
    if(label == correct_label):
        # network's answer matces correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match corect answer, add 0 to scorecard
        scorecard.append(0)

# calculate the performance score, the fraction of crrect answers
scorecard_array = np.asarray(scorecard)
print("performance =", scorecard_array.sum() / 
      scorecard_array.size)

