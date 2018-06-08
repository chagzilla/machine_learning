# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 19:02:39 2018

@author: chagulwi
"""




class LinearRegression():
    
    def __init__(self, x_values, y_values):
        self.x = x_values
        self.y = y_values
        self.items = len(x_values)
        self.error = 100000000000
        self.m = 0
        self.c = 0
    
    def trainAlgorithm(self, iters , a, b):
        #this is the slope of the equation
        self.m = a
        # this is the c constant of the equation
        self.c = b
        # this is sthe index of the training algorithm
        i = 0
        while i < iters and not self.converge():
            step = 2 / (i  + 2)
            a_grad = 0
            b_grad = 0
            
            for x in range(5):
                a_grad += self.x[x] * ((self.m * self.x[x] + self.c) - self.y[x])
            
            a_grad = (2 * a_grad)/ 5
            
            for x in range(5):
                b_grad += (self.m * self.x[x] + self.c) - self.y[x]
            b_grad = (2 * b_grad) / 5
            
            #take a step
            self.m = self.m - (step * a_grad)
            self.c = self.c - (step * b_grad)
            
            print("a:\t" + str(self.m) + ", b:\t" + str(self.c) + "\r\n")
            print("grad_a:\t" + str(a_grad) + ", b_grad:\t" + str(b_grad) + "\r\n")
            i += 1
    def converge(self):
        error = 0
        thresh = .001
        for x in range(self.items):
            error += ((self.m * self.x[x] + self.c) - self.y[x]) * ((self.m * self.x[x]) - self.y[x])
        error /= self.items
        b = abs(error) > self.error - thresh and abs(error) < self.error + thresh
        self.error = error
        return b
    
    def regress(self, x):
        return self.m * x + self.c
            
        
        

lin = LinearRegression([1,2,3,4,5], [ 2.8,2.9,7.6,9,8.6 ])
lin.trainAlgorithm(1000, 3, -10)
a = input("input a number##")
while a != 'q':
    print(lin.regress(int(a)))
    a = input()
      