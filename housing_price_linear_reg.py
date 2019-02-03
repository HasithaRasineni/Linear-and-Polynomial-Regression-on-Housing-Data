# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 23:30:16 2018

@author: Hasitha Rasineni
"""
import numpy as np
import matplotlib.pyplot as plt

price = np.genfromtxt('housing.csv', delimiter = ',', skip_header = 1, usecols= 3)
#Normalize 
price = price/ max(price)

area = np.genfromtxt('housing.csv',delimiter=',', skip_header=1, usecols=0)
#Normalize 
area = area/ max(area)

#Learning rate
alpha = 0.01

#Linear regression
n = len(price) # no of data inputs
theta = np.zeros((2))
iterations = 1000
ones_vec = np.ones((n))
new_area = np.stack((ones_vec, area), axis=-1)

for i in range(iterations):
    
    hypothesis=np.matmul(new_area, theta)
    cost = (1/(2*n))* np.sum(np.square(hypothesis - area))
    theta = theta - (1/n)*(alpha)* (np.matmul((hypothesis - price).T ,new_area)).T
    print(theta)

#Polynomial Regression
x1 = area
x2 = area**2
x3 = area**3
x = np.stack((x1, x2, x3), axis=-1)
theta1 = np.zeros((x.shape[1]+1))
new_x = np.stack((ones_vec, x1, x2, x3), axis=-1)

for i in range(iterations):
    
    hypothesis = np.matmul(new_x, theta1)
    cost = (1/(2*n))* np.sum(np.square(hypothesis - price))
    theta1 = theta1 - (1/n)*(alpha)* (np.matmul((hypothesis - price).T ,new_x)).T


price_pred_poly = theta1[0] + theta1[1]*x1 + theta1[2]*x2 + theta1[3]*x3
price_pred = theta[0] + theta[1]*area
plt.figure()
plt.scatter(area, price, s=5, c = 'g')
plt.plot(area, price_pred, c = 'r')
plt.plot(area, price_pred_poly, c = 'b')


