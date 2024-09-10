# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 18:51:42 2024

@author: ARUN JOSHI
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
''' Error: UserWarning: KMeans is known to have a memory leak on Windows with MKL, 
when there are less chunks than available threads. You can avoid it by setting 
the environment variable OMP_NUM_THREADS=1.


To avoid the memory leak issue, we can limit the number of threads used by MKL 
by setting the environment variable OMP_NUM_THREADS to 1.'''
import os
os.environ["OMP_NUM_THREADS"] = "1"

df = pd.read_csv('Mall_Customers.csv')
df.drop(columns = ['CustomerID'], inplace = True)
df.rename(columns={'Annual Income (k$)': 'Annual Income', 
                   'Spending Score (1-100)':'Spending Score'}, inplace = True)
null_val_sum = df.isnull().sum()
print(null_val_sum )

gender_dict = {'Male' : 1, 'Female' : 0}
df['Gender'] = df['Gender'].map(gender_dict)


#Use df.desribe() to get the statistical understanding of the data
#df.describe()

def plot(type = 'hist'):
    count = 1
    plt.figure(figsize=(12,12))
    if type == 'scatter':
        for i in df.columns:
            if i != 'Gender':
                for j in df.columns:
                    if j!= 'Gender':
                        plt.subplot(3,3, count)
                        count +=1
                        sns.scatterplot(data = df, x = i, y = j, hue = 'Gender')
                        plt.xlabel(i)
                        plt.ylabel(j)
                        plt.title(f'Scatter Plot of {i} vs {j}')
    else:
        for i in df.columns:
            plt.subplot(2, 2, count)
            count+=1
            sns.histplot(data = df, x = i, kde = True, hue = 'Gender')
            plt.xlabel(i)
            plt.ylabel('Count')
            plt.title(f'Histogram of {i}')
    
    plt.tight_layout()
    plt.show()
    
plot()
plot(type = 'scatter')

def elbow_plot(type = '2d'):
    if type == '2d':
        input_features = ['Annual Income', 'Spending Score']
        X = df[input_features]
    if type == '3d': 
        input_features = ['Age', 'Annual Income', 'Spending Score']
        X = df[input_features]
    ''' This function plots the elbow method to show the most optimal
    k value for the k-means algorithm'''
    model = KMeans(random_state = 42)
    visualizer = KElbowVisualizer(model, k = (2,10))
    
    visualizer.fit(X)
    visualizer.show()
    
elbow_plot()
'''The elbow plot at k = 5, seems to be the best hyperparameter for K as the 
distortion score seems to be the lowest.'''

def model(type = '2d'):
    if type == '2d':
        input_features = ['Annual Income', 'Spending Score']
        X = df[input_features]
        model = KMeans(n_clusters=5)
    if type == '3d':
        input_features = ['Age' ,'Annual Income', 'Spending Score']
        X = df[input_features]
        model = KMeans(n_clusters=6)
    model.fit(X)
    labels = model.labels_
    centroids = model.cluster_centers_
    return model, labels, centroids

model1, labels1, centroids1 = model()
print(labels1)
print(centroids1)

def plotting_k_means():
    #Plotting the data
    plt.scatter(df['Annual Income'], df['Spending Score'], 
                c = labels1, cmap = 'viridis', marker = 'o')
    #Plotting the centroids
    plt.scatter(centroids1[:, 0], centroids1[:, 1], color = 'red', 
                marker = 'x', label = 'Centroids')
    plt.title('K Means Clustering')
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.show()


plotting_k_means()


def plotting_decision_boundary():
    x_min, x_max = df['Annual Income'].min()-1, df['Annual Income'].max() +1
    y_min, y_max = df['Spending Score'].min()-1, df['Spending Score'].max() + 1
    h = 2
    x_sample = np.arange(x_min, x_max, h)
    y_sample = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(x_sample, y_sample)
    print(f'X min: {x_min}')
    print(f'X max: {x_max}')
    print(f'y min: {y_min}')
    print(f'y max: {y_max}')
    print(f'Shape of x sample: {x_sample.shape}')
    print(f'Shape of y sample: {y_sample.shape}')
    print(f'Shape of xx : {xx.shape}')
    print(f'Shape of yy : {yy.shape}')
    
    z = model1.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    
    plt.contourf(xx,yy, z, alpha = 0.8, cmap = plt.cm.Spectral)
    plt.scatter(df['Annual Income'], df['Spending Score'], 
                c = labels1, cmap = plt.cm.Spectral, marker = 'o',
                edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.title('Decision Boundary of K-Means Model')
    plt.show()

plotting_decision_boundary()

''' Below is the further modeling of the data by using 3 input features and 
plotting a 3d model plot.'''

def three_dim_plot():
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(projection = '3d')
    X = df['Age']
    Y = df['Annual Income']
    Z = df['Spending Score']
    ax.scatter(X,Y,Z, marker = 'o')
    ax.set_xlabel('Age')
    ax.set_ylabel('Annual Income')
    ax.set_zlabel('Spending Score')
    plt.show()
    
three_dim_plot()
elbow_plot(type = '3d')
''' According to elbow plot, when k =6 the distortion score is the lowest. Thus, 
making k = 6 the best hypermater setting for the model.'''

model2, labels2, centroids2 = model(type = '3d')
print(labels2)
print(centroids2)

def model_plot_3d():
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(projection = '3d')
    X = df['Age']
    Y = df['Annual Income']
    Z = df['Spending Score']
    #Plotting the data
    ax.scatter(X, Y, Z, 
                c = labels2, cmap = 'viridis', marker = 'o')
    #Plotting the centroids
    plt.scatter(centroids2[:, 0], centroids2[:, 1],centroids2[:, 2], 
                color = 'red', marker = 'x', label = 'Centroids')
    # Creating a meshgrid to plot decision boundaries
    x_min, x_max = X.min() - 1, X.max() + 1
    y_min, y_max = Y.min() - 1, Y.max() + 1
    z_min, z_max = Z.min() - 1, Z.max() + 1

    xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 50),
                             np.linspace(y_min, y_max, 50),
                             np.linspace(z_min, z_max, 50))
    
    # Stack the grid to pass it to the model
    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    
    # Predicting the cluster for each point in the grid
    labels_grid = model2.predict(grid)
    
    # Reshaping the result to match the shape of the grid
    labels_grid = labels_grid.reshape(xx.shape)
    
    # Plotting decision boundary surfaces
    ax.plot_surface(xx[:, :, 0], yy[:, :, 0], zz[:, :, 0], 
                    facecolors=plt.cm.viridis(labels_grid[:, :, 0] / labels_grid.max()), 
                    alpha=0.3, rstride=100, cstride=100)
    ax.set_xlabel('Age')
    ax.set_xlabel('Annual Income')
    ax.set_xlabel('Spending Score')
    plt.title('K Means Clustering')
    plt.show()
    
    
model_plot_3d()

from mpl_toolkits import mplot3d
def test_plot():
    ax = plt.axes(projection = '3d')
    ax.scatter(3,5,7)
    plt.show()
    
test_plot()
    

    

