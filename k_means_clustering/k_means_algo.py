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

def elbow_plot():
    input_features = ['Annual Income', 'Spending Score']
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

def model():
    input_features = ['Annual Income', 'Spending Score']
    X = df[input_features]
    model = KMeans(n_clusters=5)
    model.fit(X)
    labels = model.labels_
    centroids = model.cluster_centers_
    return model, labels, centroids

model1, labels1, centroids1 = model()
print(labels1)
print(centroids1)

def plotting_knn():
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


plotting_knn()


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


    

