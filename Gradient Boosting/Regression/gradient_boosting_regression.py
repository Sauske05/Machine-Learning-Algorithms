# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:37:05 2024

@author: Arun Joshi
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import shap
df = pd.read_csv('./affairs.csv')
df.drop(columns = ['Unnamed: 0'], inplace = True)
print(df.info())
statistical_data = df.describe()
def plot():
    count = 1
    plt.figure(figsize=(24,24))
    for i in df.columns:
        plt.subplot(3, 3, count)
        count +=1
        sns.histplot(data = df, x = i, kde=True)
        plt.xlabel(i)
        plt.ylabel('Count')
        plt.title(f'Historgram of {i}')
    plt.tight_layout()
    plt.show()

plot()

def scatter_plot():
    count = 1
    plt.figure(figsize=(24,24))
    for i in df.columns:
        if i!= 'affairs':
            plt.subplot(4,2,count)
            count +=1
            sns.scatterplot(x = df[i], y = df['affairs'])
            plt.xlabel(i)
            plt.ylabel('Affairs')
            plt.title(f'Scatter plot of {i} vs Affairs')
    plt.tight_layout()
    plt.show()

scatter_plot()

#Correlation Matrix 
corr_matrix = df.corr()
corr_matrix = pd.DataFrame(corr_matrix)
affair_correlation = corr_matrix['affairs']

def initial_model():
    input_features = list(affair_correlation.index)[:-1]
    output_features = 'affairs'
    X_train, X_test, y_train, y_test = train_test_split(
        df[input_features], df[output_features], test_size=0.2, 
        random_state = 42)
    regr = GradientBoostingRegressor()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'The mean absolute error of the model is {mae}')
    percentage_error = ((mae)/(y_test.max() - y_test.min())) * 100
    print(f'Percentage error of the model is {percentage_error} %')
    
    #return X_train, X_test, y_train, y_test, regr
    
initial_model()

'''
Training the model via feature selection based on correlation matrix
'''
def selected_feature_model():
    input_features = affair_correlation[(affair_correlation.values > 0.1) |
                                        (affair_correlation.values < -0.1)].index
    input_features = list(input_features)[:-1]
    output_features = 'affairs'
    X_train, X_test, y_train, y_test = train_test_split(
        df[input_features], df[output_features], test_size=0.2, 
        random_state = 42)
    regr = GradientBoostingRegressor()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'The mean absolute error of the second model is {mae}')
    percentage_error = ((mae)/(y_test.max() - y_test.min())) * 100
    print(f'Percentage error of the second model is {percentage_error} %')
    return X_train, X_test, y_train, y_test, regr, y_pred
X_train, X_test, y_train, y_test, model, y_pred = selected_feature_model()

'''
def line_plot():
    plt.figure(figsize=(12,12))
    ax = plt.figure().add_subplot(projection = '3d')
    X = X_test['rate_marriage']
    Y = X_test['religious']
    Z = y_pred
    z1 = y_test
    
    #ax.plot(X,Y,Z, label = 'Model Function')
    ax.scatter(X,Y, z1)
    ax.legend()
    ax.set_xlabel('Rate Marriage')
    ax.set_ylabel('Religious')
    ax.set_zlabel('Prediction')
    plt.show()
    
line_plot()
'''   

'''
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Plot the global importance of features
shap.summary_plot(shap_values, X_test)

# Plot dependence of one feature (e.g., feature 0)
shap.dependence_plot(0, shap_values.values, X_test)
print(shap_values)'''

    
    
        