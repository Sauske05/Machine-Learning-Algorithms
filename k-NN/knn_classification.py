# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 16:05:37 2024

@author: LENOVO
"""

''' 
#Information on the Dataset
The dataset is related to cancer diagnosis, particularly breast cancer. 
The columns include various measurements and characteristics of cell nuclei 
present in breast cancer biopsies. Each row in the dataset likely represents a 
different biopsy sample, with the 'id' column serving as a unique identifier for 
each sample. Here's a brief explanation of some of the columns:
'diagnosis': This column probably contains information about whether the biopsy 
is diagnosed as malignant (cancerous) or benign (non-cancerous).
The remaining columns seem to contain numerical measurements of different 
features for each biopsy sample. These features are typically computed from images of 
cell nuclei, and they include mean values, standard errors, and worst (largest) values 
for various characteristics such as radius, texture, perimeter, area, smoothness, 
compactness, concavity, concave points, symmetry, and fractal dimension.
For example:
'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 
 'fractal_dimension_mean': These columns likely represent the mean values of 
 these features for each cell nucleus.
'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 
'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 
'fractal_dimension_se': These columns probably represent the standard errors of 
the corresponding features.
'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', '
smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 
'symmetry_worst', 'fractal_dimension_worst': These columns likely represent the 
worst (largest) values of these features.
Analysing this dataset could involve exploring the relationships between these 
features and the diagnosis to identify patterns that may help in the diagnosis of breast cancer. 
Machine learning models could be trained on this dataset to predict the diagnosis 
based on the provided features. Additionally, statistical analyses and visualizations
 may be performed to gain insights into the characteristics of malignant and benign tumors.


3

'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv('./KNNAlgorithmDataset.csv')
df.drop(columns = ['Unnamed: 32'],inplace = True)

df['diagnosis'].value_counts()

def plot(type = 'scatter'):
    count = 1
    plt.figure(figsize=(12,12))
    for i in df.columns:
        plt.subplot(8, 4, count)
        count +=1
        if type == 'hist':
            sns.histplot(data=df, x=i, kde = True, hue = 'diagnosis')
        else:
            sns.scatterplot(data = df, x = i, y = 'diagnosis', hue = 'diagnosis')
    plt.tight_layout()
    plt.show()

plot('hist')
#plot()


#M represents cancerous, B represents Non-cancerous
diagnosis_dict = {'M' : 1, 'B' : 0}
df['diagnosis'] = df['diagnosis'].map(diagnosis_dict)
print(df.diagnosis.head())

corr_matrix = df.corr()
print(corr_matrix)

selected_features = corr_matrix[(corr_matrix['diagnosis'] > 0.1) | (corr_matrix['diagnosis'] < -0.1)] 
selected_features = list(selected_features.index)
input_features = selected_features
output_features = 'diagnosis'
X = df[input_features]
y = df[output_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

def train_test_plot():
    accuracy_list = []
    k_list = []
    for i in range(1, 16):
        knn_model = KNeighborsClassifier(n_neighbors=i)
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_list.append(accuracy)
        k_list.append(i)
    sns.lineplot(x = k_list, y = accuracy_list)


train_test_plot()

#According to the plot, we can use K as 11 as the accuracy is highest.
model = KNeighborsClassifier(n_neighbors=11)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'The accuracy of the model is {accuracy}')
    
    

    