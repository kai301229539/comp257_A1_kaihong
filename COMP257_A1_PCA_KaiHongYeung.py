# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 22:11:53 2022

@author: phili
"""

#Comp257
#Assignment 1: Dimensionality Reduction using PCA


#Retrieve and load the mnist_784 dataset of 70,000 instances. [5 points]



import numpy as np
import pandas as pd


# =============================================================================
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# =============================================================================

import matplotlib as mpl
import matplotlib.pyplot as plt

#Load the data
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
#print(mnist.keys())


#print(mnist["feature_names"])


X, y = mnist['data'], mnist['target']



print(X.shape)
print(y.shape)

X.head(1)

y.info()


X = pd.DataFrame(X).to_numpy()
y = pd.DataFrame(y).to_numpy()

#Display each digit. [5 points]

#plot each digit
def plot_digit(data, target, index):
    some_digit = data[index]
    print(target[index])
    print(some_digit)
    print(type(some_digit))
    print(type(y))
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=mpl.cm.binary)
    plt.axis("off")
    plt.show()


plot_digit(X, y, 1)#0
plot_digit(X, y, 3)#1
plot_digit(X, y, 5)#2
plot_digit(X, y, 7)#3
plot_digit(X, y, 9)#4
plot_digit(X, y, 0)#5
plot_digit(X, y, 11)#6
plot_digit(X, y, 15)#7
plot_digit(X, y, 17)#8
plot_digit(X, y, 4)#9

#Display each digit
print(y)


#Use PCA to retrieve the 1th and 2nd principal component and output their explained variance ratio. [5 points]


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
scaled_data=scaler.transform(X)

print(scaled_data)

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(scaled_data)
X_pca = pca.transform(scaled_data)

scaled_data.shape

X_pca.shape

scaled_data

X_pca

print(pca.explained_variance_ratio_)

print(pca.explained_variance_ratio_[0])
print(pca.explained_variance_ratio_[1])

# =============================================================================
# [0.05642719 0.04041226]
# =============================================================================

# =============================================================================
# The output indicates that 5.6% of the dataset’s variance lies along the first PC, 
# and 4.0% lies along the second PC leaving more than 90% for the others PC
# =============================================================================



# =============================================================================
# pca = PCA()
# pca.fit(scaled_data)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# d = np.argmax(cumsum >= 0.95)+1
# print("minimum number of dimensions required to preserve 95%:"+str(d))
# =============================================================================

# =============================================================================
# pca = PCA(n_components=0.95)
# X_pca = pca.fit_transform(scaled_data)
# print(pca.explained_variance_ratio_)
# =============================================================================

#Plot the projections of the 1th and 2nd principal component onto a 2D hyperplane. [5 points]

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0],X_pca[:,1], c=y)
plt.xlabel('First principle component')
plt.ylabel('Second principle component')



#Use Incremental PCA to reduce the dimensionality of the MNIST dataset down to 154 dimensions. [10 points]

from sklearn.decomposition import  IncrementalPCA

n_batches = 100
ipca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(scaled_data, n_batches):
    ipca.partial_fit(X_batch)
    
X_ipca = ipca.transform(scaled_data)


X_ipca.shape

# =============================================================================
# ipca = IncrementalPCA(n_components=154, batch_size=154)
# X_ipca = ipca.fit_transform(scaled_data)
# =============================================================================


#Display the original and compressed digits from (5). [5 points]


ipca.components_.shape



import seaborn as sns

fig, axarr = plt.subplots(10, 2, figsize=(12, 40))
sns.heatmap(X[1].reshape(28, 28), ax=axarr[0][0], cmap='gray_r')
sns.heatmap(ipca.components_[0, :].reshape(28, 28), ax=axarr[0][1], cmap='gray_r')

sns.heatmap(X[3].reshape(28, 28), ax=axarr[1][0], cmap='gray_r')
sns.heatmap(ipca.components_[1, :].reshape(28, 28), ax=axarr[1][1], cmap='gray_r')

sns.heatmap(X[5].reshape(28, 28), ax=axarr[2][0], cmap='gray_r')
sns.heatmap(ipca.components_[2, :].reshape(28, 28), ax=axarr[2][1], cmap='gray_r')

sns.heatmap(X[7].reshape(28, 28), ax=axarr[3][0], cmap='gray_r')
sns.heatmap(ipca.components_[3, :].reshape(28, 28), ax=axarr[3][1], cmap='gray_r')

sns.heatmap(X[9].reshape(28, 28), ax=axarr[4][0], cmap='gray_r')
sns.heatmap(ipca.components_[4, :].reshape(28, 28), ax=axarr[4][1], cmap='gray_r')

sns.heatmap(X[0].reshape(28, 28), ax=axarr[5][0], cmap='gray_r')
sns.heatmap(ipca.components_[5, :].reshape(28, 28), ax=axarr[5][1], cmap='gray_r')

sns.heatmap(X[11].reshape(28, 28), ax=axarr[6][0], cmap='gray_r')
sns.heatmap(ipca.components_[6, :].reshape(28, 28), ax=axarr[6][1], cmap='gray_r')

sns.heatmap(X[15].reshape(28, 28), ax=axarr[7][0], cmap='gray_r')
sns.heatmap(ipca.components_[7, :].reshape(28, 28), ax=axarr[7][1], cmap='gray_r')

sns.heatmap(X[17].reshape(28, 28), ax=axarr[8][0], cmap='gray_r')
sns.heatmap(ipca.components_[8, :].reshape(28, 28), ax=axarr[8][1], cmap='gray_r')

sns.heatmap(X[4].reshape(28, 28), ax=axarr[9][0], cmap='gray_r')
sns.heatmap(ipca.components_[9, :].reshape(28, 28), ax=axarr[9][1], cmap='gray_r')

# =============================================================================
# fig, axarr = plt.subplots(2, 2, figsize=(12, 8))
# sns.heatmap(X[0].reshape(28, 28), ax=axarr[0][0], cmap='gray_r')
# sns.heatmap(X[1].reshape(28, 28), ax=axarr[0][1], cmap='gray_r')
# sns.heatmap(X[2].reshape(28, 28), ax=axarr[1][0], cmap='gray_r')
# sns.heatmap(X[3].reshape(28, 28), ax=axarr[1][1], cmap='gray_r')
# 
# fig, axarr = plt.subplots(2, 2, figsize=(12, 8))
# sns.heatmap(pca.components_[8, :].reshape(28, 28), ax=axarr[0][0], cmap='gray_r')
# sns.heatmap(pca.components_[9, :].reshape(28, 28), ax=axarr[0][1], cmap='gray_r')
# sns.heatmap(pca.components_[0, :].reshape(28, 28), ax=axarr[1][0], cmap='gray_r')
# sns.heatmap(pca.components_[5, :].reshape(28, 28), ax=axarr[1][1], cmap='gray_r')
# 
# 
# fig, axarr = plt.subplots(2, 2, figsize=(12, 8))
# sns.heatmap(ipca.components_[4, :].reshape(28, 28), ax=axarr[0][0], cmap='gray_r')
# sns.heatmap(ipca.components_[5, :].reshape(28, 28), ax=axarr[0][1], cmap='gray_r')
# sns.heatmap(ipca.components_[6, :].reshape(28, 28), ax=axarr[1][0], cmap='gray_r')
# sns.heatmap(ipca.components_[7, :].reshape(28, 28), ax=axarr[1][1], cmap='gray_r')
# =============================================================================





#Show the Demonstration in Lab or Create a video discussing the code and result for each question.
#Write an analysis report discussing the challenges you confronted and solutions to overcome them, if applicable [15 points]



# =============================================================================
# Question 2
# [50 points]

# Generate Swiss roll dataset. [5 points]
import matplotlib.pyplot as plt
from sklearn import manifold, datasets


# =============================================================================
# Xndarray of shape (n_samples, 3)
# The points.
# 
# tndarray of shape (n_samples,)
# The univariate position of the sample according to the main dimension of the points in the manifold.
# =============================================================================


X, t = datasets.make_swiss_roll(n_samples=1500, random_state=39)
y = t > 7

X = pd.DataFrame(X).to_numpy()
t = pd.DataFrame(t).to_numpy()

# Plot the resulting generated Swiss roll dataset. [2 points]
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
fig.add_axes(ax)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
fig.show()    

# Use Kernel PCA (kPCA) with linear kernel (2 points), a RBF kernel (2 points), and a sigmoid kernel (2 points). [6 points]

from sklearn.decomposition import KernelPCA

linear_pca = KernelPCA(n_components=2, kernel="linear")
X_linear_reduced = linear_pca.fit_transform(X)

rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
X_rbf_reduced = rbf_pca.fit_transform(X)

sig_pca = KernelPCA(n_components=2, kernel="sigmoid", gamma=0.001, coef0=1)
X_sig_reduced = sig_pca.fit_transform(X)


# Plot the kPCA results of applying the linear kernel (2 points), a RBF kernel (2 points), and a sigmoid kernel (2 points) from (3). Explain and compare the results in your analysis report [6 points]
plt.scatter(X_linear_reduced[:, 0], X_linear_reduced[:, 1], c=t, cmap=plt.cm.hot)
plt.show()

plt.scatter(X_rbf_reduced[:, 0], X_rbf_reduced[:, 1], c=t, cmap=plt.cm.hot)
plt.show()

plt.scatter(X_sig_reduced[:, 0], X_sig_reduced[:, 1], c=t, cmap=plt.cm.hot)
plt.show()


# Using kPCA and a kernel of your choice, apply Logistic Regression for classification. Use GridSearchCV to find the best kernel and gamma value for kPCA in order to get the best classification accuracy at the end of the pipeline. Print out best parameters found by GridSearchCV. [14 points]
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.datasets import make_swiss_roll 


clf = Pipeline([
     ('kpca', KernelPCA(n_components=2)),
     ('log_reg', LogisticRegression())
])


param_gird = [{
     'kpca__gamma': np.linspace(0.03, 0.05, 10),
     'kpca__kernel': ['rbf', 'sigmoid']
}]

grid_search = GridSearchCV(clf, param_gird, cv=3)
grid_search.fit(X, y)

# =============================================================================
# GridSearchCV(cv=3,
#              estimator=Pipeline(steps=[('kpca', KernelPCA(n_components=2)),
#                                        ('log_reg', LogisticRegression())]),
#              param_grid=[{'kpca__gamma': array([0.03      , 0.03222222, 0.03444444, 0.03666667, 0.03888889,
#        0.04111111, 0.04333333, 0.04555556, 0.04777778, 0.05      ]),
#                           'kpca__kernel': ['rbf', 'sigmoid']}])
# =============================================================================


# Plot the results from using GridSearchCV in (5). [2 points]

print(grid_search.best_params_)
# =============================================================================
# {'kpca__gamma': 0.05, 'kpca__kernel': 'rbf'}
# =============================================================================


gs_rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.05)
X_gs_rbf_reduced = gs_rbf_pca.fit_transform(X)

plt.scatter(X_gs_rbf_reduced[:, 0], X_gs_rbf_reduced[:, 1], c=t, cmap=plt.cm.hot)
plt.show()


# Show the Demonstration in Lab or Create a video discussing the code and result for each question.
# In the analysis report discuss challenges you confronted and solutions to overcoming them, if applicable [15 points]
# =============================================================================
