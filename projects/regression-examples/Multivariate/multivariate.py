# Multivariate Linear Regression in Python WITHOUT Scikit-Learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------#
#                  Functional defintions                     #
# -----------------------------------------------------------#


def load_data(path, sep, names):

    # Load the dataset from the given path and separator.

    data = pd.read_csv(path, sep=sep, names=names)
    return data


def describe_dataset(data, n=5):
    # Display the first n rows of the dataset.
    print(data.head(n))
    # Display the shape of the dataset.
    print("Shape of the dataset: ", data.shape)
    # Display the columns of the dataset.
    classes = data.columns.to_list()
    for col in classes:
        print(f"Column: {col}, Type: {data[col].dtype}")
    print("Broad summary :\n {}".format( data.describe()))

def normalize_data(data):
    # Normalize the dataset.
    data = (data - data.min()) / (data.max() - data.min())
    return data

def compute_cost(X, y, theta):
    # Compute the cost function for linear regression.
    m = len(y)
    J = (1/(2*m)) * np.sum(np.square(X.dot(theta.T) - y))
    return J

def gradient_descent(X, y, theta, alpha, iters):
    # Perform gradient descent to learn theta.
    m = y.shape[0]
    J_history = np.zeros(iters)

    for i in range(iters):
        theta = theta - (alpha/m) * (X.T.dot(X.dot(theta.T) - y))
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history

# -----------------------------------------------------------#
#                        Main program                        #
# -----------------------------------------------------------#

path = "datasets/home.txt"
sep = ","
names = ['size', 'bedrooms', 'price']

data = load_data(path, sep, names)

describe_dataset(data, 10)

data = normalize_data(data)
print("Normalized data: \n", data.head(10))


#setting the matrixes
X = data.iloc[:,0:2].values
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)

y = data.iloc[:,2:3].values #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray
theta = np.zeros([1,3])

#set hyper parameters
alpha = 0.01
iters = 500

g,cost = gradient_descent(X,y,theta,alpha,iters)
print("Theta: \n", g)

# Plot the cost function
plt.plot(range(1, iters + 1), cost, color='blue')
plt.title('Cost function')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.savefig('visualizations/cost_function.png')
