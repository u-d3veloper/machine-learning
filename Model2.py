import numpy as np
import pandas as pd
from time import time
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import visuals as vs
from sklearn.model_selection import train_test_split
import Model2Visualize
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.svm import SVC


# Import the data
try: 
    data = pd.read_csv("datasets/winequality-red.csv",sep=';')
except Exception as e:
    print("Error: ", e)
    print("Please ensure the dataset is in the correct path.")

# Display the first few rows of the dataset
display(data.head(n=5))
# Display the shape of the dataset
print("Shape of the dataset: ", data.shape)

# Display the columns of the dataset
classnames = data.columns.to_list()
print("Columns of the dataset: ", classnames)

# Ensure the dataset has no missing values
print("Missing values in the dataset: ")
print("Number of missing values: ", data.isnull().sum().sum())
print(data.isnull().any())

# Display the data types of the columns
print("Data types of the columns: ", data.dtypes)
print(data.info())



# -----------------------------------------------------------#
#                    Visualize the data                      #
# -----------------------------------------------------------#


print("Summary statistics of the dataset: ", np.round(data.describe()))
df_size = data.shape[0]

above_6 = data.loc[(data['quality'] > 6)]
above_6_size = above_6.shape[0]

below_5 = data.loc[(data['quality'] < 5)]
below_5_size = below_5.shape[0]

between_5_and_6 = data.loc[(data['quality'] >= 5) & (data['quality'] <= 6)]
between_5_and_6_size = between_5_and_6.shape[0]

print("Total number of wine data :{} ".format(df_size))
print("Total number of wine data with quality above including 7 :{} ".format(above_6_size))
print("Total number of wine data with quality below including 4 :{} ".format(below_5_size))
print("Total number of wine data with quality between including 5 and 6 :{} ".format(between_5_and_6_size))



# Outlier detection


# Tukey's method
# For each feature find the data points with extreme high or low values

def outlier_tukey(data):
    for feature in data.keys():
        # TODO: Calculate Q1 (25th percentile of the data) for the given feature
        Q1 = np.percentile(data[feature], q=25)

        # TODO: Calculate Q3 (75th percentile of the data) for the given feature
        Q3 = np.percentile(data[feature], q=75)

        # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
        interquartile_range = Q3 - Q1
        step = 1.5 * interquartile_range

        # # Display the outliers
        # print("Data points considered outliers for the feature '{}':".format(feature))
        # display(data[~((data[feature] >= Q1 - step) & (data[feature] <= Q3 + step))])

        # OPTIONAL: Select the indices for data points you wish to remove
        outliers = []
        # Remove the outliers, if any were specified
        return data.drop(data.index[outliers]).reset_index(drop = True)

clean_data = outlier_tukey(data)

X = clean_data.drop('quality', axis=1).to_numpy()
y = clean_data['quality'].to_numpy()

y = np.where(y > 6, 1, 0)  # Convert the target variable to binary

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

model = SGDClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Confusion matrix: \n", confusion_matrix(y_test, y_pred))
print("Classification report: \n", classification_report(y_test, y_pred))