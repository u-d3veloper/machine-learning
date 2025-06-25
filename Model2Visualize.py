import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def visualize(data):
    # Visualize the distribution of the target variable
    plt.figure(figsize=(10, 5))
    sns.countplot(x='quality', data=data, palette='Set2')
    plt.title('Distribution of Quality')
    plt.xlabel('Quality')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.savefig('visualizations/quality_distribution.png')

    # Visualize the distribution of the target variable with KDE
    pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (40,40), diagonal = 'kde')
    plt.savefig('visualizations/scatter_matrix.png')

    # Visualize the correlation matrix
    correlation = data.corr()
    # display(correlation)
    plt.figure(figsize=(14, 12))
    heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
    plt.savefig('visualizations/correlation_matrix.png')