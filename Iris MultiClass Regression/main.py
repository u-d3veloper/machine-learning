# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt

from imports import *


# Load dataset
def load_dataset():
    # Load dataset
    url = "iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    return read_csv(url, names=names)

def resume_dataset(dataset):
    # shape
    print(dataset.shape)
    # head
    print(dataset.head(20))
    # descriptions
    print(dataset.describe())
    # class distribution
    print(dataset.groupby('class').size())

def univariate_plots(dataset):
    # box and whisker plots
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()
    # histograms
    dataset.hist()
    plt.show()

def multivariate_plots(dataset):
    # scatter plot matrix
    scatter_matrix(dataset)
    plt.show()

def split_dataset(dataset):
    # Split-out validation dataset
    array = dataset.values
    X = array[:,0:4] # 4 features
    y = array[:,4] # 1 label
    # Batch is 80% of the dataset
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
    return X_train, X_validation, Y_train, Y_validation

def build_models(X_train, Y_train):
    # Spot Check Algorithms
    models = []
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    # Compare Algorithms
    plt.boxplot(results, tick_labels=names)
    plt.title('Algorithm Comparison')
    plt.show()

def make_predictions(X_train, Y_train, X_validation, Y_validation):
    # Make predictions on validation dataset
    model = SVC(gamma='auto')
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    # Evaluate predictions
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

def main():
    dataset = load_dataset();
    # resume_dataset(dataset);
    # multivariate_plots(dataset);
    X_train, X_validation, Y_train, Y_validation = split_dataset(dataset);
    # build_models(X_train, Y_train);
    make_predictions(X_train, Y_train, X_validation, Y_validation);
    
if __name__ == "__main__":
    main()