import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression


class LogisticRegressionScratch:
    def __init__(self, lr=0.01, num_iter=10000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        # weights initialization
        self.theta = np.zeros(X.shape[1])

    @staticmethod
    def __add_intercept(x):
        intercept = np.ones((x.shape[0], 1))
        return np.concatenate((intercept, x), axis=1)

    @staticmethod
    def __sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def __loss(h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)

            if self.verbose == True and i % 10000 == 0:
                print(f'loss: {loss} \t')

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        return self.predict_prob(X).round()


iris = datasets.load_iris()


def plotDataset():
    # The indices of the features that we are plotting
    x_index = 0
    y_index = 1

    # this formatter will label the colorbar with the correct target names
    formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

    plt.figure(figsize=(10, 8))
    plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
    plt.colorbar(ticks=[0, 1, 2], format=formatter)
    plt.xlabel(iris.feature_names[x_index])
    plt.ylabel(iris.feature_names[y_index])

    plt.tight_layout()
    plt.show()


# test data with :2 (only sepal length/width) & :4 (all 4 attributes - sepal length/width, petal length/width)
X = iris.data[:, :2]
y = (iris.target != 0) * 1


# figure with 2 types of iris: setosa & virginica
def plotRestrictedDataset():
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0 - setosa')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1 - virginica')
    plt.legend()
    plt.show()


def sklearnLogisticRegression(iterations):
    model = LogisticRegression(C=iterations, solver='lbfgs')
    # solver used to avoid FutureWarning on default solver
    model.fit(X, y)
    preds = model.predict(X)
    (preds == y).mean()
    print("sklearn accuracy: ", model.score(X, y))


def scratchLogisticRegression(iterations):
    model = LogisticRegressionScratch(lr=0.1, num_iter=iterations)
    model.fit(X, y)
    preds = model.predict(X)
    print("scratch accuracy: ", (preds == y).mean())


def plotScratchLogisticRegression(iterations):
    model = LogisticRegressionScratch(lr=0.1, num_iter=iterations)
    model.fit(X, y)

    preds = model.predict(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0 - setosa')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1 - virginica')
    plt.legend()
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max(),
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    probs = model.predict_prob(grid).reshape(xx1.shape)
    plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.show()


# tests: 1.000, 50.000, 200.000, 300.000
numberOfIterations = 50000
print("number of iterations: ", numberOfIterations)


def scratch(numberOfIterations):
    scratchMethodStart = datetime.datetime.now()
    scratchLogisticRegression(numberOfIterations)
    scratchMethodEnd = datetime.datetime.now()
    print("scratch duration: ", scratchMethodEnd - scratchMethodStart)


def scikit(numberOfIterations):
    skMethodStart = datetime.datetime.now()
    sklearnLogisticRegression(numberOfIterations)
    skMethodEnd = datetime.datetime.now()
    print("sklearn duration: ", skMethodEnd - skMethodStart)


def display_title_bar():
    # Clears the terminal screen, and displays a title bar.
    os.system('clear')

    print("\t**********************************************")
    print("\t***********  Logistic Regression  ************")
    print("\t**********************************************")


choice = ''
while choice != 'quit':
    display_title_bar()

    # Let users know what they can do.
    print("\n[1] See logistic regression from scratch.")
    print("[2] See logistic regression from scikit.")
    print("[3] Compare logistic regression implementations.")
    print("[4] See logistic regression from scratch plot.")
    print("[5] See Iris dataset plot.")
    print("[6] See binary Iris dataset plot.")
    print("[quit] Quit.")

    choice = input("What would you like to do? ")

    # Respond to the user's choice.
    if choice == '1':
        numberOfIterations = int(float(input("How many iterations should we try? \n")))
        scratch(numberOfIterations)
        input("Press Enter to continue..")
    elif choice == '2':
        numberOfIterations = int(float(input("How many iterations should we try? \n")))
        scikit(numberOfIterations)
        input("Press Enter to continue..")
    elif choice == '3':
        numberOfIterations = int(float(input("How many iterations should we try? \n")))
        scratch(numberOfIterations)
        scikit(numberOfIterations)
        input("Press Enter to continue..")
    elif choice == '4':
        numberOfIterations = int(float(input("How many iterations should we try? \n")))
        plotScratchLogisticRegression(numberOfIterations)
        input("Press Enter to continue..")
    elif choice == '5':
        plotDataset()
        input("Press Enter to continue..")
    elif choice == '6':
        plotRestrictedDataset()
        input("Press Enter to continue..")
    elif choice == 'quit':
        print("\nThanks for coming. Bye.")
    else:
        print("\nI didn't understand that choice.\n")
        input("Press Enter to continue..")
