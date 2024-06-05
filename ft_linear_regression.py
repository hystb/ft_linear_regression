import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

def initialisation(X):
    W = 0
    b = 0
    return W, b

def model(X, W, b):
    Z = X * W + b
    return Z

def logLoss(Z, y):
    epsilon = 1e-15
    loss = 1 / len(y) * np.sum(y * np.log(Z + epsilon) + (1 - y) * np.log(1 - Z + epsilon))
    return loss

def update(dW, db, W, b, learning_rate):
    W -= learning_rate * dW
    b -= learning_rate * db
    return W, b

def predict(X, W, b):
    Z = model(X, W, b)
    return Z

def gradients(Z, X, y):
    dW = 1 / len(y) * np.sum((Z - y) * X)
    db = 1 / len(y) * np.sum(Z - y)
    return dW, db

def runModel(filename, scale=0.01, iter=50000):
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    X = data[:, 0]
    y = data[:, 1]
    y = y.reshape(-1, 1)
    X = X.reshape(-1, 1)
    
    yMax = y.max()
    XMax = X.max()
    y = y / yMax
    X = X / XMax
    W, b = initialisation(X)
    Train_loss = []

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(X * XMax, y * yMax, color='blue', label='Data points')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title('Initial Data Scatter Plot')
    plt.legend()

    for i in tqdm(range(iter)):
        Z = model(X, W, b)
        Train_loss.append(logLoss(Z, y))
        dW, db = gradients(Z, X, y)
        W, b = update(dW, db, W, b, scale)
        
    plt.subplot(1, 3, 2)
    plt.plot(range(iter), Train_loss, color='red')
    plt.xlabel('Iterations')
    plt.ylabel('Log Loss')
    plt.title('Loss Over Iterations')

    newY = predict(X, W, b)
    plt.subplot(1, 3, 3)
    plt.scatter(X * XMax, y * yMax, color='blue', label='Data points')
    plt.plot(X * XMax, newY * yMax, color='red', label='Regression Line')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title('Regression Line Fit')
    plt.legend()

    plt.tight_layout()
    print("Trained Weights:", W)
    print("Trained Bias:", b)
    print("XMax :", XMax)
    print("yMax :", yMax)
    plt.show()

    return W, b

def controleInput(prompt, min_val, max_val, input_type=float):
    while True:
        try:
            value = input_type(input(prompt))
            if min_val <= value <= max_val:
                return value
            else:
                print(f"Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("Invalid input. Please try again.")
            
def controleFilename(prompt):
    while True:
        filename = input(prompt)
        if os.path.isfile(filename):
            try:
                np.genfromtxt(filename, delimiter=',', skip_header=1)
                return filename
            except Exception as e:
                print(f"Error reading file: {e}. Please provide a valid CSV file.")
        else:
            print("File does not exist. Please provide a valid filename.")
