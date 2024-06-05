# Linear Regression Model

This repository contains a simple implementation of a linear regression model using Python. The model trains on a given dataset to predict the target values based on the input features. The implementation includes gradient descent for optimization and visualization of the training process.

## Prerequisites

Ensure you have the following libraries installed before running the code:

- matplotlib
- numpy
- tqdm
- os (built-in library)

You can install the required libraries using pip:

```bash
pip install matplotlib numpy tqdm
```

## Usage

1. Prepare your dataset as a CSV file with the following format: Mileage,Price

2. Run training
```bash
python3 train.py
```
return trained model weight, bias, xMax, yMax

3. Run predict
```bash
python3 predict.py
```
return the price for a specific mileage with the data of your trained model
