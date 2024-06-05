Linear Regression Model
This repository contains a simple implementation of a linear regression model using Python. The model trains on a given dataset to predict the target values based on the input features. The implementation includes gradient descent for optimization and visualization of the training process.

Prerequisites
Ensure you have the following libraries installed before running the code:

matplotlib
numpy
tqdm
os (built-in library)
You can install the required libraries using pip:

bash
Copy code
pip install matplotlib numpy tqdm
Functions
initialisation(X)
Initializes the weights (W) and bias (b) to zero.

Parameters:
X: Input features array.
Returns:
W: Initial weight (0).
b: Initial bias (0).
model(X, W, b)
Computes the linear model prediction.

Parameters:
X: Input features array.
W: Weight.
b: Bias.
Returns:
Z: Predicted values.
logLoss(Z, y)
Calculates the log loss between predictions and actual values.

Parameters:
Z: Predicted values.
y: Actual target values.
Returns:
loss: Computed log loss.
update(dW, db, W, b, learning_rate)
Updates the weights and bias using gradient descent.

Parameters:
dW: Gradient of the loss with respect to weights.
db: Gradient of the loss with respect to bias.
W: Current weight.
b: Current bias.
learning_rate: Learning rate for gradient descent.
Returns:
W: Updated weight.
b: Updated bias.
predict(X, W, b)
Generates predictions using the trained model.

Parameters:
X: Input features array.
W: Weight.
b: Bias.
Returns:
Z: Predicted values.
gradients(Z, X, y)
Computes the gradients for weights and bias.

Parameters:
Z: Predicted values.
X: Input features array.
y: Actual target values.
Returns:
dW: Gradient of the loss with respect to weights.
db: Gradient of the loss with respect to bias.
runModel(filename, scale=0.01, iter=50000)
Runs the linear regression model on the provided dataset and visualizes the results.

Parameters:
filename: Path to the CSV file containing the dataset.
scale: Learning rate for gradient descent (default: 0.01).
iter: Number of iterations for training (default: 50000).
Returns:
W: Trained weight.
b: Trained bias.
controleInput(prompt, min_val, max_val, input_type=float)
Prompts the user for input and validates it within a specified range.

Parameters:
prompt: Input prompt message.
min_val: Minimum acceptable value.
max_val: Maximum acceptable value.
input_type: Type of input (default: float).
Returns:
value: Validated input value.
controleFilename(prompt)
Prompts the user for a valid filename and verifies its existence.

Parameters:
prompt: Input prompt message.
Returns:
filename: Validated filename.
Usage
Prepare your dataset as a CSV file with the following format:

python
Copy code
mileage,price
15000,20000
30000,15000
...
Run the runModel function with the path to your CSV file:

python
Copy code
W, b = runModel('path_to_your_file.csv')
The model will train and display plots showing the initial data scatter plot, loss over iterations, and the regression line fit.

Example
Here's an example of how to use the script:

python
Copy code
if __name__ == "__main__":
    filename = controleFilename("Enter the path to your CSV file: ")
    scale = controleInput("Enter the learning rate (e.g., 0.01): ", 0.0001, 1.0)
    iter = controleInput("Enter the number of iterations (e.g., 50000): ", 100, 1000000, int)
    
    W, b = runModel(filename, scale, iter)
    print(f"Final weights: {W}, Final bias: {b}")
This script will guide you through providing the necessary inputs and running the linear regression model on your dataset.
