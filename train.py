from ft_linear_regression import *

filename = controleFilename("Enter the filename for the dataset: ")
scale = controleInput("Enter the learning rate (scale) [0-0.9]: ", 0, 0.9, float)
iter = controleInput("Enter the number of iterations [1-50000]: ", 1, 50000, int)

W, b = runModel(filename, scale, iter)
