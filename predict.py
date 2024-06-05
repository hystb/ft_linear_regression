from ft_linear_regression import *

value = controleInput("Enter the mileage [0 - 100 000 000]: ", 0, 100000000, int)
Weight = controleInput("Enter the Weight [-10 - 10]: ", -10, 10, float)
Bias = controleInput("Enter the Bias [-10 - 10]: ", -10, 10, float)
xMax = controleInput("Enter the XMax [0 - 100 000 000]: ", 0, 100000000, int)
yMax = controleInput("Enter the yMax [0 - 100 000 000]: ", 0, 100000000, int)

X = predict(value/xMax, Weight, Bias)
X *= yMax
print(f"Estimated price {X}")
