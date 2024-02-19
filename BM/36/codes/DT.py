import numpy as np
import matplotlib.pyplot as plt
# Define the function x(nh)
def x_n_h(n, h, x_prev):
    a = (1 + h/2) * x_prev - (h/400) * x_prev**2
    x_n = (-(1 - h/2) + np.sqrt((1 - h/2)**2 + h * a / 100)) / (h/200)
    return x_n

# Define parameters
x0 = 100  # Initial value of x
h = 0.1 # Step size
num_steps = 100 # Number of steps

# Initialize array to store x values
x_values = np.zeros(num_steps + 1)
x_values[0] = x0

# Calculate x(nh) for all time steps using a for loop
for n in range(1, num_steps + 1):
    x_values[n] = x_n_h(n, h, x_values[n-1])

# Create a stem plot for all n
plt.figure(figsize=(10, 6))
plt.stem(range(num_steps + 1), x_values, markerfmt=' ', basefmt=" ")
plt.xlabel('n')
plt.ylabel('x(n)')
plt.grid(True)
plt.show()
