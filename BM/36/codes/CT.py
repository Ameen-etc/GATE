import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def dxdt(t, x):
    return x - x**2/200

x0 = 100
t = np.linspace(0, 100, 100000)
sol = odeint(dxdt, y0=x0, t=t, tfirst=True)

# Define the function x(nh)
def x_n_h(n, h, x_prev):
    a = (1 + h/2) * x_prev - (h/400) * x_prev**2
    x_n = (-(1 - h/2) + np.sqrt((1 - h/2)**2 + h * a / 100)) / (h/200)
    return x_n

# Define parameters
x0 = 100  # Initial value of x
h = 0.1    # Step size
num_steps = 100  # Number of steps

# Initialize array to store x values
x_values = np.zeros(num_steps + 1)
x_values[0] = x0

# Calculate x(nh) for all time steps using a for loop
for n in range(1, num_steps + 1):
    x_values[n] = x_n_h(n, h, x_values[n-1])

# Create a single figure with subplots
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

# Plot the solution of the ODE
ax[0].plot(t, sol, label='ODE Solution')
ax[0].set_xlabel('$t$')
ax[0].set_ylabel('$x(t)$')
ax[0].legend()

# Create a stem plot for all n
ax[1].stem(range(num_steps + 1), x_values, markerfmt=' ', basefmt=" ")
ax[1].set_xlabel('Time step (n)')
ax[1].set_ylabel('x')

plt.tight_layout()
plt.show()
