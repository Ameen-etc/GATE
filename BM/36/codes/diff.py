import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Differential equation
def dxdt(t, x):
    return x - x**2 / 200

# Initial condition
x0 = 100

# Time span for the continuous solution
t = np.linspace(0, 100, 10000)  # Use a smaller number of points for efficiency

# Solve ODE
sol = odeint(dxdt, y0=x0, t=t, tfirst=True)

# Discrete-time implementation for h = 0.1
def x_n_h(h, x_prev):
    """Compute x at the next time step for given h."""
    a = (1 + h/2) * x_prev - (h/400) * x_prev**2
    x_n = (-(1 - h/2) + np.sqrt((1 - h/2)**2 + h * a / 100)) / (h/200)
    return x_n

# Function to compute discrete values for h = 0.1
def compute_discrete_values(h=0.1, num_steps=1000):
    """Compute x values for discrete time steps with h=0.1."""
    x_values = np.zeros(num_steps + 1)
    x_values[0] = x0
    for n in range(1, num_steps + 1):
        x_values[n] = x_n_h(h, x_values[n-1])
    return x_values

# Compute discrete values with h = 0.1
h = 0.1
num_steps = int(100 / h)  # Ensures coverage of the same time span
x_discrete = compute_discrete_values(h, num_steps)

# Plotting
plt.figure(figsize=(12, 8))

# Plot continuous solution
plt.plot(t, sol, label='Continuous ODE Solution')

# Plot discrete solution for h = 0.1
t_discrete = np.linspace(0, 100, num_steps + 1)
plt.plot(t_discrete, x_discrete, 'o', label=f'Discrete, h={h}')

plt.xlabel('$t$')
plt.ylabel('$x(t)$')
plt.legend()
plt.show()

