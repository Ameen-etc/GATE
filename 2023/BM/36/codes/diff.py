import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Differential equation
def dxdt(t, x):
    return x - x**2 / 200

# Initial condition
x0 = 100

# Time span for the continuous solution
t_continuous = np.linspace(0, 100, 10000)  # Adjusted for a smoother line

# Solve ODE
sol = odeint(dxdt, y0=x0, t=t_continuous, tfirst=True)

# Read data from the .dat file
data = np.loadtxt('discrete_data.dat')

# Plotting
plt.figure(figsize=(12, 8))

# Plot continuous solution
plt.plot(t_continuous, sol, label='Continuous ODE Solution', linewidth=2)

# Plot discrete solution from the .dat file
plt.plot(data[:, 0], data[:, 1], 'o', label='Discrete Data from .dat File, h=0.1', markersize=4)

plt.xlabel('$t$')
plt.ylabel('$x(t)$')
plt.legend()
plt.title('Continuous and Discrete Solutions')
plt.show()

