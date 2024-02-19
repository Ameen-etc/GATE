import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint

def dxdt(t, x):
    return x-x**2/200
x0=100

t=np.linspace(0, 100, 100000)
sol=odeint(dxdt, y0=x0, t=t, tfirst=True)

plt.plot(t, sol)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.show()
