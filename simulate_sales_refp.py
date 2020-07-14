import numpy as np
import matplotlib.pyplot as plt

from simtools import CustomerSimulator, DEFAULT_PARAMS_REF

DEFAULT_SIM = CustomerSimulator(DEFAULT_PARAMS_REF)
values = []
multiplied = []
DEFAULT_SIM.factor = 800
for p in range(0, 50):
    p_values = []
    p_mult = []
    for a in range(0, 50):
        sales_prob = DEFAULT_SIM.get_lambda(0.5, p, 31, a)
        p_values.append(sales_prob)
        p_mult.append(sales_prob * a)
    values.append(p_values)
    multiplied.append(p_mult)

arr = np.asarray(values)

plt.plot(arr[20], label="20")
plt.plot(arr[30], label="30")
plt.plot(arr[40], label="40")
plt.legend()
plt.show()

plt.plot(multiplied[10], label="10")
plt.plot(multiplied[30], label="30")
plt.plot(multiplied[49], label="50")
plt.legend()
plt.show()
