import numpy as np
import matplotlib.pyplot as plt

from simtools import DEFAULT_SIM

values = []
multiplied = []
DEFAULT_SIM.factor = 800
for p in range(0, 50):
    p_values = []
    p_mult = []
    for a in range(0, 50):
        sales_prob = DEFAULT_SIM.get_lambda(0.5, a, p, 30)
        p_values.append(sales_prob)
        p_mult.append(sales_prob * a)
    values.append(p_values)
    multiplied.append(p_mult)

arr = np.asarray(values)

plt.plot(arr[20], label="20")
plt.plot(arr[30], label="30")
plt.plot(arr[40], label="40")
axes = plt.gca()
axes.set_xlabel("Price")
axes.set_ylabel("Expected sales")
plt.legend()
plt.show()

plt.plot(multiplied[10], label="10")
plt.plot(multiplied[30], label="30")
plt.plot(multiplied[49], label="50")
axes = plt.gca()
axes.set_xlabel("Price")
axes.set_ylabel("Expected profit")
plt.legend()
plt.show()
