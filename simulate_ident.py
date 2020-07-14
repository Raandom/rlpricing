import numpy as np
import matplotlib.pyplot as plt

from simtools import DEFAULT_SIM

values = []
multiplied = []
DEFAULT_SIM.factor = 800
for p in range(0, 50):
    sales_prob = DEFAULT_SIM.get_lambda(0.5, p, p, p)
    values.append(sales_prob)
    multiplied.append(sales_prob * p)

arr = np.asarray(values)

plt.plot(arr)
plt.legend()
plt.show()

print("Argmax prob")
print(np.argmax(arr))

plt.plot(multiplied)
plt.legend()
plt.show()

print("Argmax profit")
print(np.argmax(multiplied))