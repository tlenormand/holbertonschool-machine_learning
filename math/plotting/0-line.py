#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

plt.plot([i for i in range(0, 11)], y, 'red')
plt.xlim(0, 10)

plt.show()
