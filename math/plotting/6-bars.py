#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

name = ['Farrah', 'Fred', 'Felicia']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
fruits = ['apples', 'bananas', 'oranges', 'peaches']

for i in range(len(fruit)):
    plt.bar(
        name,
        fruit[i],
        bottom=np.sum(fruit[:i], axis=0),
        color=colors[i],
        width=0.5,
    )

plt.ylim(0, 80, 10)

plt.legend(
    loc="upper right",
    labels=fruits,
)

plt.title("Number of Fruit per Person")
plt.ylabel("Quantity of Fruit")

plt.show()
