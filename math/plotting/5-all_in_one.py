#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

figure, axis = plt.subplots(3, 2)

####################
# 0. Line Graph
####################
ax1 = plt.subplot2grid((3, 2), (0, 0))

ax1.plot(y0, color='red')

ax1.set_xlim(0, 10)

####################
# 1. Scatter Plot
####################
ax2 = plt.subplot2grid((3, 2), (0, 1))

ax2.scatter(x1, y1, c='magenta', s=10)

ax2.set_title("Men's Height vs Weight", fontsize="x-small")

ax2.set_xlabel("Height (in)", fontsize="x-small")
ax2.set_ylabel("Weight (lbs)", fontsize="x-small")

####################
# 2. Change of Scale
####################
ax3 = plt.subplot2grid((3, 2), (1, 0))

ax3.plot(x2, y2)

ax3.set_title('Exponential Decay of C-14', fontsize="x-small")

ax3.set_xlabel('Time (years)', fontsize="x-small")
ax3.set_ylabel('Fraction Remaining', fontsize="x-small")

ax3.set_xlim(0, 28650)
ax3.set_yscale('log')

####################
# 3. Two or More
####################
ax4 = plt.subplot2grid((3, 2), (1, 1))

ax4.plot(x3, y31, '--', color='r', label='C-14')
ax4.plot(x3, y32, color='g', label='Ra-226')

ax4.set_title('Exponential Decay of Radioactive Elements', fontsize="x-small")

ax4.set_xlabel('Time (years)', fontsize="x-small")
ax4.set_ylabel('Fraction Remaining', fontsize="x-small")
ax4.legend()

ax4.set_xlim(0, 20000)
ax4.set_ylim(0, 1)

####################
# 4. Frequency
####################
ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)

ax5.hist(student_grades, bins=10, range=(0, 100), edgecolor='black')

ax5.set_title('Project A', fontsize="x-small")

ax5.set_xlabel('Grades', fontsize="x-small")
ax5.set_ylabel('Number of Students', fontsize="x-small")

ax5.set_xlim(0, 100)
ax5.set_xticks(np.arange(0, 101, 10))
ax5.set_yticks(np.arange(0, 31, 5))

####################
# 5. All in One
####################

figure.suptitle('All in One')
figure.tight_layout()

plt.show()
