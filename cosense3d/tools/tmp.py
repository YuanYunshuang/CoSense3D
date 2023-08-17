import numpy as np
import torch
import glob
import os
import tqdm

import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [5, 7, 2, 8, 6]
labels = ['A', 'B', 'C', 'D', 'E']

# Create a scatter plot
plt.scatter(x, y, color='blue')

# Add labels to data points
for i, label in enumerate(labels):
    plt.annotate(label, (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Add labels to axes
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Scatter Plot with Labeled Data Points')

# Show the plot
plt.show()


