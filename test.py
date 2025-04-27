import matplotlib.pyplot as plt
import numpy as np

xpoints = np.array([0, 6])
ypoints = np.array([0, 250])

# plt.plot(xpoints, ypoints)
# plt.show()

max_sequence_length = 4
token_list = [12, 14, 25, 32, 45, 50]
print(token_list[-(max_sequence_length-1):])