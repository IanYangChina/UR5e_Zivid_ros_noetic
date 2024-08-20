import os
import numpy as np
import matplotlib.pyplot as plt

script_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(script_path, '..', 'soil_manipulation', 'src', 'moveit_trajectories')

task_id = 1
for n in range(4):
    timestamps = np.load(os.path.join(data_path, f'sys_id_{task_id}_timestamps_{n}.npy'))
    v = np.load(os.path.join(data_path, f'sys_id_{task_id}_v_{n}.npy'))
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(timestamps)
    ax[0].set_title(f'Last timestamp: {timestamps[-1]}')
    ax[1].plot(v)
    ax[1].legend(['x', 'y', 'z', 'a', 'b', 'c'])
    plt.tight_layout()
    plt.show()
    plt.close()
