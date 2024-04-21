import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2


def calibration_plot():
    # Data for plots
    x = np.linspace(0, 1, 10)
    y1 = np.linspace(1, 0, 10)

    y2 = (x ** 2)[::-1]

    y3 = 1 - x ** 2
    y3[-1] = 0.05

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1
    axs[0].bar(x, y1, width=0.05)
    axs[0].set_title('Calibrated')
    axs[0].set_aspect('equal')
    axs[0].set_xlabel('Uncertainty')
    axs[0].set_ylabel('Weighted Accuracy')

    # Plot 2
    axs[1].bar(x, y2, width=0.05)
    axs[1].set_title('Overconfident')
    axs[1].set_aspect('equal')
    axs[1].set_xlabel('Uncertainty')

    # Plot 3
    axs[2].bar(x, y3, width=0.05)
    axs[2].set_title('Underconfident')
    axs[2].set_aspect('equal')
    axs[2].set_xlabel('Uncertainty')

    # Draw dashed black line
    for ax in axs:
        ax.plot([0, 1], [1, 0], 'k--')

    plt.savefig('/home/yuan/Pictures/calibration_plot.pdf')
    plt.close()







if __name__=="__main__":
    bevmap = cv2.imread("/cosense3d/carla/assets/maps/png/Town10HD_Opt.png")
    bev_roadmap_to_roadline(bevmap[..., 0] / 255)