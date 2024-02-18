import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker


def draw_temp_scan():
    loc1 = [0, 0]
    loc2 = [100, 0]
    angle1 = 0
    angle2 = -180
    time_offset = 0.05

    na, nr = 50, 10
    scan_angles1 = np.deg2rad(np.linspace(angle1, angle1 + 360 - 1, na))
    scan_angles2 = np.deg2rad(np.linspace(angle2, angle2 + 360 - 1, na))
    radii = np.linspace(0, 100, nr)

    a1, r1 = np.meshgrid(scan_angles1, radii)
    x1 = (r1 * np.cos(a1)).reshape(-1) + loc1[0]
    y1 = (r1 * np.sin(a1)).reshape(-1) + loc1[1]
    a2, r2 = np.meshgrid(scan_angles2, radii)
    x2 = (r2 * np.cos(a2)).reshape(-1) + loc2[0]
    y2 = (r2 * np.sin(a2)).reshape(-1) + loc2[1]

    angle_min = min(scan_angles1.min(), scan_angles2.min())
    angle_max = max(scan_angles1.max(), scan_angles2.max())
    color_scale1 = (a1 - angle_min).reshape(-1) / angle_max / 10
    color_scale2 = (a2 - angle_min).reshape(-1) / angle_max / 10
    sizes = r1 / 3

    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_subplot()
    plot = ax.scatter(x1, y1, c=color_scale1, cmap='jet', alpha=0.75, s=sizes, vmin=0, vmax=0.15)
    ax.scatter(x2, y2, c=color_scale2, cmap='jet', alpha=0.75, s=sizes, vmin=0, vmax=0.15)
    plt.colorbar(plot, label="Time (s)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("/home/yys/Pictures/temp_scan.png")
    plt.close()


if __name__ == "__main__":
    draw_temp_scan()

