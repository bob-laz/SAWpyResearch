import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
import logging


log = logging.getLogger('__main__')


def plot_saw(num_segments, min_configs, directory):
    """
    Plots all minimum energy SAWs of the length specified in num_segments, this length must be <= max saw length
    generated in main
    :param num_segments: length of filaments to be graphed, integer > 2
    :param min_configs: MinEnergyMatrix object
    :param directory: test or final directory to draw plots to
    """
    x_values, y_values, z_values = [], [], []

    num_configs = len(min_configs.matrix_config)
    log.info("%i min configs found for SAW of length %i" % (num_configs, num_segments))
    subplot_rows = int(math.ceil(math.sqrt(num_configs)))
    subplot_cols = int(math.ceil(math.sqrt(num_configs)))
    log.info("making plot with %i rows and %i cols" % (subplot_rows, subplot_cols))

    fig, axes = plt.subplots(subplot_rows, subplot_cols, subplot_kw=dict(projection='3d'), figsize=(14.0, 10.0))

    # delete extra figures
    j = 0
    if subplot_rows * subplot_cols > num_configs:
        log.info("fewer plot required than generated, deleting extra plots")
        for g in range(1, subplot_rows + 1):
            for h in range(1, subplot_cols + 1):
                j += 1
                if j > num_configs:
                    fig.delaxes(axes[g - 1, h - 1])

    title = 'Length %d Minimum Energy SAW Configurations' % num_segments
    plt.suptitle(title)
    plt.subplots_adjust(left=0, bottom=0.03, right=0.97, top=0.95, wspace=0.10, hspace=0.08)

    for p in range(num_configs):
        # determine the row and col of this figure
        sub_row = math.floor(p / math.sqrt(num_configs))
        sub_col = math.floor(p % math.sqrt(num_configs))
        # get x y and z values
        for r in range(len(min_configs.matrix_config[p])):
            x_values.append(min_configs.matrix_config[p][r][0])
            y_values.append(min_configs.matrix_config[p][r][1])
            z_values.append(min_configs.matrix_config[p][r][2])

        # change color of end points and set color of lines
        colors = ['r' for _ in range(len(x_values))]
        colors[0] = 'b'
        colors[-1] = 'b'

        # plot the points and connect with lines
        axes[sub_row, sub_col].scatter(x_values, y_values, z_values, c=colors)
        axes[sub_row, sub_col].plot(x_values, y_values, z_values, color='r')

        # Adjusts axes to use 0.5 step size
        axes[sub_row, sub_col].set_xticks(np.arange(min(x_values), max(x_values) + 0.5, 0.5))
        axes[sub_row, sub_col].set_yticks(np.arange(min(y_values), max(y_values) + 0.5, 0.5))
        axes[sub_row, sub_col].set_zticks(np.arange(min(z_values), max(z_values) + 0.5, 0.5))

        # label points
        for x, y, z in zip(x_values, y_values, z_values):
            label = '(%d,%d,%d)' % (x, y, z)
            axes[sub_row, sub_col].text(x, y, z, label)

        # set axis labels
        axes[sub_row, sub_col].set_xlabel('X')
        axes[sub_row, sub_col].set_ylabel('Y')
        axes[sub_row, sub_col].set_zlabel('Z')

        # clear x y z value lists for next iteration
        x_values.clear()
        y_values.clear()
        z_values.clear()
    file_name = directory + '%dgraph.png' % num_segments
    plt.savefig(file_name, dpi=192)
    log.info("plot for SAW of length %i saved" % num_segments)
