import numpy as np
from numba import jit
import time
from Bob.csv_writing import write_to_file
from Bob.plotting import plot_saw
import logging
import random
from numba import jitclass
from numba import int32, float32

log = logging.getLogger('__main__')
total_possible_walks = {
    1: 6,
    2: 30,
    3: 150,
    4: 726,
    5: 3534,
    6: 16926,
    7: 81390,
    8: 387966,
    9: 1853886,
    10: 8809878,
    11: 41934150,
    12: 198842742,
    13: 943974510,
    14: 4468911678
}


class MinEnergyMatrix:
    """
    Stores the minimum energy value and configurations with the energy for a saw of a particular length
    """

    def __init__(self, min_energy, matrix_config):
        self.min_energy = min_energy
        self.matrix_config = matrix_config

    def __str__(self):
        ret_val = "Min energy: " + str(self.min_energy) + "\n"
        for k in range(0, len(self.matrix_config)):
            ret_val += np.array2string(self.matrix_config[k], separator=",", precision=0) + ";\n"
        return ret_val


@jit
def do_run(n, data_file, directory, final):
    """
    Executes the computation of one SAW of a given length, write to file and saves image of the plot
    :param n: length of SAW being generated
    :param data_file: CSV file to write results to
    :param directory: test or final directory to save image to
    :param final: true if writing to final directory, false if writing to test
    :return:
    """
    energy_list = []
    min_config_main = MinEnergyMatrix(float('inf'), [])
    matrix = np.zeros(shape=(n + 1, 3), dtype=np.float32)
    matrix[0, :] = [0, 0, 0]
    matrix[1, :] = [1, 0, 0]
    start = time.clock()
    energies = recursively_generate_saws(1, matrix, energy_list, min_config_main)
    log.info("computation of %s SAWs completed" % str(len(energies) * 6))
    end = time.clock()
    total_time = end - start
    if total_time < 1:
        scaled_time = '%f ms' % (total_time * 1000)
    elif total_time < 60:
        scaled_time = '%f s' % total_time
    elif total_time < 3600:
        scaled_time = '%d min %d s' % (total_time // 60, total_time % 60)
    else:
        hours = total_time // 3600
        minutes = (total_time - (hours * 3600)) // 60
        seconds = total_time % 60
        scaled_time = '%d hr %d min %d s' % (hours, minutes, seconds)
    log.info("recursive generation of length %i took %s" % (n, scaled_time))
    write_to_file(data_file, final, n, min(energy_list), scaled_time, len(energy_list) * 6,
                  len(min_config_main.matrix_config))
    plot_saw(n, min_config_main, directory)


@jit('f8(f4[:,:])', nopython=True)
def energy_of_saw(saw_matrix):
    """
    Computes the energy of the self-avoiding walk passed into the function as per the equation from Chronin's paper
    :param saw_matrix: an N x 3 matrix of the x, y, z coordinates of each point in the SAW
    :return: a floating point energy value
    """
    midpoints = saw_matrix.copy()
    directions = saw_matrix.copy()
    # calculate direction of each filament and midpoints of N-1 filaments
    for a in range(0, midpoints.shape[0] - 1):
        directions[a] = (directions[a + 1] - directions[a])
        midpoints[a] = (midpoints[a] + midpoints[a + 1]) / 2.0
    # total energy in system, used to keep track of energy throughout summations
    total_energy = 0.0
    # outer summation from i = 0 to N-1, computes energy of every combination of filaments and tracks total for SAW
    for n in range(0, midpoints.shape[0] - 1):
        direction_i = directions[n]
        midpoints_i = midpoints[n]
        # inner summation from j = i+1 to N-1
        for j in range(n + 1, midpoints.shape[0] - 1):
            direction_j = directions[j]
            midpoint_j = midpoints[j]
            dot_product = (direction_i[0] * direction_j[0]) + (direction_i[1] * direction_j[1]) + (
                direction_i[2] * direction_j[2])
            x_dif = (midpoints_i[0] - midpoint_j[0]) ** 2.0
            y_dif = (midpoints_i[1] - midpoint_j[1]) ** 2.0
            z_dif = (midpoints_i[2] - midpoint_j[2]) ** 2.0
            distance_abs = abs((x_dif + y_dif + z_dif) ** (1.0 / 2.0))
            total_energy += dot_product / distance_abs
    # scale the total energy
    # return (1.0 / (8.0 * np.pi)) * total_energy
    return total_energy


@jit('b1(f4[:,:],f4[:,:])', nopython=True)
def matrix_comparison(matrix1, matrix2):
    """
    Checks to see if two matrices have at least one identical point. This function is very slow for large N, but is
    faster than numpy's method of comparison.
    :param matrix1: N x 3 matrix to be compared (3 cols are x y z coordinates)
    :param matrix2: N x 3 matrix to be compared (3 cols are x y z coordinates)
    :return: False if no identical points are found, True after first identical point found
    """
    rows_matrix1 = matrix1.shape[0]
    rows_matrix2 = matrix2.shape[0]
    for m in range(0, rows_matrix1):
        xyz1 = matrix1[m]
        for j in range(rows_matrix2 - 1, -1, -1):
            xyz2 = matrix2[j]
            if xyz2[0] == xyz1[0] and xyz2[1] == xyz1[1] and xyz2[2] == xyz1[2]:
                return True
    return False


@jit('i4[:,:](f4[:],f4[:])', nopython=True)
def find_directions(prev_point, point):
    """
    Given the direction of the current filament, this function finds all possible directions the next filament can move
    by subtracting prev_point from point to get the direction the filament is currently going and determining all
    allowed moves
    :param prev_point: the point before the current point
    :param point: the current point
    :return: a 2d integer array containing valid moves
    """
    direction_vector = point - prev_point
    options = np.zeros(shape=(5, 3), dtype=np.int32)
    if abs(direction_vector[0]) == 1:
        options[0, :] = [direction_vector[0], 0, 0]
        options[1, :] = [0, 1, 0]
        options[2, :] = [0, -1, 0]
        options[3, :] = [0, 0, 1]
        options[4, :] = [0, 0, -1]
    elif abs(direction_vector[1]) == 1:
        options[0, :] = [1, 0, 0]
        options[1, :] = [-1, 0, 0]
        options[2, :] = [0, direction_vector[1], 0]
        options[3, :] = [0, 0, 1]
        options[4, :] = [0, 0, -1]
    elif abs(direction_vector[2]) == 1:
        options[0, :] = [1, 0, 0]
        options[1, :] = [-1, 0, 0]
        options[2, :] = [0, 1, 0]
        options[3, :] = [0, -1, 0]
        options[4, :] = [0, 0, direction_vector[2]]
    return options


def recursively_generate_saws(point_n, working_matrix, energies, min_config):
    """
    Recursively generates all possible self avoiding walks
    :param point_n: point in the walk we are on, between 1 and N
    :param working_matrix: the x y z coordinates of SAW generated so far, N x 3 matrix
    :param energies: list of floating point numbers holding energies for each SAW
    :param min_config: object storing current min energy and configurations that have this energy
    :return: the list of energies for each SAW
    """
    # create a copy of the original matrix
    current_matrix = working_matrix.copy()
    # a single point, used to find direction vector
    point = current_matrix[point_n, :].copy()
    # point before current point, used to find possible directions vector
    previous_point = current_matrix[point_n - 1, :].copy()
    # returns all possible directions given the current direction
    options = find_directions(previous_point, point)
    # go through all possible options and find which directions are self-avoiding
    for pt in options:
        # find the next point with the given vector
        current_matrix[point_n + 1, :] = current_matrix[point_n, :] + pt
        # check to see if the filament intersects itself
        check = matrix_comparison(current_matrix[point_n + 1:point_n + 2, :], current_matrix[0:point_n, :])
        # if the filament does not intersect itself continue
        if not check:
            current_matrix_copy = current_matrix.copy()
            # if the filament is shorter than N make a recursive call
            if point_n <= working_matrix.shape[0] - 3:
                recursively_generate_saws(point_n + 1, current_matrix_copy, energies, min_config)
            # else add the energy of the filament to the energy list
            else:
                # calculate energy of current configuration
                this_energy = energy_of_saw(current_matrix_copy)
                # append to energy list
                energies.append(this_energy)
                # check if energy of this config is equal to min energy of filaments of this length with machine epsilon
                if abs(min_config.min_energy - this_energy) < 10 * np.finfo(float).eps:
                    # add to min energy list
                    min_config.matrix_config.append(current_matrix_copy)
                # check if energy of this config is less than min energy of filaments of this length
                elif this_energy < min_config.min_energy:
                    # set min energy to this config's energy value
                    min_config.min_energy = this_energy
                    # clear min configs stored so far
                    min_config.matrix_config.clear()
                    # add this config to list of min configs
                    min_config.matrix_config.append(current_matrix_copy)
    if random.random() < 0.0000001:
        total = str(total_possible_walks.get(working_matrix.shape[0] - 1)) if total_possible_walks.get(working_matrix.shape[0] - 1) is not None else '?'
        log.info("completed %s of %s" % (str(len(energies) * 6), total))
    return energies
