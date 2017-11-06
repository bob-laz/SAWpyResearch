# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 13:49:24 2016

@author: Aleksandr Lukanen
"""

import numpy as np
import csv
from numba import jit


# compare matrices. return False if no
# identical point are found. return True
# if an identical point exists. This function
# is very slow for large N, but is faster than
# numpy's method of comparison.
@jit(nopython=True)
def compare_matrix_to_matrix(mat_x1, mat_x2):
    height_of_matrix1 = mat_x1.shape[0]
    height_of_matrix2 = mat_x2.shape[0]
    for pos in range(0, height_of_matrix1):
        pos_mat_x1 = mat_x1[pos]
        for xyz in range(height_of_matrix2 - 1, -1, -1):
            pos_mat_x2 = mat_x2[xyz]
            if pos_mat_x2[0] == pos_mat_x1[0] and pos_mat_x2[1] == pos_mat_x1[1] and pos_mat_x2[2] == pos_mat_x1[2]:
                return True
    return False


# calculate the energy of the filament
@jit('f8(f8[:,:])', nopython=True, nogil=True)
def energy_in_chain_o_pc(new_config):
    midpoints = new_config.copy()
    vecs = new_config.copy()
    for a in range(0, midpoints.shape[0] - 1):
        vecs[a] = (vecs[a + 1] - vecs[a])
        midpoints[a] = (midpoints[a] + midpoints[a + 1]) / 2.0
    # total energy in system
    total_energy = 0.0
    for i in range(0, midpoints.shape[0] - 1):
        veci = vecs[i]
        xPosi = midpoints[i]
        for j in range(i + 1, midpoints.shape[0] - 1):
            vecj = vecs[j]
            xPosj = midpoints[j]
            product_sub = (veci[0] * vecj[0]) + (veci[1] * vecj[1]) + (veci[2] * vecj[2])
            xdif = (xPosi[0] - xPosj[0]) ** 2.0
            ydif = (xPosi[1] - xPosj[1]) ** 2.0
            zdif = (xPosi[2] - xPosj[2]) ** 2.0
            dif_sub = abs((xdif + ydif + zdif) ** (1.0 / 2.0))
            quot_sub = product_sub / dif_sub
            total_energy = total_energy + quot_sub
    # scale the total energy
    answer_last = (1.0 / (4.0 * np.pi)) * total_energy
    return answer_last


# given the direction the current segment is going
# find all possible directions the filament can move.
def find_directions(point, point_before):
    vector = point - point_before
    if abs(vector[0]) == 1:
        options = np.zeros(shape=(5, 3))
        options[0, :] = [vector[0], 0, 0]
        options[1, :] = [0, 1, 0]
        options[2, :] = [0, -1, 0]
        options[3, :] = [0, 0, 1]
        options[4, :] = [0, 0, -1]
    elif abs(vector[1]) == 1:
        options = np.zeros(shape=(5, 3))
        options[0, :] = [1, 0, 0]
        options[1, :] = [-1, 0, 0]
        options[2, :] = [0, vector[1], 0]
        options[3, :] = [0, 0, 1]
        options[4, :] = [0, 0, -1]
    elif abs(vector[2]) == 1:
        options = np.zeros(shape=(5, 3))
        options[0, :] = [1, 0, 0]
        options[1, :] = [-1, 0, 0]
        options[2, :] = [0, 1, 0]
        options[3, :] = [0, -1, 0]
        options[4, :] = [0, 0, vector[2]]

    return options


def recus(point_n, matrix, energies):
    # create a copy of the original matrix
    matrix_sub = matrix.copy()
    # a single point, used to find direction vector
    point = matrix_sub[point_n, :].copy()
    # a single point, used to find direction vector
    point_before = matrix_sub[point_n - 1, :].copy()
    # returns all possible directions given the current direction
    options = find_directions(point, point_before)
    # go through all possible options and find which directions are self-avoiding
    for pt in options:
        # find the next point with the given vector
        matrix_sub[point_n + 1, :] = matrix_sub[point_n, :] + pt
        # check to see if the filament intersects itself
        check = compare_matrix_to_matrix(matrix_sub[point_n + 1:point_n + 2, :], matrix_sub[0:point_n, :])
        # if the filament does not intersect itself continue
        if not check:
            matrix_sub_sub = matrix_sub.copy()
            # if the filament is shorter than N make a recursive call
            if point_n <= matrix.shape[0] - 3:
                recus(point_n + 1, matrix_sub_sub, energies)
            # else add the energy of the filament to the energy list
            else:
                energies.append(energy_in_chain_o_pc(matrix_sub_sub))
    return energies


if __name__ == '__main__':
    # contains the N+1 values you want to enumerate
    # example: N=10 would be entered as 11
    N_s = [5]
    for N in N_s:
        total_energy = 0.0
        energy_list = []
        count = 0
        fileName = 'N%d.csv' % (N - 1)
        # start from a single vector on the x-axis
        matrix = np.zeros(shape=(N, 3))
        matrix[0, :] = [0, 0, 0]
        matrix[1, :] = [1, 0, 0]
        # find all filaments of length N
        recus(1, matrix, energy_list)
        # number of filaments
        count = len(energy_list)
        # total the energy
        for i in energy_list:
            total_energy = total_energy + i
        # write all E_i to csv file
        f = open(fileName, 'w')
        try:
            writer = csv.writer(f)
            writer.writerow(("Ei",))
            for a in energy_list:
                writer.writerow((a,))
        finally:
            f.close()
        # print out values
        print(('total_energy: ', total_energy))
        print(('average_energy(1/6): ', (total_energy / count)))
        print(('average_energy(1): ', (total_energy / count) * 6.0))
        print(('count: ', count))
