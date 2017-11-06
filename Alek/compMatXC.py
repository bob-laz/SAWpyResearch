# -*- coding: utf-8 -*-
"""
Created on Tue Mar 08 20:26:18 2016

@author: lukanena
"""
import numpy as np
from numba import jit


# compare matrices. return False if no
# identical point are found. return True
# if an identical point exists. This function
# is very slow.
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


@jit(nopython=True)
def compare_matrix_to_matrix_rounded(mat_x1, mat_x2, seps):
    height_of_matrix1 = mat_x1.shape[0]
    height_of_matrix2 = mat_x2.shape[0]
    for pos in range(0, height_of_matrix1):
        pos_mat_x1 = mat_x1[pos]
        for xyz in range(0, height_of_matrix2):
            pos_mat_x2 = mat_x2[xyz]
            if (abs(pos_mat_x2[0] - pos_mat_x1[0]) <= seps and abs(pos_mat_x2[1] - pos_mat_x1[1]) <= seps and abs(
                        pos_mat_x2[2] - pos_mat_x1[2]) <= seps):
                return True
    return False


# does the same operation as compareMatrixToMatrix, but
# uses numpy to speed up the precess.
def compare_matrices_with_np(mat_x1, mat_x2):
    return (mat_x1[..., np.newaxis] == mat_x2[..., np.newaxis].T).all(1).any()
