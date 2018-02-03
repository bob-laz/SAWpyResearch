# -*- coding: utf-8 -*-
"""
Created on Tue Mar 08 20:26:18 2016

@author: lukanena
"""
import numpy as np
from numba import jit


# returns:
# True - if filament intersects
# False - if not
@jit(nopython=True)
def checkForSelfIntersection(filament):
    heightOfMatrix = filament.shape[0]
    for pos1 in range(0, heightOfMatrix):
        for pos2 in range(pos1 + 1, heightOfMatrix):
            if (filament[pos1][0] == filament[pos2][0] and filament[pos1][1] == filament[pos2][1] and filament[pos1][
                    2] == filament[pos2][2]):
                return True
    return False


# compare matrices. return False if no
# identical point are found. return True
# if an identical point exists. This function
# is very slow.
@jit(nopython=True)
def compareMatrixToMatrix(matX1, matX2):
    heightOfMatrix1 = matX1.shape[0]
    heightOfMatrix2 = matX2.shape[0]
    for pos in range(0, heightOfMatrix1):
        posMatX1 = matX1[pos]
        for xyz in range(heightOfMatrix2 - 1, -1, -1):
            posMatX2 = matX2[xyz]
            if posMatX2[0] == posMatX1[0] and posMatX2[1] == posMatX1[1] and posMatX2[2] == posMatX1[2]:
                return True
    return False


@jit(nopython=True)
def compareMatrixToMatrixRounded(matX1, matX2, seps):
    heightOfMatrix1 = matX1.shape[0]
    heightOfMatrix2 = matX2.shape[0]
    for pos in range(0, heightOfMatrix1):
        posMatX1 = matX1[pos]
        for xyz in range(0, heightOfMatrix2):
            posMatX2 = matX2[xyz]
            if (abs(posMatX2[0] - posMatX1[0]) <= seps and abs(posMatX2[1] - posMatX1[1]) <= seps and abs(
                        posMatX2[2] - posMatX1[2]) <= seps):
                return True
    return False


# does the same operation as compareMatrixToMatrix, but
# uses numpy to speed up the precess.
def compareMatricisWithNP(matX1, matX2):
    return (matX1[..., np.newaxis] == matX2[..., np.newaxis].T).all(1).any()
