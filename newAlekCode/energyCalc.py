# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:29:03 2017

@author: alek
"""

import numpy as np
import numba as nb


#####EQ. STATISTICS###############
##################################

# energy calculation of a given configuration.
@nb.jit('f8(f8[:,:])', nopython=True, nogil=True)
def energyInChainOPc(newConfig):
    midpoints = newConfig.copy()
    vecs = newConfig.copy()
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


@nb.jit(nopython=True, nogil=True)
def calculateMuTwoWithRn(rN, matX):
    point = int((matX.shape[0] - 1) / 2.0)
    pi = matX[point]
    pj = matX[point - 1]
    vec = pi - pj
    total_dotVecs = 0.0
    product_sub = (rN[0] * vec[0]) + (rN[1] * vec[1]) + (rN[2] * vec[2])
    total_dotVecs = total_dotVecs + product_sub
    return total_dotVecs


# just the distance between points
@nb.jit(nopython=True, nogil=True)
def calculateMuOne(start, end):
    xdist = (start[0] - end[0]) ** 2.0
    ydist = (start[1] - end[1]) ** 2.0
    zdist = (start[2] - end[2]) ** 2.0
    dista = (xdist + ydist + zdist) ** (1.0 / 2.0)
    return dista


# this is used in the ROOT MEAN SQUARED computation of mu_1,N
# I repeat ROOT MEAN SQUARED. Thats why there is a squared value
# at the end.
@nb.jit(nopython=True, nogil=True)
def calculateMuOneEdit(start, end):
    xdist = (start[0] - end[0]) ** 2.0
    ydist = (start[1] - end[1]) ** 2.0
    zdist = (start[2] - end[2]) ** 2.0
    dista = (xdist + ydist + zdist) ** (1.0 / 2.0)
    return dista ** 2.0


##################################
##################################
##################################


####TRANSFORMATION SECTION########
##################################

# check if the filament folded back on itself
@nb.jit(nopython=True, nogil=True)
def logicCheck(pb, p, pa):
    dire = p - pb
    new_dir = p + dire
    if not (pa[0] == new_dir[0] and pa[1] == new_dir[1] and pa[2] == new_dir[2]):
        return False
    return True


# New transformation
@nb.jit(nopython=True, nogil=True)
def diagonalMove(matX, pb, p, pa, point):
    # this function moves all points after the pivot point
    # down to one unit
    dire = p - pb
    new_dir = p + dire
    if not (pa[0] == new_dir[0] and pa[1] == new_dir[1] and pa[2] == new_dir[2]):
        # print 'here'
        dnew = (p - pb) + (p - pa)
        matXPar = matX[point + 1:matX.shape[0], :].copy()
        matXPar = matXPar + dnew
        matX[point + 1:matX.shape[0], :] = matXPar
    return matX


# New transformation
@nb.jit(nopython=True, nogil=True)
def reverseDiagonalMove(matX, p, pa, point, p_rand):
    # this function moves all of the points after the pivot point
    # down outward one unit in one of four directions
    dire = p - matX[point - 1]
    new_dir = p + dire
    p_dir = pa - p
    p_offset = p - pa
    if p_dir[0] != 0.0:
        h = abs(p_dir[0])
    elif p_dir[1] != 0.0:
        h = abs(p_dir[1])
    elif p_dir[2] != 0.0:
        h = abs(p_dir[2])

    if pa[0] == new_dir[0] and pa[1] == new_dir[1] and pa[2] == new_dir[2]:
        # print 'in method'
        p_gen = p.copy()
        if abs(p_dir[0]) == h:
            if p_rand == 0:
                p_gen[0] = 0.0
                p_gen[1] = h
                p_gen[2] = 0.0
            elif p_rand == 1:
                p_gen[0] = 0.0
                p_gen[1] = -h
                p_gen[2] = 0.0
            elif p_rand == 2:
                p_gen[0] = 0.0
                p_gen[1] = 0.0
                p_gen[2] = h
            elif p_rand == 3:
                p_gen[0] = 0.0
                p_gen[1] = 0.0
                p_gen[2] = -h
            matX[point + 1:matX.shape[0], :] = matX[point + 1:matX.shape[0], :] + (p_offset + p_gen)
        elif abs(p_dir[1]) == h:
            if p_rand == 0:
                p_gen[0] = h
                p_gen[1] = 0.0
                p_gen[2] = 0.0
            elif p_rand == 1:
                p_gen[0] = -h
                p_gen[1] = 0.0
                p_gen[2] = 0.0
            elif p_rand == 2:
                p_gen[0] = 0.0
                p_gen[1] = 0.0
                p_gen[2] = h
            elif p_rand == 3:
                p_gen[0] = 0.0
                p_gen[1] = 0.0
                p_gen[2] = -h
            matX[point + 1:matX.shape[0], :] = matX[point + 1:matX.shape[0], :] + (p_offset + p_gen)
        elif abs(p_dir[2]) == h:
            if p_rand == 0:
                p_gen[0] = h
                p_gen[1] = 0.0
                p_gen[2] = 0.0
            elif p_rand == 1:
                p_gen[0] = -h
                p_gen[1] = 0.0
                p_gen[2] = 0.0
            elif p_rand == 2:
                p_gen[0] = 0.0
                p_gen[1] = h
                p_gen[2] = 0.0
            elif p_rand == 3:
                p_gen[0] = 0.0
                p_gen[1] = -h
                p_gen[2] = 0.0
            matX[point + 1:matX.shape[0], :] = matX[point + 1:matX.shape[0], :] + (p_offset + p_gen)
    return matX

##################################
##################################
##################################
