# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 03:23:27 2016

@author: Aleksandr Lukanen
"""

import numpy as np
import numba as nb


@nb.jit(nopython=True, nogil=True)
def calculate_mu_two(mat_x):
    vecs = mat_x.copy()
    for a in range(0, vecs.shape[0] - 1):
        vecs[a] = (vecs[a + 1] - vecs[a])
    total_dot_vecs = 0.0
    veci = vecs[int((vecs.shape[0] - 1) / 2.0)]
    for j in range(0, vecs.shape[0] - 1):
        vecj = vecs[j]
        product_sub = (veci[0] * vecj[0]) + (veci[1] * vecj[1]) + (veci[2] * vecj[2])
        total_dot_vecs = total_dot_vecs + product_sub
    return total_dot_vecs


@nb.jit(nopython=True, nogil=True)
def calculate_mu_two_with_rn(rn, mat_x):
    point = int((mat_x.shape[0] - 1) / 2.0)
    pi = mat_x[point]
    pj = mat_x[point - 1]
    vec = pi - pj
    total_dot_vecs = 0.0
    product_sub = (rn[0] * vec[0]) + (rn[1] * vec[1]) + (rn[2] * vec[2])
    total_dot_vecs = total_dot_vecs + product_sub
    return total_dot_vecs


@nb.jit(nopython=True, nogil=True)
def calculate_mu_two_edit(mat_x):
    vecs = mat_x.copy()
    for a in range(0, vecs.shape[0] - 1):
        vecs[a] = (vecs[a + 1] - vecs[a])
    total_dot_vecs = 0.0
    for i in range(0, vecs.shape[0] - 1):
        total_dot_vecs_sub = 0.0
        veci = vecs[i]
        for j in range(0, vecs.shape[0] - 1):
            vecj = vecs[j]
            product_sub = (veci[0] * vecj[0]) + (veci[1] * vecj[1]) + (veci[2] * vecj[2])
            total_dot_vecs_sub = total_dot_vecs_sub + product_sub
        total_dot_vecs = total_dot_vecs + total_dot_vecs_sub
    return total_dot_vecs / (vecs.shape[0] - 1)


@nb.jit(nopython=True, nogil=True)
def mu_two_one(mat_x, rn):
    midpoints = mat_x.copy()
    vecs = mat_x.copy()
    for a in range(0, midpoints.shape[0] - 1):
        vecs[a] = (vecs[a + 1] - vecs[a])
        midpoints[a] = (midpoints[a] + midpoints[a + 1]) / 2.0
    total_dot_vecs = 0.0
    vecj = vecs[int((vecs.shape[0] - 1) / 2.0)]
    distj = midpoints[int((midpoints.shape[0] - 1) / 2.0)]
    for i in range(0, vecs.shape[0] - 1):
        veci = vecs[i]
        disti = midpoints[i]
        xdist = (disti[0] - distj[0]) ** 2.0
        ydist = (disti[1] - distj[1]) ** 2.0
        zdist = (disti[2] - distj[2]) ** 2.0
        dista = abs((xdist + ydist + zdist) ** (1.0 / 2.0))
        if dista <= rn:
            product_sub = (veci[0] * vecj[0]) + (veci[1] * vecj[1]) + (veci[2] * vecj[2])
            total_dot_vecs = total_dot_vecs + product_sub
    return total_dot_vecs


@nb.jit(nopython=True, nogil=True)
def mu_two_two(mat_x, rn):
    midpoints = mat_x.copy()
    vecs = mat_x.copy()
    for a in range(0, midpoints.shape[0] - 1):
        vecs[a] = (vecs[a + 1] - vecs[a])
        midpoints[a] = (midpoints[a] + midpoints[a + 1]) / 2.0
    total_dot_vecs = 0.0
    for i in range(0, vecs.shape[0] - 1):
        total_dot_vecs_sub = 0.0
        veci = vecs[i]
        disti = midpoints[i]
        for j in range(0, vecs.shape[0] - 1):
            vecj = vecs[j]
            distj = midpoints[j]
            xdist = (disti[0] - distj[0]) ** 2.0
            ydist = (disti[1] - distj[1]) ** 2.0
            zdist = (disti[2] - distj[2]) ** 2.0
            dista = abs((xdist + ydist + zdist) ** (1.0 / 2.0))
            if dista <= rn:
                product_sub = (veci[0] * vecj[0]) + (veci[1] * vecj[1]) + (veci[2] * vecj[2])
                total_dot_vecs_sub = total_dot_vecs_sub + product_sub
        total_dot_vecs = total_dot_vecs + total_dot_vecs_sub
    return total_dot_vecs / (vecs.shape[0] - 1)


@nb.jit(nopython=True, nogil=True)
def calculate_mu_one(start, end):
    xdist = (start[0] - end[0]) ** 2.0
    ydist = (start[1] - end[1]) ** 2.0
    zdist = (start[2] - end[2]) ** 2.0
    dista = (xdist + ydist + zdist) ** (1.0 / 2.0)
    return dista


@nb.jit(nopython=True, nogil=True)
def calculate_mu_one_edit(start, end):
    xdist = (start[0] - end[0]) ** 2.0
    ydist = (start[1] - end[1]) ** 2.0
    zdist = (start[2] - end[2]) ** 2.0
    dista = (xdist + ydist + zdist) ** (1.0 / 2.0)
    return dista ** 2.0


@nb.jit(nopython=True, nogil=True)
def logic_check(pb, p, pa):
    dire = p - pb
    new_dir = p + dire
    if not (pa[0] == new_dir[0] and pa[1] == new_dir[1] and pa[2] == new_dir[2]):
        return False
    return True


@nb.jit(nopython=True, nogil=True)
def logic_check_2(pb, p, pa):
    dire = p - pb
    new_dir = p + dire
    if pa[0] == new_dir[0] and pa[1] == new_dir[1] and pa[2] == new_dir[2]:
        return False
    return True


@nb.jit(nopython=True, nogil=True)
def diagonal_move(mat_x, pb, p, pa, point):
    # this function moves all points after the pivot point
    # down to one unit
    dire = p - pb
    new_dir = p + dire
    if not (pa[0] == new_dir[0] and pa[1] == new_dir[1] and pa[2] == new_dir[2]):
        # print 'here'
        dnew = (p - pb) + (p - pa)
        mat_x_par = mat_x[point + 1:mat_x.shape[0], :].copy()
        mat_x_par = mat_x_par + dnew
        mat_x[point + 1:mat_x.shape[0], :] = mat_x_par
    return mat_x


@nb.jit(nopython=True, nogil=True)
def reverse_diagonal_move(mat_x, p, pa, point, p_rand):
    # this function moves all of the points after the pivot point
    # down outward one unit in one of four directions
    global h
    dire = p - mat_x[point - 1]
    new_dir = p + dire
    p_dir = pa - p
    p_offset = p - pa
    if p_dir[0] != 0.0:
        h = abs(p_dir[0])
    elif p_dir[1] != 0.0:
        h = abs(p_dir[1])
    elif p_dir[2] != 0.0:
        h = abs(p_dir[2])
    # print new_dir
    # print pa
    # print (pa[0]==new_dir[0] and pa[1]==new_dir[1] and pa[2]==new_dir[2])
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
            mat_x[point + 1:mat_x.shape[0], :] = mat_x[point + 1:mat_x.shape[0], :] + (p_offset + p_gen)
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
            mat_x[point + 1:mat_x.shape[0], :] = mat_x[point + 1:mat_x.shape[0], :] + (p_offset + p_gen)
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
            mat_x[point + 1:mat_x.shape[0], :] = mat_x[point + 1:mat_x.shape[0], :] + (p_offset + p_gen)
    return mat_x


# this is just some random function (testing)
@nb.jit(nopython=True, nogil=True)
def make_contiguous(mult_mat, end_mat):
    end_mat[0, :] = mult_mat[0, :]
    end_mat[1, :] = mult_mat[1, :]
    end_mat[2, :] = mult_mat[2, :]
    return end_mat


@nb.jit(nopython=True)
def one_row_multiply_all(all_mats, to_delete, pa, p, pb):
    height = 3
    for i in range(0, all_mats.shape[0] / 3):
        froma = i * 3
        tob = froma + height
        a1 = all_mats[froma:tob, 0]
        a2 = all_mats[froma:tob, 1]
        a3 = all_mats[froma:tob, 2]
        adj = pa - p
        x = (adj[0] * a1[0] + adj[1] * a1[1] + adj[2] * a1[2]) + p[0]
        y = (adj[0] * a2[0] + adj[1] * a2[1] + adj[2] * a2[2]) + p[1]
        z = (adj[0] * a3[0] + adj[1] * a3[1] + adj[2] * a3[2]) + p[2]
        if pb[0] == x and pb[1] == y and pb[2] == z:
            to_delete[0][i] = -1
    dx = p[0] - pb[0]
    dy = p[1] - pb[1]
    dz = p[2] - pb[2]
    xp = p[0] + dx
    yp = p[1] + dy
    zp = p[2] + dz
    # print 'pa[0]=%f, xp=%f' % (pa[0],xp)
    # print 'pa[1]=%f, yp=%f' % (pa[1],yp)
    # print 'pa[2]=%f, zp=%f' % (pa[2],zp)
    if pa[0] == xp and pa[1] == yp and pa[2] == zp:
        to_delete[0][9] = -1
    if not (pa[0] == xp and pa[1] == yp and pa[2] == zp):
        to_delete[0][10] = -1
    return to_delete


# @nb.jit(nopython=True)
def one_row_multiply_all_minus_one(all_mats, to_delete, pa, p, pb):
    height = 3
    for i in range(0, all_mats.shape[0] / 3):
        print(i)
        froma = i * 3
        tob = froma + height
        a1 = all_mats[froma:tob, 0]
        a2 = all_mats[froma:tob, 1]
        a3 = all_mats[froma:tob, 2]
        adj = pa - p
        print(adj)
        x = (adj[0] * a1[0] + adj[1] * a1[1] + adj[2] * a1[2]) + p[0]
        y = (adj[0] * a2[0] + adj[1] * a2[1] + adj[2] * a2[2]) + p[1]
        z = (adj[0] * a3[0] + adj[1] * a3[1] + adj[2] * a3[2]) + p[2]
        print('x: ', x)
        print('y: ', y)
        print('z: ', z)
        if pb[0] == x and pb[1] == y and pb[2] == z:
            to_delete[0][i] = -1
            print('here')
    return to_delete


@nb.jit(nopython=True)
def one_row_multiply(a, pa, p, pb):
    a1 = a[:, 0]
    a2 = a[:, 1]
    a3 = a[:, 2]
    adj = pa - p
    x = (adj[0] * a1[0] + adj[1] * a1[1] + adj[2] * a1[2]) + p[0]
    y = (adj[0] * a2[0] + adj[1] * a2[1] + adj[2] * a2[2]) + p[1]
    z = (adj[0] * a3[0] + adj[1] * a3[1] + adj[2] * a3[2]) + p[2]
    if pb[0] == x and pb[1] == y and pb[2] == z:
        return True
    return False


@nb.jit(nopython=True, nogil=True)
def energy_in_chain_o_p_tc(new_config, ai, i_s):
    midpoints = new_config
    vecs = new_config
    for a in range(0, midpoints.shape[0] - 1):
        vecs[a] = (vecs[a + 1] - vecs[a])
        midpoints[a] = (midpoints[a] + midpoints[a + 1]) / 2.0
    # total energy in system
    total_energy = 0.0
    for i in range(i_s, i_s + ai):
        veci = vecs[i]
        x_posi = midpoints[i]
        for j in range(i + 1, midpoints.shape[0] - 1):
            vecj = vecs[j]
            x_posj = midpoints[j]
            product_sub = (veci[0] * vecj[0]) + (veci[1] * vecj[1]) + (veci[2] * vecj[2])
            xdif = (x_posi[0] - x_posj[0]) ** 2.0
            ydif = (x_posi[1] - x_posj[1]) ** 2.0
            zdif = (x_posi[2] - x_posj[2]) ** 2.0
            dif_sub = abs((xdif + ydif + zdif) ** (1.0 / 2.0))
            quot_sub = product_sub / dif_sub
            total_energy = total_energy + quot_sub
    # scale the total energy
    answer_last = (1.0 / (4.0 * np.pi)) * total_energy
    return answer_last


#@nb.jit('f8(f8[:,:])', nopython=True, nogil=True)
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
        x_posi = midpoints[i]
        for j in range(i + 1, midpoints.shape[0] - 1):
            vecj = vecs[j]
            x_posj = midpoints[j]
            product_sub = (veci[0] * vecj[0]) + (veci[1] * vecj[1]) + (veci[2] * vecj[2])
            xdif = (x_posi[0] - x_posj[0]) ** 2.0
            ydif = (x_posi[1] - x_posj[1]) ** 2.0
            zdif = (x_posi[2] - x_posj[2]) ** 2.0
            dif_sub = abs((xdif + ydif + zdif) ** (1.0 / 2.0))
            quot_sub = product_sub / dif_sub
            total_energy = total_energy + quot_sub
    # scale the total energy
    answer_last = (1.0 / (4.0 * np.pi)) * total_energy
    return answer_last


@nb.jit(nopython=True, nogil=True)
def initial_energy_map(config, energy_map):
    midpoints = config.copy()
    vecs = config.copy()
    for a in range(0, midpoints.shape[0] - 1):
        vecs[a] = (vecs[a + 1] - vecs[a])
        midpoints[a] = (midpoints[a] + midpoints[a + 1]) / 2.0

    for i in range(0, midpoints.shape[0] - 1):
        veci = vecs[i]
        x_posi = midpoints[i]
        for j in range(i + 1, midpoints.shape[0] - 1):
            vecj = vecs[j]
            x_posj = midpoints[j]
            product_sub = (veci[0] * vecj[0]) + (veci[1] * vecj[1]) + (veci[2] * vecj[2])
            xdif = (x_posi[0] - x_posj[0]) ** 2.0
            ydif = (x_posi[1] - x_posj[1]) ** 2.0
            zdif = (x_posi[2] - x_posj[2]) ** 2.0
            dif_sub = abs((xdif + ydif + zdif) ** (1.0 / 2.0))
            quot_sub = product_sub / dif_sub
            energy_map[i][j] = quot_sub
    # return energy map
    return energy_map


@nb.jit(nopython=True, nogil=True)
def sum_energy_map(energy_map):
    total_energy = 0.0
    for i in range(0, energy_map.shape[0]):
        for j in range(i, energy_map.shape[1]):
            total_energy = total_energy + energy_map[i][j]
    return total_energy


@nb.jit(nopython=True, nogil=True)
def energy_in_chain_with_e_mat(new_config, energy_map, pivot_point):
    # the pivot point should be in zero to N format
    midpoints = new_config.copy()
    vecs = new_config.copy()
    for a in range(0, midpoints.shape[0] - 1):
        vecs[a] = (vecs[a + 1] - vecs[a])
        midpoints[a] = (midpoints[a] + midpoints[a + 1]) / 2.0
    # total energy in system
    di = pivot_point - 1
    dj = pivot_point
    for i in range(0, di + 1):
        veci = vecs[i]
        x_posi = midpoints[i]
        for j in range(dj, midpoints.shape[0] - 1):
            vecj = vecs[j]
            x_posj = midpoints[j]
            product_sub = (veci[0] * vecj[0]) + (veci[1] * vecj[1]) + (veci[2] * vecj[2])
            xdif = (x_posi[0] - x_posj[0]) ** 2.0
            ydif = (x_posi[1] - x_posj[1]) ** 2.0
            zdif = (x_posi[2] - x_posj[2]) ** 2.0
            dif_sub = abs((xdif + ydif + zdif) ** (1.0 / 2.0))
            quot_sub = product_sub / dif_sub
            energy_map[i][j] = quot_sub
    # scale the total energy
    return energy_map


def energy_in_chain_with_e_mat2(new_config, energy_map, pivot_point):
    # the pivot point should be in zero to N format
    midpoints = new_config.copy()
    vecs = new_config.copy()
    for a in range(0, midpoints.shape[0] - 1):
        vecs[a] = (vecs[a + 1] - vecs[a])
        midpoints[a] = (midpoints[a] + midpoints[a + 1]) / 2.0
    print(vecs)
    # total energy in system
    di = pivot_point - 1
    dj = pivot_point
    for i in range(0, di + 1):
        veci = vecs[i]
        x_posi = midpoints[i]
        for j in range(dj, midpoints.shape[0] - 1):
            vecj = vecs[j]
            x_posj = midpoints[j]
            product_sub = (veci[0] * vecj[0]) + (veci[1] * vecj[1]) + (veci[2] * vecj[2])
            xdif = (x_posi[0] - x_posj[0]) ** 2.0
            ydif = (x_posi[1] - x_posj[1]) ** 2.0
            zdif = (x_posi[2] - x_posj[2]) ** 2.0
            dif_sub = abs((xdif + ydif + zdif) ** (1.0 / 2.0))
            quot_sub = product_sub / dif_sub
            print('dif_sub: ', dif_sub)
            print('product: ', product_sub)
            print('quot_sub: ', quot_sub)
            energy_map[i][j] = quot_sub
    # scale the total energy
    return energy_map


# @jit(nopython=False)
@nb.jit(nopython=False)
def energy_in_chain_optimized(new_config):
    p = np.delete(new_config, 0, axis=0)
    t = np.delete(new_config, new_config.shape[0] - 1, axis=0)
    # finds the midpoints of the matrix
    midpoints = (p + t) / 2.0
    # finds the vectors of the matrix
    vecs = p - t
    # total energy in system
    total_energy = 0.0
    for i in range(0, midpoints.shape[0] - 1):
        veci = vecs[i]
        x_posi = midpoints[i]
        for j in range(i + 1, midpoints.shape[0]):
            vecj = vecs[j]
            x_posj = midpoints[j]
            product_sub = (veci[0] * vecj[0]) + (veci[1] * vecj[1]) + (veci[2] * vecj[2])
            xdif = (x_posi[0] - x_posj[0]) ** 2.0
            ydif = (x_posi[1] - x_posj[1]) ** 2.0
            zdif = (x_posi[2] - x_posj[2]) ** 2.0
            dif_sub = abs((xdif + ydif + zdif) ** (1.0 / 2.0))
            quot_sub = product_sub / dif_sub
            total_energy = total_energy + quot_sub
    # scale the total energy
    answer_last = (1.0 / (4.0 * np.pi)) * total_energy
    return answer_last


# calculate the energy in a given configuration
def energy_in_chain(new_config):
    mat_new = new_config
    midpoints = get_mid_points(mat_new.copy())
    vecs = get_vectors(mat_new.copy())

    total_energy = 0.0

    for i in range(0, midpoints.shape[0] - 1):

        # ith vector
        veci = vecs[i]

        # ith position
        x_posi = midpoints[i]

        for j in range(i + 1, midpoints.shape[0]):
            # vector at position j
            vecj = vecs[j]

            # position a position j using midpoints
            x_posj = midpoints[j]

            # calculate product of veci and vecj
            product_sub = calc_dot_product(veci, vecj)

            # calculate the distance between xPosi and x_posj
            dif_sub = calc_distance(x_posi, x_posj)

            # divide product_sub by dif_sub
            quot_sub = product_sub / dif_sub

            # add quot_sub to the total_energy
            total_energy = total_energy + quot_sub

    # scale the total energy
    answer_last = (1.0 / (4.0 * np.pi)) * total_energy
    return answer_last


# this will return the vector based on the
# midpoint of the lines in the chain. The scale
# of each part will either be a 0 or the unit of length
# set at the start fo the program.
def get_vectors(mat_x):
    p = mat_x.copy()
    t = mat_x.copy()
    p = np.delete(p, 0, axis=0)
    t = np.delete(t, t.shape[0] - 1, axis=0)
    p_t = p - t
    # print p_t
    return p_t


def calc_dot_product(veci, vecj):
    dot_vector = (veci[0] * vecj[0]) + (veci[1] * vecj[1]) + (veci[2] * vecj[2])
    return dot_vector


def calc_distance(x_posi, x_posj):
    xdif = (x_posi[0] - x_posj[0]) ** 2.0
    ydif = (x_posi[1] - x_posj[1]) ** 2.0
    zdif = (x_posi[2] - x_posj[2]) ** 2.0

    distance = abs((xdif + ydif + zdif) ** (1.0 / 2.0))
    return distance


# returns the midpoints of the vectors
def get_mid_points(mat_x):
    p = mat_x.copy()
    t = mat_x.copy()
    p = np.delete(p, 0, axis=0)
    t = np.delete(t, t.shape[0] - 1, axis=0)
    p_t = (p + t) / 2.0
    return p_t
