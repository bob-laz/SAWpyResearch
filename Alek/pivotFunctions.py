# -*- coding: utf-8 -*-
"""
Created on Fri May 20 18:32:31 2016

@author: Aleksandr Lukanen
"""

from math import exp, log
from sys import exit

import matplotlib as mpl
# mpl.use('Agg')
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from Alek import compMatXC as comp
from Alek import energyCalc


def central_function(bend_type, N, B, n, dn, mat_x, has_matrix, return_value, burn, has_quo, quo, try_new, h, proci,
                     has_new_trans, seed, ddn, initial, has_s):
    if not has_quo:
        if bend_type == 1:
            if burn:
                sub_val = all_bends_alg_all(B, N, dn, mat_x, has_matrix)
                sub_values = all_bends_alg_all(B, N, n, sub_val[1], True)
                sub_values.append(sub_values)
                return sub_values[return_value]
            elif not burn:
                sub_val = all_bends_alg_all(B, N, n, mat_x, has_matrix, h, try_new, has_new_trans, seed)
                sub_val.append(sub_val)
                return sub_val[return_value]
        elif bend_type == 0:
            if burn:
                sub_val = limited_bends(B, N, dn, mat_x, has_matrix)
                sub_values = limited_bends(B, N, n, sub_val[1], True)
                sub_values.append(sub_values)
                return sub_values[return_value]
            elif not burn:
                sub_val = limited_bends(B, N, n, mat_x, has_matrix, h, seed, try_new, has_new_trans)
                sub_val.append(sub_val)
                return sub_val[return_value]
        else:
            print('--not a bend selection')
    elif has_quo:
        if bend_type == 1:
            if burn:
                sub_val = all_bends_alg_all(B, N, dn, mat_x, has_matrix, h, try_new, has_new_trans, seed)
                sub_values = all_bends_alg_all(B, N, n, sub_val[1], True, h, try_new, has_new_trans, seed)
                sub_values.append(sub_values)
                if not initial:
                    if has_s:
                        quo.put(
                            [sub_values[0], sub_values[2], sub_values[3], N, dn, n, B, try_new, has_new_trans, h, ddn,
                             bend_type, proci, sub_values[1]])
                    else:
                        quo.put(
                            [sub_values[0], sub_values[2], sub_values[3], N, dn, n, B, try_new, has_new_trans, h, ddn,
                             bend_type, proci])
                else:
                    quo.put([proci, sub_values[1]])
                quo.close()
                exit()
            elif not burn:
                sub_val = all_bends_alg_all(B, N, n, mat_x, has_matrix, h, try_new, has_new_trans, seed)
                sub_val.append(sub_val)
                if not initial:
                    if has_s:
                        quo.put(
                            [sub_values[0], sub_values[2], sub_values[3], N, dn, n, B, try_new, has_new_trans, h, ddn,
                             bend_type, proci, sub_values[1]])
                    else:
                        quo.put(
                            [sub_values[0], sub_values[2], sub_values[3], N, dn, n, B, try_new, has_new_trans, h, ddn,
                             bend_type, proci])
                else:
                    quo.put([proci, sub_val[1]])
                quo.close()
                exit()
        elif bend_type == 0:
            if burn:
                sub_val = limited_bends(B, N, dn, mat_x, has_matrix, h, seed, try_new, has_new_trans)
                sub_values = limited_bends(B, N, n, sub_val[1], True, h, seed, try_new, has_new_trans)
                sub_values.append(sub_values)
                if not initial:
                    if has_s:
                        quo.put(
                            [sub_values[0], sub_values[2], sub_values[3], N, dn, n, B, try_new, has_new_trans, h, ddn,
                             bend_type, proci, sub_values[1]])
                    else:
                        quo.put(
                            [sub_values[0], sub_values[2], sub_values[3], N, dn, n, B, try_new, has_new_trans, h, ddn,
                             bend_type, proci])
                else:
                    quo.put([proci, sub_values[1]])
                quo.close()
                exit()
            elif not burn:
                sub_val = limited_bends(B, N, n, mat_x, has_matrix, h, -1.0, try_new, has_new_trans)
                sub_val.append(sub_val)
                if not initial:
                    if has_s:
                        quo.put(
                            [sub_values[0], sub_values[2], sub_values[3], N, dn, n, B, try_new, has_new_trans, h, ddn,
                             bend_type, proci, sub_values[1]])
                    else:
                        quo.put(
                            [sub_values[0], sub_values[2], sub_values[3], N, dn, n, B, try_new, has_new_trans, h, ddn,
                             bend_type, proci])
                else:
                    quo.put([proci, sub_val[1]])
                quo.close()
                exit()
        else:
            print('--not a bend selection')


def all_bends_alg_all(B, N, n, mat_x, has_matrix, h, try_new, has_new_trans, seed):
    if seed != -1.0:
        np.random.seed(seed)
    else:
        np.random.seed()

    if not has_matrix:
        matrix = np.zeros(shape=(N, 3))
        for iy in range(0, matrix.shape[0]):
            matrix[iy, 0] = iy * h
            matrix[iy, 1] = 0
            matrix[iy, 2] = 0

    elif has_matrix:
        matrix = mat_x

    # current_energy = 0.0
    mean_energy = 0.0
    mu_one = 0.0
    mu_two = 0.0

    comp_mat_x = np.zeros(shape=[3, 3])
    comp_mat_x[0] = [1, 0, 0]
    comp_mat_x[1] = [0, 1, 0]
    comp_mat_x[2] = [0, 0, 1]

    current_energy = energyCalc.energy_in_chain_o_pc(matrix)
    mean_energy = mean_energy + current_energy
    mu_two = mu_two + energyCalc.calculate_mu_two(matrix)
    mu_one = mu_one + energyCalc.calculate_mu_one_edit(matrix[0], matrix[-1]) / n
    count = 1

    for i in range(1, n):
        matrix_sub = matrix.copy()
        mult_mat = np.zeros(shape=[3, 3])
        pos_in_mat = np.random.random_integers(1, matrix_sub.shape[0] - 2)
        center = matrix_sub[pos_in_mat, :].copy()
        str_section = False
        if has_new_trans:
            # check if on straigh section of filament
            if energyCalc.logic_check(matrix_sub[pos_in_mat - 1, :], center, matrix_sub[pos_in_mat + 1, :]):
                str_section = True
                if pos_in_mat == matrix_sub.shape[0] - 2:
                    choice = np.random.random_integers(1, 47)
                else:
                    choice = np.random.random_integers(1, 51)
            # not on straight section of filament
            else:
                str_section = False
                if pos_in_mat == matrix_sub.shape[0] - 2:
                    choice = np.random.random_integers(1, 47)
                else:
                    choice = np.random.random_integers(0, 47)
        else:
            # always do one of the 47
            choice = 1
        # stre transformation
        if choice == 0:
            matrix_sub = energyCalc.diagonal_move(matrix_sub, matrix_sub[pos_in_mat - 1, :], center,
                                                  matrix_sub[pos_in_mat + 1, :], pos_in_mat)
        # perform one of the 47 tranformations
        elif 1 <= choice <= 47:
            mat_to_rot = matrix_sub[pos_in_mat:matrix_sub.shape[0], :]
            while True:
                options = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                for ia in range(0, 3):
                    posi = np.random.random_integers(0, len(options) - 1)
                    mult_mat[ia] = options[posi]
                    del options[posi]
                    neg = np.random.random_integers(0, 1)
                    if neg == 0:
                        mult_mat[ia] = mult_mat[ia] * -1.0
                mult_mat = np.rint(mult_mat).astype(int)
                if np.all(mult_mat == comp_mat_x):
                    continue
                mutation = np.dot((mat_to_rot - center), mult_mat) + center
                matrix_sub[pos_in_mat:matrix_sub.shape[0]] = np.rint(mutation)
                break
        # 0 shrinking
        elif choice == 48:
            # shrinking
            matrix_sub = energyCalc.reverse_diagonal_move(matrix_sub, center, matrix_sub[pos_in_mat + 1, :], pos_in_mat,
                                                          0)
        elif choice == 49:
            # stretching
            matrix_sub = energyCalc.reverse_diagonal_move(matrix_sub, center, matrix_sub[pos_in_mat + 1, :], pos_in_mat,
                                                          1)
        elif choice == 50:
            # stretching
            matrix_sub = energyCalc.reverse_diagonal_move(matrix_sub, center, matrix_sub[pos_in_mat + 1, :], pos_in_mat,
                                                          2)
        elif choice == 51:
            # stretching
            matrix_sub = energyCalc.reverse_diagonal_move(matrix_sub, center, matrix_sub[pos_in_mat + 1, :], pos_in_mat,
                                                          3)
        # check for self-avoidance
        check = comp.compare_matrix_to_matrix(matrix_sub[pos_in_mat + 1:matrix_sub.shape[0], :],
                                              matrix_sub[0:pos_in_mat + 1, :])
        if check:
            mean_energy = mean_energy + current_energy
        else:
            multiplier = 1
            # on straight section again
            if has_new_trans:
                if (energyCalc.logic_check(matrix_sub[pos_in_mat - 1, :], center,
                                           matrix_sub[pos_in_mat + 1, :]) == True):
                    if str_section:
                        multiplier = 1.
                    else:
                        multiplier = 48. / 51.
                else:
                    if str_section:
                        multiplier = 51. / 48.
                    else:
                        multiplier = 1.

            current_energy_sub = energyCalc.energy_in_chain_o_pc(matrix_sub)
            if not try_new:
                try:
                    acceptance_probability = exp(-(current_energy_sub - current_energy) * B) * multiplier
                except:
                    if (current_energy_sub > current_energy) and B < 0.0:
                        acceptance_probability = 1.0
                    elif (current_energy_sub < current_energy) and B < 0.0:
                        acceptance_probability = 0.0
                    elif (current_energy_sub < current_energy) and B > 0.0:
                        acceptance_probability = 1.0
                    elif (current_energy_sub > current_energy) and B > 0.0:
                        acceptance_probability = 0.0
            else:
                try:
                    acceptance_probability = (1 / (1 + exp((current_energy_sub - current_energy) * B))) * (
                        np.sqrt(multiplier))
                except:
                    if (current_energy_sub > current_energy) and B > 0.0:
                        acceptance_probability = 0.0
                    elif (current_energy_sub < current_energy) and B < 0.0:
                        acceptance_probability = 0.0
                    elif (current_energy_sub < current_energy) and B > 0.0:
                        acceptance_probability = 1.0
                    elif (current_energy_sub > current_energy) and B < 0.0:
                        acceptance_probability = 1.0
            random_probability = np.random.random_sample()  # [0.0,1.0)
            if acceptance_probability > random_probability:
                current_energy = current_energy_sub
                matrix = matrix_sub
            mean_energy = mean_energy + current_energy

        mu_one = mu_one + (energyCalc.calculate_mu_one_edit(matrix[0], matrix[-1])) / n
        if n > 100000:
            mu_two = mu_two + energyCalc.calculate_mu_two_with_rn((matrix[-1] - matrix[0]), matrix)
        count = count + 1

    # mu_one = log(mu_one*(1.0/n))/log(N)
    mean_energy = mean_energy / count
    if n > 100000:
        if mu_two > 0:
            mu_two = log(np.sqrt(mu_one)) / log(mu_two / n)
        else:
            mu_two = -2.0
        mu_one = log(np.sqrt(mu_one)) / log(N - 1)
    else:
        mu_one = log(np.sqrt(mu_one)) / log(N - 1)
        mu_two = -1.0
    return [mean_energy, matrix, mu_one, mu_two]


def limited_bends(B, N, n, mat_x, has_matrix, h, seed, try_new, has_new_trans):
    if seed != -1.0:
        np.random.seed(seed)
    else:
        np.random.seed()
    rotxp = np.zeros(shape=[3, 3])
    rotxn = rotxp.copy()
    rotyp = rotxp.copy()
    rotyn = rotxp.copy()
    rotzp = rotxp.copy()
    rotzn = rotxp.copy()
    rotxp1 = rotxp.copy()
    rotyp1 = rotxp.copy()
    rotzp1 = rotxp.copy()
    rotxp[0] = [1, 0, 0]
    rotxp[1] = [0, 0, 1]
    rotxp[2] = [0, -1, 0]
    rotxn[0] = [1, 0, 0]
    rotxn[1] = [0, 0, -1]
    rotxn[2] = [0, 1, 0]
    rotyp[0] = [0, 0, -1]
    rotyp[1] = [0, 1, 0]
    rotyp[2] = [1, 0, 0]
    rotyn[0] = [0, 0, 1]
    rotyn[1] = [0, 1, 0]
    rotyn[2] = [-1, 0, 0]
    rotzp[0] = [0, 1, 0]
    rotzp[1] = [-1, 0, 0]
    rotzp[2] = [0, 0, 1]
    rotzn[0] = [0, -1, 0]
    rotzn[1] = [1, 0, 0]
    rotzn[2] = [0, 0, 1]
    rotxp1[0] = [1, 0, 0]
    rotxp1[1] = [0, -1, 0]
    rotxp1[2] = [0, 0, -1]
    rotyp1[0] = [-1, 0, 0]
    rotyp1[1] = [0, 1, 0]
    rotyp1[2] = [0, 0, -1]
    rotzp1[0] = [-1, 0, 0]
    rotzp1[1] = [0, -1, 0]
    rotzp1[2] = [0, 0, 1]
    options = [rotxp, rotxn, rotyp, rotyn, rotzp, rotzn, rotxp1, rotyp1, rotzp1]
    all_mats = rotxp.copy()
    for di in range(1, len(options)):
        all_mats = np.concatenate((all_mats, options[di]), axis=0)
    # plus one is for diagonal move
    to_delete = np.zeros(shape=(1, len(options) + 7))
    for ddi in range(0, len(options) + 1):
        to_delete[0][ddi] = ddi
    if not has_matrix:
        matrix = np.zeros(shape=(N, 3))
        for iy in range(0, matrix.shape[0]):
            matrix[iy, 0] = iy * h
            matrix[iy, 1] = 0
            matrix[iy, 2] = 0
    elif has_matrix:
        matrix = mat_x
    energy_matrix = np.zeros(shape=(N - 1, N - 1))
    energy_map = energyCalc.initial_energy_map(matrix, energy_matrix)
    current_emap = energy_map
    # current_energy = 0.0
    mean_energy = 0.0
    mu_one = 0.0
    mu_two = 0.0
    current_energy = energyCalc.sum_energy_map(current_emap) / (4.0 * np.pi)
    mean_energy = mean_energy + current_energy
    mu_two = mu_two + energyCalc.calculate_mu_two(matrix)
    mu_one = mu_one + energyCalc.calculate_mu_one(matrix[0], matrix[-1]) / n
    if has_new_trans:
        options_list = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]
    else:
        options_list = [0, 1, 2, 3, 4, 5, 6, 8]
    for i in range(1, n):
        matrix_sub = matrix.copy()
        pivot_point = np.random.random_integers(1, matrix_sub.shape[0] - 2)
        random_option = np.random.random_integers(0, len(options_list) - 1)
        list_object = options_list[random_option]
        if list_object < 9:
            p = matrix_sub[pivot_point, :]
            pivot_mut_mat = options[list_object]
            mat_to_rot = matrix_sub[pivot_point:matrix_sub.shape[0], :]
            mat_to_keep = matrix_sub[0:pivot_point, :]
            mut_mat_end = np.dot((mat_to_rot - p), pivot_mut_mat) + p
            matrix_sub[pivot_point:matrix_sub.shape[0]] = mut_mat_end
            check = comp.compare_matrix_to_matrix(mat_to_rot, mat_to_keep)
        elif list_object == 9:
            p = matrix_sub[pivot_point, :]
            pb = matrix_sub[pivot_point - 1, :]
            pa = matrix_sub[pivot_point + 1, :]
            matrix_sub = energyCalc.diagonal_move(matrix_sub, pb, p, pa, pivot_point)
            check = comp.compare_matrix_to_matrix(matrix_sub[pivot_point + 1:matrix_sub.shape[0], :],
                                                  matrix_sub[0:pivot_point + 1, :])
        elif list_object == 10:
            p = matrix_sub[pivot_point, :]
            pa = matrix_sub[pivot_point + 1, :]
            p_rand = np.random.random_integers(0, 3)
            matrix_sub = energyCalc.reverse_diagonal_move(matrix_sub, p, pa, pivot_point, p_rand)
            check = comp.compare_matrix_to_matrix(matrix_sub[pivot_point + 1:matrix_sub.shape[0], :],
                                                  matrix_sub[0:pivot_point + 1, :])
        if check:
            mean_energy = mean_energy + current_energy
        else:
            current_emap_sub = current_emap.copy()
            current_emap_sub = energyCalc.energy_in_chain_with_e_mat(matrix_sub, current_emap_sub, pivot_point)
            current_energy_sub = energyCalc.sum_energy_map(current_emap_sub) / (4.0 * np.pi)
            # current_energy_sub = energyCalc.energyInChainOPc(matrix.copy())
            if not try_new:
                try:
                    acceptance_probability = exp(-(current_energy_sub - current_energy) * B)
                except:
                    if (current_energy_sub > current_energy) and B < 0.0:
                        acceptance_probability = 1.0
                    elif (current_energy_sub < current_energy) and B < 0.0:
                        acceptance_probability = 0.0
                    elif (current_energy_sub < current_energy) and B > 0.0:
                        acceptance_probability = 1.0
                    elif (current_energy_sub > current_energy) and B > 0.0:
                        acceptance_probability = 0.0
            else:
                try:
                    acceptance_probability = 1 / (1 + exp((current_energy_sub - current_energy) * B))
                except:
                    if (current_energy_sub > current_energy) and B > 0.0:
                        acceptance_probability = 0.0
                    elif (current_energy_sub < current_energy) and B < 0.0:
                        acceptance_probability = 0.0
                    elif (current_energy_sub < current_energy) and B > 0.0:
                        acceptance_probability = 1.0
                    elif (current_energy_sub > current_energy) and B < 0.0:
                        acceptance_probability = 1.0
            random_probability = np.random.random_sample()  # [0.0,1.0)
            if acceptance_probability > random_probability:
                # print 'good'
                current_energy = current_energy_sub
                matrix = matrix_sub
                current_emap = current_emap_sub
            mean_energy = mean_energy + current_energy
        mu_one = mu_one + energyCalc.calculate_mu_one(matrix[0], matrix[-1]) / n
        # if (B==-1.0 or B==0.0 or B==0.4):
        if n > 100000:
            mu_two = mu_two + energyCalc.calculate_mu_two_with_rn((matrix[-1] - matrix[0]), matrix)
    # mu_one = log(mu_one*(1.0/n))/log(N)
    mean_energy = mean_energy / n
    if n > 100000:
        if mu_two > 0:
            mu_two = log(mu_one) / log(mu_two / n)
        else:
            mu_two = -2.0
        mu_one = log(mu_one) / log(N)
    else:
        mu_one = log(mu_one) / log(N)
        mu_two = -1.0
    return [mean_energy, matrix, mu_one, mu_two]


def all_bends_alg_all_graphs(B, N, n, mat_x, has_matrix, h, try_new, has_new_trans, seed, inter, count, location):
    if seed != -1.0:
        np.random.seed(seed)
    else:
        np.random.seed()

    if not has_matrix:
        matrix = np.zeros(shape=(N, 3))
        for iy in range(0, matrix.shape[0]):
            matrix[iy, 0] = iy * h
            matrix[iy, 1] = 0
            matrix[iy, 2] = 0

    elif has_matrix:
        matrix = mat_x

    # current_energy = 0.0
    mean_energy = 0.0
    mean_energy_final = 0.0
    mu_one = 0.0
    mu_two = 0.0
    # mu_one = 0.0

    comp_mat_x = np.zeros(shape=[3, 3])
    comp_mat_x[0] = [1, 0, 0]
    comp_mat_x[1] = [0, 1, 0]
    comp_mat_x[2] = [0, 0, 1]

    # current_energy = energyCalc.energyInChainOPc(matrix.copy())
    current_energy = energyCalc.energy_in_chain(matrix)
    mean_energy = mean_energy + current_energy
    mu_two = mu_two + energyCalc.calculate_mu_two(matrix)
    xdist = (matrix[0][0] - matrix[-1][0]) ** 2.0
    ydist = (matrix[0][1] - matrix[-1][1]) ** 2.0
    zdist = (matrix[0][2] - matrix[-1][2]) ** 2.0
    dista = abs((xdist + ydist + zdist) ** (1.0 / 2.0))
    mu_one = mu_one + dista / n

    dn_vals = []
    mu_vals = []
    e_vals = []
    dnn_vals = []
    mun_vals = []

    # lims = 150
    test = 100000
    di = 0
    '''
    mpl.rcParams['legend.fontsize'] = 21
    fig = plt.figure(figsize=(16, 8), dpi=80)
    #fig1 = plt.figure(figsize=(11,8),dpi=80)
    myTitle = 'B=%1.3f, N=%d, n=%d' % (B,N-1,n)
    #fig.suptitle(myTitle, fontsize=21, fontweight='bold')
    '''
    for i in range(1, n):
        matrix_sub = matrix.copy()
        mult_mat = np.zeros(shape=[3, 3])
        pos_in_mat = np.random.random_integers(1, matrix_sub.shape[0] - 2)
        center = matrix_sub[pos_in_mat, :].copy()

        str_section = False

        if has_new_trans:
            # check if on straigh section of filament
            if energyCalc.logic_check(matrix_sub[pos_in_mat - 1, :], center, matrix_sub[pos_in_mat + 1, :]):
                str_section = True
                if pos_in_mat == matrix_sub.shape[0] - 2:
                    choice = np.random.random_integers(1, 47)
                else:
                    choice = np.random.random_integers(1, 51)

            # not on straight section of filament
            else:
                str_section = False
                if pos_in_mat == matrix_sub.shape[0] - 2:
                    choice = np.random.random_integers(1, 47)
                else:
                    choice = np.random.random_integers(0, 47)

        else:
            # always do one of the 47
            choice = 1

        # stre transformation
        if choice == 0:
            matrix_sub = energyCalc.diagonal_move(matrix_sub, matrix_sub[pos_in_mat - 1, :], center,
                                                  matrix_sub[pos_in_mat + 1, :], pos_in_mat)

        # perform one of the 47 tranformations
        elif 1 <= choice <= 47:
            mat_to_rot = matrix_sub[pos_in_mat:matrix_sub.shape[0], :]
            while True:
                options = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                for ia in range(0, 3):
                    posi = np.random.random_integers(0, len(options) - 1)
                    mult_mat[ia] = options[posi]
                    del options[posi]
                    neg = np.random.random_integers(0, 1)
                    if neg == 0:
                        mult_mat[ia] = mult_mat[ia] * -1.0
                mult_mat = np.rint(mult_mat).astype(int)
                if np.all(mult_mat == comp_mat_x):
                    continue

                mutation = np.dot((mat_to_rot - center), mult_mat) + center

                matrix_sub[pos_in_mat:matrix_sub.shape[0]] = np.rint(mutation)
                # check = comp.compareMatrixToMatrix(mat_to_rot,matToKeep)
                break

        # 0 shrinking
        elif choice == 48:
            # shrinking
            matrix_sub = energyCalc.reverse_diagonal_move(matrix_sub, center, matrix_sub[pos_in_mat + 1, :], pos_in_mat,
                                                          0)

        elif choice == 49:
            # stretching
            matrix_sub = energyCalc.reverse_diagonal_move(matrix_sub, center, matrix_sub[pos_in_mat + 1, :], pos_in_mat,
                                                          1)

        elif choice == 50:
            # stretching
            matrix_sub = energyCalc.reverse_diagonal_move(matrix_sub, center, matrix_sub[pos_in_mat + 1, :], pos_in_mat,
                                                          2)

        elif choice == 51:
            # stretching
            matrix_sub = energyCalc.reverse_diagonal_move(matrix_sub, center, matrix_sub[pos_in_mat + 1, :], pos_in_mat,
                                                          3)

        # check for self-avoidance
        check = comp.compare_matrix_to_matrix(matrix_sub[pos_in_mat + 1:matrix_sub.shape[0], :],
                                              matrix_sub[0:pos_in_mat + 1, :])
        if check:
            mean_energy = mean_energy + current_energy
            if i >= test:
                mean_energy_final = mean_energy_final + current_energy
                di = di + 1
        else:
            multiplier = 1
            # on straight section again
            if has_new_trans:
                if energyCalc.logic_check(matrix_sub[pos_in_mat - 1, :], center, matrix_sub[pos_in_mat + 1, :]):
                    if str_section:
                        multiplier = 1.
                    else:
                        multiplier = 48. / 51.
                else:
                    if str_section:
                        multiplier = 51. / 48.
                    else:
                        multiplier = 1.

            current_energy_sub = energyCalc.energy_in_chain_o_pc(matrix_sub)
            if not try_new:
                try:
                    acceptance_probability = exp(-(current_energy_sub - current_energy) * B) * multiplier
                except:
                    if (current_energy_sub > current_energy) and B < 0.0:
                        acceptance_probability = 1.0
                    elif (current_energy_sub < current_energy) and B < 0.0:
                        acceptance_probability = 0.0
                    elif (current_energy_sub < current_energy) and B > 0.0:
                        acceptance_probability = 1.0
                    elif (current_energy_sub > current_energy) and B > 0.0:
                        acceptance_probability = 0.0
            else:
                try:
                    acceptance_probability = (1 / (1 + exp((current_energy_sub - current_energy) * B))) * (
                        np.sqrt(multiplier))
                except:
                    if (current_energy_sub > current_energy) and B > 0.0:
                        acceptance_probability = 0.0
                    elif (current_energy_sub < current_energy) and B < 0.0:
                        acceptance_probability = 0.0
                    elif (current_energy_sub < current_energy) and B > 0.0:
                        acceptance_probability = 1.0
                    elif (current_energy_sub > current_energy) and B < 0.0:
                        acceptance_probability = 1.0
            random_probability = np.random.random_sample()  # [0.0,1.0)
            if acceptance_probability > random_probability:
                # print 'good'
                current_energy = current_energy_sub
                matrix = matrix_sub
            mean_energy = mean_energy + current_energy
            if i >= test:
                mean_energy_final = mean_energy_final + current_energy
                di = di + 1

        if (i - 1) % inter == 0:
            dn_vals.append(i + 1)
            mu_vals.append(mean_energy / (i + 1))
            e_vals.append(current_energy)
            if i >= test:
                dnn_vals.append(i + 1)
                mun_vals.append(mean_energy_final / (di))

        xdist = (matrix[0][0] - matrix[-1][0]) ** 2.0
        ydist = (matrix[0][1] - matrix[-1][1]) ** 2.0
        zdist = (matrix[0][2] - matrix[-1][2]) ** 2.0
        dista = abs((xdist + ydist + zdist) ** (1.0 / 2.0))
        mu_one = mu_one + dista / n
        if n > 10000:
            mu_two = mu_two + energyCalc.calculate_mu_two_edit(matrix)

    # mu_one = log(mu_one*(1.0/n))/log(N)
    mean_energy = mean_energy / n
    if n > 100000:
        # and (B==-1.0 or B==0.0 or B==0.4)
        print('--------')
        print('-N= ', N)
        print('-n= ', n)
        print('-B= ', B)
        print('-type= 1')
        print('-mu_two(sum): ', mu_two)
        print('-mu_one/n: ', mu_one)

        #     if mu_two > 0:
        #         mu_two = log(mu_one) / log(mu_two / n)
        #     else:
        #         mu_two = -2.0
        #     mu_one = log(mu_one) / log(N)
        # else:
        #     mu_one = log(mu_one) / log(N)
        #     mu_two = -1.0
        #     # return mean_energy
        '''
    ax2 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2.set_aspect('equal')
    ax2.set_zlim3d(-lims, lims)
    ax2.set_xlim3d(-lims, lims)
    ax2.set_ylim3d(-lims, lims)
    ax2.set_title('Final Filament')
    
    ax2.plot(matrix[:,0],matrix[:,1],matrix[:,2], 'ro-' ,label='filament')
    
    a = np.zeros(shape=(N,3))
    for iy in xrange(0,a.shape[0]):
        a[iy,0] = iy
        a[iy,1] = 0
        a[iy,2] = 0
    energy = energyCalc.energyInChainOPc(a)
            
    ae = fig.add_subplot(1,2,2)
    ae.set_xlabel('Monte Carlo Step',fontsize=17)
    ae.set_ylabel('E',fontsize=17)
    ae.set_ylim([-1.0,energy+1.0])
    ae.set_xlim([0,n+100])
    ae.plot(dn_vals,mu_vals, 'o-' ,label=r'$\langle{E}\rangle$',zorder=2)
    ae.plot(dnn_vals,mun_vals, 'o-' ,label=r'$\langle{E_{a}}\rangle$',zorder=3)
    ae.plot(dn_vals,e_vals,'o-',label='$E$',zorder=1)    
    #ae.tick_params(axis='both', which='major', labelsize=21)
    #ae.tick_params(axis='both', which='minor', labelsize=17)
    
    ae.axhline(y=energy,linestyle='--',label='maxE')
    ae.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ae.grid()
    
    #comment out this and uncomment plt.show() to display graph
    
    #location = '%s/config%df.png' % (location,count)
    #fig.savefig(location, dpi=80, format='png', bbox_inches='tight')    # save the figure to file
    #plt.close(fig)
    
    plt.show()
    '''

    return [dn_vals, mu_vals, dnn_vals, mun_vals, dn_vals, e_vals]


def graph_old_new_comparison(b, N, n):
    # matrix = allBendsAlgAll(0.0,N,150000,-1,False,1.0,False,False,1)[1]
    print('done')
    chorin = all_bends_alg_all_graphs(b, N, n, -1, False, 1.0, False, False, 1, 2000, 1, '')
    # dn_vals = chorin[0]
    mu_vals = chorin[1]
    dnn_vals = chorin[2]
    mun_vals = chorin[3]
    dn_vals = chorin[4]
    e_vals = chorin[5]

    mpl.rcParams['legend.fontsize'] = 21
    fig = plt.figure(figsize=(16, 8), dpi=80)
    # myTitle = 'b=%1.3f, N=%d, n=%d' % (b,N-1,n)

    a = np.zeros(shape=(N, 3))
    for iy in range(0, a.shape[0]):
        a[iy, 0] = iy
        a[iy, 1] = 0
        a[iy, 2] = 0
    energy = energyCalc.energy_in_chain_o_pc(a)

    ae = fig.add_subplot(111)
    ae.set_xlabel('Monte Carlo Step', fontsize=37)
    ae.set_ylabel(r"$E_i$", fontsize=37)
    ae.set_ylim([-1.0, energy + 1.0])
    ae.set_xlim([0, n + 100])
    ae.tick_params(axis='both', which='major', labelsize=31)
    ae.tick_params(axis='both', which='minor', labelsize=27)
    ae.plot(dn_vals, mu_vals, 'o-', label=r'$\langle{E}\rangle$', zorder=2)
    ae.plot(dnn_vals, mun_vals, 'o-', label=r'$\langle{E_{a}}\rangle$', zorder=3)
    ae.plot(dn_vals, e_vals, 'o-', label='$E$', zorder=1)

    ae.axhline(y=energy, linestyle='--', label='maxE')
    # ae.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ae.grid()
    plt.show()


def limited_bends_energy(B, N, n, dn, inter, count, config, seed, try_new, location):
    global acceptance_probability
    if seed != -1.0:
        np.random.seed(seed)
    else:
        np.random.seed()
    rotxp = np.zeros(shape=[3, 3])
    rotxn = rotxp.copy()
    rotyp = rotxp.copy()
    rotyn = rotxp.copy()
    rotzp = rotxp.copy()
    rotzn = rotxp.copy()
    rotxp1 = rotxp.copy()
    rotyp1 = rotxp.copy()
    rotzp1 = rotxp.copy()
    rotxp[0] = [1, 0, 0]
    rotxp[1] = [0, 0, 1]
    rotxp[2] = [0, -1, 0]
    rotxn[0] = [1, 0, 0]
    rotxn[1] = [0, 0, -1]
    rotxn[2] = [0, 1, 0]
    rotyp[0] = [0, 0, -1]
    rotyp[1] = [0, 1, 0]
    rotyp[2] = [1, 0, 0]
    rotyn[0] = [0, 0, 1]
    rotyn[1] = [0, 1, 0]
    rotyn[2] = [-1, 0, 0]
    rotzp[0] = [0, 1, 0]
    rotzp[1] = [-1, 0, 0]
    rotzp[2] = [0, 0, 1]
    rotzn[0] = [0, -1, 0]
    rotzn[1] = [1, 0, 0]
    rotzn[2] = [0, 0, 1]
    rotxp1[0] = [1, 0, 0]
    rotxp1[1] = [0, -1, 0]
    rotxp1[2] = [0, 0, -1]
    rotyp1[0] = [-1, 0, 0]
    rotyp1[1] = [0, 1, 0]
    rotyp1[2] = [0, 0, -1]
    rotzp1[0] = [-1, 0, 0]
    rotzp1[1] = [0, -1, 0]
    rotzp1[2] = [0, 0, 1]
    options = [rotxp, rotxn, rotyp, rotyn, rotzp, rotzn, rotxp1, rotyp1, rotzp1]
    all_mats = rotxp.copy()
    for di in range(1, len(options)):
        all_mats = np.concatenate((all_mats, options[di]), axis=0)
    # plus on is for diagonal move
    to_delete = np.zeros(shape=(1, 16))
    for ddi in range(0, 16):
        to_delete[0][ddi] = ddi
    matrix = config
    dn_vals = []
    mu_vals = []
    e_vals = []
    rot_pick_cs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    rot_rej_cs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    rot_accept_cs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    rot_energy_cs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    rot_energyn_cs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    energy_matrix = np.zeros(shape=(N - 1, N - 1))
    energy_map = energyCalc.initial_energy_map(matrix, energy_matrix)
    current_emap = energy_map
    # current_energy = 0.0
    mean_energy = 0.0
    counter = 0
    current_energy = energyCalc.sum_energy_map(current_emap) / (4.0 * np.pi)
    mean_energy = mean_energy + current_energy
    # current_energy = energyCalc.energyInChainOPc(matrix.copy())
    # mean_energy = mean_energy + current_energy
    lims = 155
    lims_2 = 50

    mpl.rcParams['legend.fontsize'] = 12
    fig = plt.figure(figsize=(11, 8), dpi=80)
    fig1 = plt.figure(figsize=(11, 8), dpi=80)
    my_title = 'B=%1.3f, N=%d, dn=%d, n=%d' % (B, N, dn, n)
    fig.suptitle(my_title, fontsize=14, fontweight='bold')

    # fig.subplots_adjust(hspace=.75)
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.set_aspect('equal')
    ax.set_zlim3d(-lims_2, lims_2)
    ax.set_xlim3d(-lims_2, lims_2)
    ax.set_ylim3d(-lims_2, lims_2)
    ax.set_title('Initial Filament')
    ax.plot(matrix[:, 0], matrix[:, 1], matrix[:, 2], 'ro-', label='Vortex Filament')

    options_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i in range(1, n):
        matrix_sub = matrix.copy()
        current_emap_sub = current_emap.copy()
        pivot_point = np.random.random_integers(1, matrix_sub.shape[0] - 2)
        p = matrix_sub[pivot_point, :]
        pb = matrix_sub[pivot_point - 1, :]
        pa = matrix_sub[pivot_point + 1, :]
        random_option = np.random.random_integers(0, len(options_list) - 1)
        # print options_list
        list_object = options_list[random_option]
        rot_pick_cs[list_object] = rot_pick_cs[list_object] + 1
        if list_object < 9:
            pivot_mut_mat = options[list_object]
            mat_to_rot = matrix_sub[pivot_point:matrix_sub.shape[0], :]
            mat_to_keep = matrix_sub[0:pivot_point, :]
            mut_mat_end = np.dot((mat_to_rot - p), pivot_mut_mat) + p
            matrix_sub[pivot_point:matrix_sub.shape[0]] = mut_mat_end
            check = comp.compare_matrix_to_matrix(mat_to_rot, mat_to_keep)
        elif list_object == 9:
            counter = counter + 1
            matrix_sub = energyCalc.diagonal_move(matrix_sub, pb, p, pa, pivot_point)
            check = comp.compare_matrix_to_matrix(matrix_sub[pivot_point + 1:matrix_sub.shape[0], :],
                                                  matrix_sub[0:pivot_point + 1, :])
        elif list_object == 10:
            p_rand = np.random.random_integers(0, 3)
            matrix_sub = energyCalc.reverse_diagonal_move(matrix_sub, p, pa, pivot_point, p_rand)
            check = comp.compare_matrix_to_matrix(matrix_sub[pivot_point + 1:matrix_sub.shape[0], :],
                                                  matrix_sub[0:pivot_point + 1, :])
        if check:
            rot_rej_cs[list_object] = rot_rej_cs[list_object] + 1
            mean_energy = mean_energy + current_energy
        else:
            rot_accept_cs[list_object] = rot_accept_cs[list_object] + 1
            current_emap_sub = energyCalc.energy_in_chain_with_e_mat(matrix_sub, current_emap_sub, pivot_point)
            current_energy_sub = energyCalc.sum_energy_map(current_emap_sub) / (4.0 * np.pi)
            # current_energy_sub = energyCalc.energyInChainOPc(matrix.copy())
            if not try_new:
                try:
                    acceptance_probability = exp(-(current_energy_sub - current_energy) * B)
                except:
                    if (current_energy_sub > current_energy) and B < 0.0:
                        acceptance_probability = 1.0
                    elif (current_energy_sub < current_energy) and B < 0.0:
                        acceptance_probability = 0.0
                    elif (current_energy_sub < current_energy) and B > 0.0:
                        acceptance_probability = 1.0
                    elif (current_energy_sub > current_energy) and B > 0.0:
                        acceptance_probability = 0.0
            else:
                try:
                    acceptance_probability = 1 / (1 + exp((current_energy_sub - current_energy) * B))
                except:
                    if (current_energy_sub > current_energy) and B > 0.0:
                        acceptance_probability = 0.0
                    elif (current_energy_sub < current_energy) and B < 0.0:
                        acceptance_probability = 0.0
                    elif (current_energy_sub < current_energy) and B > 0.0:
                        acceptance_probability = 1.0
                    elif (current_energy_sub > current_energy) and B < 0.0:
                        acceptance_probability = 1.0
            random_probability = np.random.random_sample()  # [0.0,1.0)
            if acceptance_probability > random_probability:
                # print 'good'
                rot_energy_cs[list_object] = rot_energy_cs[list_object] + 1
                current_energy = current_energy_sub
                matrix = matrix_sub
                current_emap = current_emap_sub
            else:
                rot_energyn_cs[list_object] = rot_energyn_cs[list_object] + 1
            mean_energy = mean_energy + current_energy
        if (i - 1) % inter == 0:
            # data.append([N,B,temp,n,i,mean_energy/(i+1),window_average/(i),current_energy])
            dn_vals.append(i + 1)
            mu_vals.append(mean_energy / (i + 1))
            e_vals.append(current_energy)
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_aspect('equal')
    ax2.set_zlim3d(-lims, lims)
    ax2.set_xlim3d(-lims, lims)
    ax2.set_ylim3d(-lims, lims)
    ax2.set_title('Final Filament')

    ae = fig.add_subplot(2, 2, 3)
    ae.set_title('Energy')
    ae.set_xlabel('Monte Carlo Step')
    ae.set_ylabel('E')
    ae.set_ylim([-5.0, 130.0])

    ae = fig1.add_subplot(1, 1, 1)
    ae.set_xlabel('Monte Carlo Step')
    ae.set_ylabel('E')
    ae.set_ylim([-5.0, 130.0])
    ae.grid()

    ae2 = fig.add_subplot(2, 2, 4)
    ae2.set_title('Transform Stats')

    # plot out data to the screen
    ax2.plot(matrix[:, 0], matrix[:, 1], matrix[:, 2], 'ro-', label='Vortex Filament')
    # ae.plot(dn_vals,mu_vals, 'o-' ,label='1/x')
    # ae.plot(dn_vals,e_vals,'o-')

    n_groups = 11
    index = np.arange(n_groups)
    # bar_width1 = 0.20
    #
    # opacity = 0.4
    # error_config = {'ecolor': '0.3'}

    # rects2 = ae2.bar(index, rot_pick_cs, bar_width1,
    #                  alpha=opacity,
    #                  color='r',
    #                  error_kw=error_config,
    #                  label='Picked')
    #
    # rects3 = ae2.bar(index + bar_width1, rot_accept_cs, bar_width1,
    #                  alpha=opacity,
    #                  color='g',
    #                  error_kw=error_config,
    #                  label='Accepted')
    #
    # rects4 = ae2.bar(index + bar_width1 * 2, rot_energy_cs, bar_width1,
    #                  alpha=opacity,
    #                  color='b',
    #                  error_kw=error_config,
    #                  label='Energy')
    #
    # rects4 = ae2.bar(index + bar_width1 * 3, rot_energyn_cs, bar_width1,
    #                  alpha=opacity,
    #                  color='m',
    #                  error_kw=error_config,
    #                  label='Energy neg')
    #
    # rects4 = ae2.bar(index + bar_width1 * 4, rot_rej_cs, bar_width1,
    #                  alpha=opacity,
    #                  color='y',
    #                  error_kw=error_config,
    #                  label='Rejected')

    ae2.set_xlabel('Rotation Index')
    ae2.set_ylabel('Occurrence')
    ae2.set_title('Number of Occurrences for Rotation Indices')
    ae2.set_xticks(index + 1.0 / 2.0)
    ae2.set_xticklabels(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
    # ae2.legend()

    ae.plot(dn_vals, mu_vals, 'o-', label=r'$\langle{E}\rangle$')
    ae.plot(dn_vals, e_vals, 'o-', label='$E$')
    ae.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

    # ax.legend()

    # comment out this and uncomment plt.show() to display graph
    location = '%s/config%df.png' % (location, count)
    fig.savefig(location)  # save the figure to file
    plt.close(fig)
    # plt.show()

    return [mean_energy / n, matrix]
