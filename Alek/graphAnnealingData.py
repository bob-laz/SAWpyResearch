#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 20:59:18 2016

@author: Aleksandr Lukanen
"""
import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from Alek import energyCalc as ec


# def graph_mu1(fileName, graphName):
#     print('graphing mu1')


def save_to_separate_files(file_name, directory):
    data = np.genfromtxt(file_name, skip_header=1,
                         names=['N', 'B', 'dn', 'n', 'E', 'type', 'w', 'h', 'hasNewTrans', 'mu1', 'mu2', 'ddn'],
                         dtype=[('N', np.int32), ('B', np.float32), ('dn', np.int32), ('n', np.int32),
                                ('E', np.float64), ('type', np.int32), ('w', np.bool), ('h', np.float64),
                                ('hasNewTrans', np.bool), ('mu1', np.float64), ('mu2', np.float64), ('ddn', np.int32)]
                         , delimiter=",")

    params = np.unique(data[["N", "dn", "n", "ddn", "w", "hasNewTrans"]])
    # print params
    params = np.sort(params[["hasNewTrans", "w", "N", "dn", "n", "ddn"]])
    print('***Graphing Software for Vortex Filament Project***')
    print('')
    print('')
    print('****Format of Dataset: ("w","N","dn","n","ddn")')
    print('***w - true for gibbs weight and false for modified weight')
    print('***N - the length of the filament')
    print('***dn - the number of initial steps before averaging')
    print('***n - the number of averaging steps')
    print(
        '***ddn - the number of steps at beta=0 before each simulation. -99 refers to a sim. that was appart of the '
        'simulated annealing code.')
    print('')
    print('********Graphing Data*********')
    count = 0
    for par in params:
        print(count)
        print(par)
        copy_data = data.copy()
        new_dplt = copy_data[
            np.where(np.logical_and(np.logical_and(par["N"] == copy_data["N"], par["n"] == copy_data["n"],
                                                   par['hasNewTrans'] == copy_data['hasNewTrans'])
                                    , np.logical_and(par['w'] == copy_data['w'],
                                                     np.logical_and(par['dn'] == copy_data['dn'],
                                                                    par['ddn'] == copy_data['ddn']))))]
        print(len(new_dplt))
        print(new_dplt)
        name_of_file = '%s/C%d_n%d_w%s_dn%d_ddn%d_N%d.csv' % (
            directory, count, par['n'], par['w'], par['dn'], par['ddn'], par['N'])
        count += 1
        np.savetxt(name_of_file, new_dplt, delimiter=' '
                   , header='"N","B","dn","n","E","type","w","h","hasNewTrans","mu1","mu2","ddn"',
                   fmt="%d %1.4f %d %d %1.4f %d %s %1.2f %s %1.4f %1.4f %d")


def graph_energy(file_name, graph_name):
    data = np.genfromtxt(file_name, skip_header=1,
                         names=['N', 'B', 'dn', 'n', 'E', 'type', 'w', 'h', 'hasNewTrans', 'mu1', 'mu2', 'ddn'],
                         dtype=[('N', np.int32), ('B', np.float32), ('dn', np.int32), ('n', np.int32),
                                ('E', np.float64), ('type', np.int32), ('w', np.bool), ('h', np.float64),
                                ('hasNewTrans', np.bool), ('mu1', np.float64), ('mu2', np.float64), ('ddn', np.int32)],
                         delimiter=",")
    # data = data[np.where(data['hasNewTrans']==True)]
    # print data

    mpl.rcParams['legend.fontsize'] = 12
    fig = plt.figure(figsize=(13, 8), dpi=80)
    ae = fig.add_subplot(111)
    ae.set_title(graph_name)  # or title
    ae.set_xlabel(r"$\beta$", fontsize=47)
    ae.set_ylabel(r"$E$", fontsize=47)
    ae.set_aspect('auto')
    ae.set_xlim(data["B"].max() + 1, data["B"].min() - 1)
    ae.set_ylim(data["E"].min() - abs(data["E"].min()), data["E"].max() + data["E"].max() / 10)
    ae.tick_params(axis='both', which='major', labelsize=31)
    ae.tick_params(axis='both', which='minor', labelsize=27)
    ae.grid()

    # print 'values of N: ', np.unique(data["N"])
    dplt = np.sort(data[["w", "N", "B", "E", "n", "dn", "ddn"]], order=["N", "B", "n", "dn", "ddn"])

    params = np.unique(data[["N", "dn", "n", "ddn", "w", "hasNewTrans"]])
    # print params
    params = np.sort(params[["hasNewTrans", "w", "N", "dn", "n", "ddn"]])
    print('***Graphing Software for Vortex Filament Project***')
    print('')
    print('')
    print('****Format of Dataset: ("w","N","dn","n","ddn")')
    print('***w - true for gibbs weight and false for modified weight')
    print('***N - the length of the filament')
    print('***dn - the number of initial steps before averaging')
    print('***n - the number of averaging steps')
    print(
        '***ddn - the number of steps at beta=0 before each simulation. -99 refers to a sim. that was appart of the '
        'simulated annealing code.')
    print('')
    print('********Graphing Data*********')
    for par in params:
        print(par)
        print('')
        print('******************')
        print("****Dataset: ", par)
        user_check = 'n'
        if True:
            user_check = 'y'
            # userCheck = raw_input('???Should this be graphed(y/anykey): ')
        if user_check != 'y':
            print('not graphing: ', par)
            print('******************')
            continue
        elif user_check == 'y':
            print('graphing: ', par)
            print('******************')

            data_e = []
            data_b = []
            for i in dplt:
                if (i["N"] == par["N"] and i["w"] == par["w"] and i["n"] == par["n"] and i["dn"] == par["dn"] and i[
                    "ddn"] == par["ddn"]):
                    data_e.append(i["E"])
                    data_b.append(i["B"])
                    # dplt = np.delete(dplt, np.where(i["N"]==dplt["N"]) and np.where(
                    #    i["w"]==dplt["w"]) and np.where(i["n"]==dplt["n"]) and
                    #    np.where(i["dn"]==par["dn"]) and np.where(i["ddn"]==par["ddn"]))
            if par['N'] == 901:
                print(data_b)

            if not par["w"]:
                stri = "N=%d,dn=%d,n=%d,w=Gibbs" % (par["N"] - 1, par["dn"], par["n"])
            else:
                stri = "N=%d,dn=%d,n=%d,w=Modified" % (par["N"] - 1, par["dn"], par["n"])
            a = np.zeros(shape=(par["N"], 3))
            for iy in range(0, a.shape[0]):
                a[iy, 0] = iy
                a[iy, 1] = 0
                a[iy, 2] = 0
            energy = ec.energy_in_chain_o_pc(a)
            ae.axhline(y=energy, linestyle='--')
            if not par["w"]:
                ae.plot(data_b, data_e, 'o-', label=stri)
            else:
                ae.plot(data_b, data_e, 'o--', label=stri)
    '''           
    vals = [3,4,5,6,7,8,9]
    for val in vals:
        e_list = []
        b_list = []
        Bi = -100.0
        step = -0.5
        fileN = "Data/E%d.csv" % val
        with open(fileN, 'rt') as csvfile2:
            reader2 = csv.reader(csvfile2)
            for row2 in reader2:
                aE = float(row2[0])
                aB = Bi
                e_list.append(aE)
                b_list.append(aB)         
                Bi = Bi - step
        stri2 = 'N=%d' % val
        ae.plot(b_list,e_list,'-o',markersize=2.0,label=stri2)
     '''
    # print dplt
    print('******************')
    print('created graph')
    print('******************')
    # plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    # plt.show()


def graph_all_files(directory, title):
    mpl.rcParams['legend.fontsize'] = 12
    fig = plt.figure(figsize=(13, 8), dpi=80)
    ae = fig.add_subplot(111)

    ae.set_title(title)  # or title
    ae.set_xlabel(r"$\beta$", fontsize=47)
    ae.set_ylabel(r"$E$", fontsize=47)
    ae.set_aspect('auto')
    # ae.set_xlim(data["B"].max()+1,data["B"].min()-1)
    # ae.set_ylim(data["E"].min()-abs(data["E"].min())
    #            ,data["E"].max()+data["E"].max()/10)
    ae.tick_params(axis='both', which='major', labelsize=31)
    ae.tick_params(axis='both', which='minor', labelsize=27)
    ae.grid()

    file_names = glob.glob("%s/*.csv" % directory)
    max_e = 0
    for fileName in file_names:
        data_graph = np.genfromtxt(fileName, skip_header=1,
                                   names=['N', 'B', 'dn', 'n', 'E', 'type', 'w', 'h', 'hasNewTrans', 'mu1', 'mu2',
                                          'ddn'],
                                   dtype=[('N', np.int32), ('B', np.float32), ('dn', np.int32), ('n', np.int32),
                                          ('E', np.float64), ('type', np.int32), ('w', np.bool), ('h', np.float64),
                                          ('hasNewTrans', np.bool), ('mu1', np.float64), ('mu2', np.float64),
                                          ('ddn', np.int32)], delimiter=" ")

        data_e = []
        data_b = []
        for i in data_graph:
            if i["E"] > max_e:
                max_e = i['E']
            data_e.append(i["E"])
            data_b.append(i["B"])

        if not data_graph["w"][0]:
            stri = "N=%d,dn=%d,n=%d,w=Gibbs" % (data_graph["N"][0] - 1, data_graph["dn"][0], data_graph["n"][0])
        else:
            stri = "N=%d,dn=%d,n=%d,w=Modified" % (data_graph["N"][0] - 1, data_graph["dn"][0], data_graph["n"][0])
        a = np.zeros(shape=(data_graph["N"][0], 3))
        for iy in range(0, a.shape[0]):
            a[iy, 0] = iy
            a[iy, 1] = 0
            a[iy, 2] = 0
        energy = ec.energy_in_chain_o_pc(a)
        ae.axhline(y=energy, linestyle='--')
        if not data_graph["w"][0]:
            ae.plot(data_b, data_e, 'o-', label=stri)
        else:
            ae.plot(data_b, data_e, 'o--', label=stri)
    ae.set_xlim(data_graph["B"].max() + 1, data_graph["B"].min() - 1)
    ae.set_ylim(-25, max_e + max_e / 10)
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.plot()


if __name__ == '__main__':
    # this program assumes there is more than one
    # row in your csv file.

    # this function takes in a csv file and breaks up the various cases into separate files
    # saveToSeparateFiles("annealingData/mns/energy.csv","dataSets/annealingSets")

    # graph all grouped data in separate plots
    # graphAllFiles( directory here, title here) #example
    # graphAllFiles("dataSets/fromStraight","Start from Straight Filament")
    # graphAllFiles("dataSets/NStep","Start from N-Step Filament")
    # graphAllFiles("dataSets/Equalized","Start from Equalized Filament")
    # graphAllFiles("dataSets/annealingSets","Annealing Method")

    # ENTER THE DIRECTORY OF THE ENERGY FILE HERE
    graph_energy("annealingData/mns/energy.csv", "Annealing Results with Step of B=0.5")
    # data = graphEnergy("Datac/smallEnergyChorin.csv","")
    # data = graphEnergy("Datac/allBendsNewTransEqualSpace.csv","")
