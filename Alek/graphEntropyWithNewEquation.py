# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 17:30:52 2016

@author: lukanen
"""

# file name: Data/dataForRecursiveEquationEntropy.csv
import csv
import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt


def si_plus_one(si, bi, bi_one, ei_one, ei):
    print(("si: ", si, ", bi: ", bi, ", bi_one: ", bi_one, ", ei_one: ", ei_one, ", ei: ", ei))
    si_one = si + (bi + (bi_one - bi) / 2.0) * (ei_one - ei)
    return si_one


def si_plus_minus_one(si, bi, bi_one, ei_one, ei):
    # print("si: ", si, ", bi: ", bi, ", bi_one: ", bi_one , ", ei_one: ", ei_one , ", ei: ", ei)
    si_one = si - (bi - (bi - bi_one) / 2.0) * (ei - ei_one)
    return si_one


def graph_data(title, data, s_zero, N, data_index):
    top = -100
    bottom = 100
    print(N)
    # line_select = -1
    gibbs_data = []
    modified_data = []
    # gibbs data
    for dat in data[0]:
        found = False
        for index in gibbs_data:
            if dat[0] == index[0] and dat[1] == index[1] and dat[2] == index[2]:
                found = True
                break
        if not found:
            gibbs_data.append([dat[0], dat[1], dat[2], []])

    for dat2 in data[1]:
        found2 = False
        for index2 in modified_data:
            if dat2[0] == index2[0] and dat2[1] == index2[1] and dat2[2] == index2[2]:
                found2 = True
                break
        if not found2:
            modified_data.append([dat2[0], dat2[1], dat2[2], []])

    for dat3 in data[0]:
        for index3 in gibbs_data:
            if dat3[0] == index3[0] and dat3[1] == index3[1] and dat3[2] == index3[2]:
                index3[3].append([dat3[3], dat3[4]])
                break

    for dat4 in data[1]:
        for index4 in modified_data:
            if dat4[0] == index4[0] and dat4[1] == index4[1] and dat4[2] == index4[2]:
                index4[3].append([dat4[3], dat4[4]])
                break

    for sortThis in gibbs_data:
        sortThis[3] = sorted(sortThis[3], reverse=True)

    for sortThis2 in modified_data:
        sortThis2[3] = sorted(sortThis2[3], reverse=True)

    entropy_array = [s_zero]
    entropy_index = 0

    for i in range(0, len(gibbs_data[0][3]) - 1):
        # siPlusOne(si,bi,biOne,EiOne,Ei)
        # print i
        if top <= gibbs_data[0][3][i][0] <= 0.0:  # -3.0<
            si = si_plus_one(entropy_array[entropy_index], gibbs_data[data_index][3][i][0],
                             gibbs_data[data_index][3][i + 1][0],
                             gibbs_data[data_index][3][i + 1][1], gibbs_data[data_index][3][i][1])
            entropy_array.append(si)
            entropy_index = entropy_index + 1

    entropy_array2 = [s_zero]
    entropy_index2 = 0
    for i in range(len(gibbs_data[0][3]) - 1, -1, -1):
        # siPlusOne(si,bi,biOne,EiOne,Ei)
        if bottom >= gibbs_data[0][3][i][0] >= 0.0:  # 3.0>
            # print gibbs_data[0][3][i][0]
            if (i - 1) != -1:
                si = si_plus_one(entropy_array2[entropy_index2], gibbs_data[data_index][3][i][0],
                                 gibbs_data[data_index][3][i - 1][0], gibbs_data[data_index][3][i - 1][1],
                                 gibbs_data[data_index][3][i][1])
                entropy_array2.append(si)
                entropy_index2 = entropy_index2 + 1
    print('###################')
    # print entropy_array2

    final_array = []
    for itemy in entropy_array:
        final_array.append(itemy / N)

    final_array2 = []
    for itemy2 in entropy_array2:
        final_array2.append(itemy2 / N)

    return [final_array, final_array2]


if __name__ == '__main__':
    print('creating graphs...')
    # fileN = 'Data/dataForRecursiveEquationEntropy.csv'
    # fileNameForEnergy = 'Datac/allBendsNewTransEqualSpace.csv'
    # fileNameForEnergy = 'Datac/meshC.csv'
    # fileNameForEnergy = 'Datac/meshCRefined.csv'
    fileNameForEnergy = 'annealingData/mns/energy.csv'
    # fileNameForEnergy = 'Datac/smallEnergyChorin.csv'
    # fileNameForEnergy = 'Datac/allBendsNewTransFineMesh300_400.csv'
    # fileNameForEnergy = 'Datac/mesh.csv'
    # fileN = 'Data/dataForRecursiveEquationEntropySmallNExact.csv'
    # fileN = 'Data/dataForRecusiveEquationEST.csv'
    fileN = 'Data/meshEST.csv'
    # fileN = 'Data/dataForRecursiveEquationEntropySmallNCopy.csv'

    # newAllBends
    newAllBendsFromStr = [[], []]
    newAllBendsFromHalf = [[], []]
    newAllBendsFromFull = [[], []]

    newAllBendsFromStrStats = [[], []]
    newAllBendsFromHalfStats = [[], []]
    newAllBendsFromFullStats = [[], []]

    ################################

    newAllBendsFromStrX = [[], []]
    newAllBendsFromHalfX = [[], []]
    newAllBendsFromFullX = [[], []]

    with open(fileNameForEnergy, 'rt') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["hasNewTrans"] == "True":
                hasNewTrans = True
            else:
                hasNewTrans = False

            if row["w"] == "True":
                tryNew = True
            else:
                tryNew = False

            if int(row["type"]) == 1:
                # does not have new transformations
                if hasNewTrans:
                    if int(row["ddn"]) == 1:
                        # str
                        if not tryNew:
                            # Gibbs
                            # allBends and hasNewTransformations and fromStr and Gibbs weight
                            newAllBendsFromStrX[0].append(
                                [int(row["N"]), int(row["n"]), int(row["dn"]), float(row["B"]), float(row["E"])])
                        else:
                            # modified Gibbs
                            newAllBendsFromStrX[1].append(
                                [int(row["N"]), int(row["n"]), int(row["dn"]), float(row["B"]), float(row["E"])])

                    elif int(row["ddn"]) == int(row["N"]):
                        # half
                        if not tryNew:
                            # Gibbs
                            # allBends and hasNewTransformations and fromStr and Gibbs weight
                            if ((int(row["N"]) == 101 or int(row["N"]) == 201 or int(row["N"]) == 301 or int(
                                    row["N"]) == 401 or (int(row["N"]) == 501 and int(row["dn"]) == 100000) or (
                                            int(row["N"]) == 601 and int(row["dn"]) == 200000) or (
                                            int(row["N"]) == 701 and int(row["dn"]) == 300000) or (
                                            int(row["N"]) == 801 and int(row["dn"]) == 250000)) is True):
                                newAllBendsFromHalfX[0].append(
                                    [int(row["N"]), int(row["n"]), int(row["dn"]), float(row["B"]), float(row["E"])])
                        else:
                            # modified Gibbs
                            newAllBendsFromHalfX[1].append(
                                [int(row["N"]), int(row["n"]), int(row["dn"]), float(row["B"]), float(row["E"])])
                    elif int(row["ddn"]) > int(row["N"]) or int(row["ddn"]) == -99:
                        # full
                        if not tryNew:
                            # Gibbs allBends and hasNewTransformations and fromStr and Gibbs weight if ((int(row[
                            # "N"])==101 or int(row["N"])==201 or int(row["N"])==301 or int(row["N"])==401 or (int(
                            # row["N"])==501 and int(row["dn"])==100000) or (int(row["N"])==601 and int(row[
                            # "dn"])==200000) or (int(row["N"])==701 and int(row["dn"])==300000) or (int(row[
                            # "N"])==801 and int(row["dn"])==350000))==True):
                            newAllBendsFromFullX[0].append(
                                [int(row["N"]), int(row["n"]), int(row["dn"]), float(row["B"]), float(row["E"])])
                        else:
                            # modified Gibbs
                            newAllBendsFromFullX[1].append(
                                [int(row["N"]), int(row["n"]), int(row["dn"]), float(row["B"]), float(row["E"])])

    with open(fileN, 'rt') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["hasNewTrans"] == "True":
                hasNewTrans = True
            else:
                hasNewTrans = False

            if row["w"] == "True":
                tryNew = True
            else:
                tryNew = False

            if int(row["type"]) == 1:
                # does not have new transformations
                if hasNewTrans:
                    if int(row["ddn"]) == 1:
                        # str
                        if not tryNew:
                            # Gibbs
                            # allBends and hasNewTransformations and fromStr and Gibbs weight
                            newAllBendsFromStrStats[0].append(
                                [int(row["N"]), float(row["B"]), float(row["F"]), float(row["S"]), float(row["SN"]),
                                 int(row["ns"])])
                        else:
                            # modified Gibbs
                            newAllBendsFromStrStats[1].append(
                                [int(row["N"]), float(row["B"]), float(row["F"]), float(row["S"]), float(row["SN"]),
                                 int(row["ns"])])

                    elif int(row["ddn"]) == int(row["N"]):
                        # half
                        if not tryNew:
                            # Gibbs
                            # allBends and hasNewTransformations and fromStr and Gibbs weight
                            newAllBendsFromHalfStats[0].append(
                                [int(row["N"]), float(row["B"]), float(row["F"]), float(row["S"]) * (int(row["N"]) - 1),
                                 float(row["SN"]), int(row["ns"])])
                        else:
                            # modified Gibbs
                            newAllBendsFromHalfStats[1].append(
                                [int(row["N"]), float(row["B"]), float(row["F"]), float(row["S"]) * (int(row["N"]) - 1),
                                 float(row["SN"]), int(row["ns"])])
                    elif int(row["ddn"]) > int(row["N"]) or int(row["ddn"]) == -99:
                        # full
                        if not tryNew:
                            # Gibbs
                            # allBends and hasNewTransformations and fromStr and Gibbs weight
                            newAllBendsFromFullStats[0].append(
                                [int(row["N"]), float(row["B"]), float(row["F"]), float(row["S"]) * (int(row["N"]) - 1),
                                 float(row["SN"]), int(row["ns"])])
                        else:
                            # modified Gibbs
                            newAllBendsFromFullStats[1].append(
                                [int(row["N"]), float(row["B"]), float(row["F"]), float(row["S"]) * (int(row["N"]) - 1),
                                 float(row["SN"]), int(row["ns"])])

    print(newAllBendsFromStrStats)
    # print newAllBendsFromFullX
    # print(newAllBendsFromHalfStats)
    # print(newAllBendsFromFullStats)

    # print(newAllBendsFromStrX)
    # print(newAllBendsFromHalfX)
    # print(newAllBendsFromFullX)

    # a1 = graphData("Graph of S/N for N=3, n=300000 and dn=100000",newAllBendsFromStrX,newAllBendsFromStrStats[0][0][3],newAllBendsFromStrStats[0][0][0]-1,0)
    # a2 = graphData("Graph of S/N for N=4, n=300000 and dn=100000",newAllBendsFromStrX,newAllBendsFromStrStats[0][1][3],newAllBendsFromStrStats[0][1][0]-1,1)
    # a3 = graphData("Graph of S/N for N=5, n=300000 and dn=100000",newAllBendsFromStrX,newAllBendsFromStrStats[0][2][3],newAllBendsFromStrStats[0][2][0]-1,2)
    # a4 = graphData("Graph of S/N for N=6, n=300000 and dn=100000",newAllBendsFromStrX,newAllBendsFromStrStats[0][3][3],newAllBendsFromStrStats[0][3][0]-1,3)
    # a5 = graphData("Graph of S/N for N=7, n=300000 and dn=100000",newAllBendsFromStrX,newAllBendsFromStrStats[0][4][3],newAllBendsFromStrStats[0][4][0]-1,4)
    # a6 = graphData("Graph of S/N for N=8, n=300000 and dn=100000",newAllBendsFromStrX,newAllBendsFromStrStats[0][5][3],newAllBendsFromStrStats[0][5][0]-1,5)
    # a7 = graphData("Graph of S/N for N=9, n=300000 and dn=100000",newAllBendsFromStrX,newAllBendsFromStrStats[0][6][3],newAllBendsFromStrStats[0][6][0]-1,6)
    print(len(newAllBendsFromFullStats[0]))
    print(len(newAllBendsFromFullX))

    a1 = graph_data("Graph of S/N for N=100", newAllBendsFromFullX, newAllBendsFromFullStats[0][0][3],
                    newAllBendsFromFullStats[0][0][0] - 1, 0)
    a2 = graph_data("Graph of S/N for N=200", newAllBendsFromFullX, newAllBendsFromFullStats[0][1][3],
                    newAllBendsFromFullStats[0][1][0] - 1, 1)
    a3 = graph_data("Graph of S/N for N=300", newAllBendsFromFullX, newAllBendsFromFullStats[0][2][3],
                    newAllBendsFromFullStats[0][2][0] - 1, 2)
    a4 = graph_data("Graph of S/N for N=400", newAllBendsFromFullX, newAllBendsFromFullStats[0][3][3],
                    newAllBendsFromFullStats[0][3][0] - 1, 3)
    a5 = graph_data("Graph of S/N for N=500", newAllBendsFromFullX, newAllBendsFromFullStats[0][4][3],
                    newAllBendsFromFullStats[0][4][0] - 1, 4)
    a6 = graph_data("Graph of S/N for N=600", newAllBendsFromFullX, newAllBendsFromFullStats[0][5][3],
                    newAllBendsFromFullStats[0][5][0] - 1, 5)
    a7 = graph_data("Graph of S/N for N=700", newAllBendsFromFullX, newAllBendsFromFullStats[0][6][3],
                    newAllBendsFromFullStats[0][6][0] - 1, 6)
    a8 = graph_data("Graph of S/N for N=800", newAllBendsFromFullX, newAllBendsFromFullStats[0][7][3],
                    newAllBendsFromFullStats[0][7][0] - 1, 7)

    # print len(a1)
    # print a1
    print('GOT TO HERE')
    top = -20
    bottom = 30

    mpl.rcParams['legend.fontsize'] = 17
    fig = plt.figure(figsize=(13, 8), dpi=80)
    ae = fig.add_subplot(111)
    # ae.set_title('All Bends Entropy(S/N) From Straight Filament With New Tranformations')
    ae.set_xlabel(r"$\beta$", fontsize=47)
    ae.set_ylabel(r"$S/N$", fontsize=47)
    ae.set_aspect('auto')
    ae.set_xlim(bottom + 0.5, top - 0.5)
    ae.set_ylim(-0.5, 1.8)

    ae.tick_params(axis='both', which='major', labelsize=31)
    ae.tick_params(axis='both', which='minor', labelsize=27)

    print(len(a1[0]))
    print(len(a1[1]))

    ae.plot(np.linspace(0, top, num=len(a1[0])), a1[0], '-ro', label='N=100')
    ae.plot(np.linspace(0, bottom, num=len(a1[1])), a1[1], '-ro')

    ae.plot(np.linspace(0, top, num=len(a2[0])), a2[0], '-bo', label='N=200')
    ae.plot(np.linspace(0, bottom, num=len(a2[1])), a2[1], '-bo')

    ae.plot(np.linspace(0, top, num=len(a3[0])), a3[0], '-ko', label='N=300')
    ae.plot(np.linspace(0, bottom, num=len(a3[1])), a3[1], '-ko')

    ae.plot(np.linspace(0, top, num=len(a4[0])), a4[0], '-yo', label='N=400')
    ae.plot(np.linspace(0, bottom, num=len(a4[1])), a4[1], '-yo')

    ae.plot(np.linspace(0, top, num=len(a5[0])), a5[0], '-co', label='N=500')
    ae.plot(np.linspace(0, bottom, num=len(a5[1])), a5[1], '-co')

    ae.plot(np.linspace(0, top, num=len(a6[0])), a6[0], '-ro', label='N=600')
    ae.plot(np.linspace(0, bottom, num=len(a6[1])), a6[1], '-ro')

    ae.plot(np.linspace(0, top, num=len(a7[0])), a7[0], '-bo', label='N=700')
    ae.plot(np.linspace(0, bottom, num=len(a7[1])), a7[1], '-bo')

    ae.plot(np.linspace(0, top, num=len(a8[0])), a8[0], '-ko', label='N=800')
    ae.plot(np.linspace(0, bottom, num=len(a8[1])), a8[1], '-ko')

    '''
    ae.plot(range(0,-101,-1),a4[0],'-yo',label='N=6')
    ae.plot(range(0,101,1),a4[1],'-yo')  
    
    ae.plot(range(0,-101,-1),a5[0],'-ko',label='N=7')
    ae.plot(range(0,101,1),a5[1],'-ko')

    ae.plot(range(0,-101,-1),a6[0],'-co',label='N=8')
    ae.plot(range(0,101,1),a6[1],'-co')     
    
    ae.plot(range(0,-101,-1),a7[0],'-mo',label='N=9')
    ae.plot(range(0,101,1),a7[1],'-mo')       
    '''

    ae.legend()
    ae.grid()
    plt.show()
    # for i in range(0,30):
