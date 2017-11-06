# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 13:12:45 2016

@author: alek
"""
import csv
import multiprocessing as mp
import time
from sys import exit

import numpy as np
import psutil

# import all pivot modules
from Alek import pivotFunctions as pf


def create_all_jobs_and_initial_configs(global_seed, types, trans, weights, simsA, presA, fromB, toB, dnA, valuesOfN,
                                        valuesB, h, BZerodns):
    jobs_for_initial_states = []
    for ddt in types:
        for hasNewTrans1 in trans:
            for selectedN1 in valuesOfN:
                for BZerodn1 in BZerodns:
                    if BZerodn1 == selectedN1 or BZerodn1 == 1 or BZerodn1 == BZeroMax:
                        jobs_for_initial_states.append([selectedN1, ddt, BZerodn1, hasNewTrans1, -1])
                        # [selectedN1,ddt,BZerodn1,hasNewTrans1,matrix]

    procs = list()
    n_cpus = psutil.cpu_count()
    queue = mp.Queue(0)
    total = len(jobs_for_initial_states)
    print('len(jobs): ', len(jobs_for_initial_states))
    index = 0
    for cpu in range(n_cpus):
        if len(jobs_for_initial_states) != 0:
            if index == len(jobs_for_initial_states):
                break
            if index != len(jobs_for_initial_states):
                for a in range(index, len(jobs_for_initial_states)):
                    ab = jobs_for_initial_states
                    p = mp.Process(target=pf.central_function, args=(
                        ab[a][1], ab[a][0], 0.0, ab[a][2], 1, -1, False, -1, False, True, queue, False, h, index,
                        ab[a][3], global_seed, -1.0, True, False))
                    # pf.centralFunction(a[1],a[0],0.0,a[2],1,-1,False,-1,False,False,-1,False,h,-1,a[3],global_seed,-1)
                    print('index: ', index)
                    break
            p.start()
            procs.append([index, p])
            index = index + 1
        else:
            print('no jobs')

    print('---collecting...')
    while True:
        if len(procs) == 0 and index == total:
            break
        new_answer = queue.get()
        jobs_for_initial_states[new_answer[0]][4] = new_answer[-1]
        for jobs_index in procs:
            if jobs_index[0] == new_answer[0]:
                time.sleep(0.2)
                jobs_index[1].terminate()
                procs.remove(jobs_index)
                if len(jobs_for_initial_states) != 0:
                    if index != len(jobs_for_initial_states):
                        for b in range(index, len(jobs_for_initial_states)):
                            ba = jobs_for_initial_states
                            pa2 = mp.Process(target=pf.central_function, args=(
                                ba[b][1], ba[b][0], 0.0, ba[b][2], -1, -1, False, -1, False, True, queue, False, h,
                                index,
                                ba[b][3], global_seed, -1.0, True, False))
                            # pf.centralFunction(a[1],a[0],0.0,a[2],1,-1,False,-1,False,False,-1,False,h,-1,a[3],global_seed,-1)
                            print('-index: ', index)
                            break
                    else:
                        break

                    pa2.start()
                    procs.append([index, pa2])
                    index = index + 1
                    break

    jobs = []
    for dn in dnA:
        for n in simsA:
            for weight in weights:
                for t in types:
                    # print '-New Type: ',t
                    for hasNewTrans in trans:
                        # print '-New trans: ',hasNewTrans
                        for B in valuesB:
                            # print '-NEw B value: ',B
                            for N in valuesOfN:
                                # print '-New N value: ',N
                                for BZerodn in BZerodns:
                                    if BZerodn == N or BZerodn == 1 or BZerodn == BZeroMax:
                                        jobs.append([N, n, B, t, hasNewTrans, dn, weight, BZerodn])

    return [jobs, jobs_for_initial_states]


if __name__ == '__main__':
    # fileName = raw_input('File name: ')
    #########################
    ######parameters#########
    #########################
    # fileName = 'Datac/smallEnergyChorin.csv'
    # fileName = 'Datac/smallEnergyChorin.csv'
    # fileName = 'Datac/allBendsNewTransEqualSpace.csv'
    fileName = 'Datac/meshCRefined.csv'
    global_seed = 2
    types = [1]
    trans = [True]
    weights = [False]
    simsA = [700000]
    presA = 61
    fromB = 30.0
    toB = -30.0
    dnA = [700000]
    valuesOfN = [801]
    valuesB = np.linspace(fromB, toB, num=presA)
    # example for valuesB
    # valuesB = [0.0]
    h = 1.0

    '''look at this VARIABLE'''
    BZeroMax = 150000
    # must have line below
    BZerodns = [BZeroMax]  # remove '1' to remove straight configs. Must have at least 'BZerodns = []'.
    # BZerodns.extend(valuesOfN) #comment out this line to remove N-step configs
    #########################
    ######parameters#########
    #########################

    ######start timer#####  
    timer = time.time()

    jobsAndConfigs = create_all_jobs_and_initial_configs(global_seed, types, trans, weights, simsA, presA, fromB, toB,
                                                         dnA,
                                                         valuesOfN, valuesB, h, BZerodns)
    jobs = jobsAndConfigs[0]
    configs = jobsAndConfigs[1]
    print('---Starting all jobs:')
    print('------------------------')
    print('------------------------')
    print('---jobs[0]: ', jobs[0])
    # jobsAndConfigs[1] = [N,type,BZerodn,hasNewTrans,[mean_energy,matrix,muOne,muTwo]]
    # jobsAndConfigs[0] = [N,n,B,type,hasNewTrans,dn,weight,BZerodn]

    with open(fileName, 'rt') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for myJob in jobs:
                if (int(row["N"]) == myJob[0] and float(row["B"]) == myJob[2] and int(row["dn"]) == myJob[5] and int(
                        row["n"]) == myJob[1] and int(row["type"]) == myJob[3] and row["w"] == str(myJob[6]) and float(
                    row["h"]) == h and row["hasNewTrans"] == str(myJob[4]) and int(row["ddn"]) == myJob[
                    7]): jobs.remove(myJob)

    proci = 0
    procs = list()
    n_cpus = psutil.cpu_count()
    queue1 = mp.Queue(0)

    print('len(jobs): ', len(jobs))
    answersNeeded = len(jobs)
    noJobs = False
    for cpu in range(n_cpus):
        if len(jobs) != 0:
            for matX in configs:
                # [selectedN1,ddt,BZerodn1,hasNewTrans1,matrix]
                if (jobs[0][0] == matX[0] and jobs[0][3] == matX[1] and jobs[0][7] == matX[2] and jobs[0][4] == matX[
                    3]):
                    # centralFunction(bendType,N,B,n,dn,matX,hasMatrix,returnValue,burn,hasQuo,quo)
                    p = mp.Process(target=pf.central_function, args=(
                        jobs[0][3], jobs[0][0], jobs[0][2], jobs[0][1], jobs[0][5], matX[4], True, -1, True, True,
                        queue1,
                        jobs[0][6], h, proci, jobs[0][4], global_seed, jobs[0][7], False, False))
                    print('proci: ', proci)
                    break

            p.start()
            procs.append([proci, p])
            del jobs[0]
            proci = proci + 1
        else:
            print('no jobs')
            noJobs = True

    done = False
    while True:
        print('-proci: ', proci)
        if len(jobs) == 0 and len(procs) == 0:
            break
        newAnswer = queue1.get()
        # “N”,”B”,”dn”,”n”,”E”,”type”,”w”,”h”,”hasNewTrans”,”mu1”,”mu2”,”ddn”
        # [E,mu1,mu2,N,dn,n,B,tryNew,hasNewTrans,h,ddn,bendType,proci]
        f = open(fileName, 'a')
        try:
            a = newAnswer
            writer = csv.writer(f)
            writer.writerow((a[3], a[6], a[4], a[5], a[0], a[11], a[7], a[9], a[8], a[1], a[2], a[10]))
        finally:
            f.close()

        for jobsIndex in procs:
            if jobsIndex[0] == newAnswer[-1]:
                time.sleep(0.2)
                jobsIndex[1].terminate()
                procs.remove(jobsIndex)
                if (len(jobs) != 0):
                    for matX in configs:
                        if (jobs[0][0] == matX[0] and jobs[0][3] == matX[1] and jobs[0][7] == matX[2] and jobs[0][4] ==
                            matX[3]):
                            pa = mp.Process(target=pf.central_function, args=(
                                jobs[0][3], jobs[0][0], jobs[0][2], jobs[0][1], jobs[0][5], matX[4], True, -1, True,
                                True,
                                queue1, jobs[0][6], h, proci, jobs[0][4], global_seed, jobs[0][7], False, False))
                            break
                    pa.start()
                    procs.append([proci, pa])
                    del jobs[0]
                    proci = proci + 1

    print("\n---time to complete: %1.3f seconds, or %1.3f minutes" % (
        float(time.time() - timer), float((time.time() - timer) / 60.0)))
    exit()
