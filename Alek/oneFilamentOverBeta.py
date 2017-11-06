# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 12:51:50 2016

@author: alek
"""
# import csv
import os.path
import sys
import time

# import energyCalc as ec
import numpy as np

from Alek import pivotFunctions as pf


# import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt

def annealing_method(b_vals, b_interval, N, n, dn, h, try_new, new_transforms, seed, file_name, directory):
    print('-* found initial state is straight')
    config = pf.all_bends_alg_all(0.0, N, 1, -1, False, h, try_new, new_transforms, seed)
    print('config[e]: ', config[0])
    config_file_name = "%sconfig_%d_%d.gz" % (directory, N, try_new)
    config = config[1]
    e_vals = []
    mu1_vals = []
    finished = False
    header = "x,y,z for beta=initial_filament, N=%d, n=%d, dn=%d, weight=%d, seed=%d" % (N, n, dn, try_new, seed)
    np.savetxt(config_file_name, config, header=header, delimiter=",", fmt='%d')
    con = np.array([(N, 0.0, b_interval, n, dn, try_new, seed, finished, config, b_vals)],
                   dtype=[('N', np.int32), ('b', np.float32), ('interval', np.float32), ('n', np.int32),
                          ('dn', np.int32), ('w', np.bool), ('seed', np.int32), ('finished', np.bool),
                          ('config', np.float64, (N, 3)), ('b_vals', np.float32, b_vals.shape)])
    data_file_name = "%sdataFor_%d_%d.npy" % (directory, N, try_new)
    print(data_file_name)
    np.save(data_file_name, con)

    # file_name = "%s%s" % (directory, file_name)

    while True:
        if b_vals.shape[0] > 0:
            b = b_vals[0]
            b_vals = np.delete(b_vals, 0)
        else:
            break
        print('-* creating the new graph using the old as the new')
        print('b: ', b)
        start = time.time()
        ans_pre = pf.all_bends_alg_all(b, N, dn, config, True, h, try_new, new_transforms, seed)[1]
        ans = pf.all_bends_alg_all(b, N, n, ans_pre, True, h, try_new, new_transforms, seed)
        e_vals.append(ans[0])
        mu1_vals.append(ans[2])
        config = ans[1]
        print('e: ', ans[0])
        print('mu_1,N: ', ans[2])
        # f = open(file_name, 'a')
        # try:
        #    writer = csv.writer(f)
        #    writer.writerow( (N,b,dn,n,ans[0],1,try_new,h,new_transforms,ans[2],ans[3],-99) )
        # finally:
        #    f.close()
        header = "x,y,z for beta=%d, N=%d, n=%d, dn=%d, weight=%d, seed=%d" % (b, N, n, dn, try_new, seed)
        np.savetxt(config_file_name, config, header=header, delimiter=",", fmt='%d')

        con = np.array([(N, b, b_interval, n, dn, try_new, seed, finished, config, b_vals)],
                       dtype=[('N', np.int32), ('b', np.float32), ('interval', np.float32), ('n', np.int32),
                              ('dn', np.int32), ('w', np.bool), ('seed', np.int32), ('finished', np.bool),
                              ('config', np.float64, (N, 3)), ('b_vals', np.float32, b_vals.shape)])
        np.save(data_file_name, con)
        print('time: ', time.time() - start)

    finished = True

    con = np.array([(N, -199, b_interval, n, dn, try_new, seed, finished, config, b_vals)],
                   dtype=[('N', np.int32), ('b', np.float32), ('interval', np.float32), ('n', np.int32),
                          ('dn', np.int32), ('w', np.bool), ('seed', np.int32), ('finished', np.bool),
                          ('config', np.float64, (N, 3)), ('b_vals', np.float32, b_vals.shape)])
    np.save(data_file_name, con)
    print('time: ', time.time() - start)
    return [e_vals, mu1_vals, config]


def annealing_method_on_old(b_vals, b_interval, N, n, dn, h, try_new, new_transforms, seed, file_name, directory,
                            config):
    # print '-* found initial state is straight'
    # config = pf.allBendsAlgAll(0.0,N,1,-1,False,h,try_new,new_transforms,seed)
    print('config[e]: ', config)
    config_file_name = "%sconfig_%d_%d.gz" % (directory, N, try_new)
    # config = config[1]
    e_vals = []
    mu1_vals = []
    finished = False
    data_file_name = "%sdataFor_%d_%d.npy" % (directory, N, try_new)
    # file_name = "%s%s" % (directory, file_name)

    while True:
        if b_vals.shape[0] > 0:
            B = b_vals[0]
            b_vals = np.delete(b_vals, 0)
        else:
            break
        print('-* creating the new graph using the old as the new')
        print('B: ', B)
        start = time.time()
        ans_pre = pf.all_bends_alg_all(B, N, dn, config, True, h, try_new, new_transforms, seed)[1]
        ans = pf.all_bends_alg_all(B, N, n, ans_pre, True, h, try_new, new_transforms, seed)
        e_vals.append(ans[0])
        mu1_vals.append(ans[2])
        config = ans[1]
        print('e: ', ans[0])
        print('mu_1,N: ', ans[2])
        # f = open(file_name, 'a')
        # try:
        #    writer = csv.writer(f)
        #    writer.writerow( (N,B,dn,n,ans[0],1,try_new,h,new_transforms,ans[2],ans[3],-99) )
        # finally:
        #    f.close()
        header = "x,y,z for beta=%d, N=%d, n=%d, dn=%d, weight=%d, seed=%d" % (B, N, n, dn, try_new, seed)
        np.savetxt(config_file_name, config, header=header, delimiter=",", fmt='%d')

        con = np.array([(N, B, b_interval, n, dn, try_new, seed, finished, config, b_vals)],
                       dtype=[('N', np.int32), ('B', np.float32), ('interval', np.float32), ('n', np.int32),
                              ('dn', np.int32), ('w', np.bool), ('seed', np.int32), ('finished', np.bool),
                              ('config', np.float64, (N, 3)), ('b_vals', np.float32, b_vals.shape)])
        np.save(data_file_name, con)
        print('time: ', time.time() - start)

    finished = True

    con = np.array([(N, -199, b_interval, n, dn, try_new, seed, finished, config, b_vals)],
                   dtype=[('N', np.int32), ('B', np.float32), ('interval', np.float32), ('n', np.int32),
                          ('dn', np.int32), ('w', np.bool), ('seed', np.int32), ('finished', np.bool),
                          ('config', np.float64, (N, 3)), ('b_vals', np.float32, b_vals.shape)])
    np.save(data_file_name, con)
    print('time: ', time.time() - start)
    return [e_vals, mu1_vals, config]


def work_on_old_sim(con, b_vals):
    print(con)
    print('HHHHHHHHHHHHHH')

    while True:
        print(b_vals)
        if con["B"][0] != b_vals[0]:
            b_vals = np.delete(b_vals, 0)
        else:
            b_vals = np.delete(b_vals, 0)
            break

    print(b_vals)
    return b_vals


if __name__ == '__main__':
    seed = 1
    tryNew = True
    newTransforms = True
    presA = 101  # 201
    fromB = -20.0
    toB = 30.0
    bVals = np.linspace(fromB, toB, num=presA)
    b_interval = bVals[1] - bVals[0]
    n = 30000
    dn = 30000
    if len(sys.argv) < 2:
        print('could not find value')
        exit()
    else:
        print('Argument List[1](N value):', int(sys.argv[1]))
        N = int(sys.argv[1])
    h = 1.0
    directory = 'annealingData/4/'
    fileName = 'energy.csv'
    matrixName = "%sconfig_%d_%d.gz" % (directory, N, tryNew)
    file_temp = "%sdataFor_%d_%d.npy" % (directory, N, tryNew)

    if os.path.exists(file_temp):
        con = np.load(file_temp)
        print(con)
        if con["finished"][0]:
            print('done')
        else:
            data = annealing_method_on_old(con["bVals"][0], con["interval"][0], con["N"][0], con["n"][0]
                                           , con["dn"][0], h, con["w"][0], newTransforms, con["seed"][0], fileName,
                                           directory, con['config'][0])
    else:
        print('IIIIIIIIIIIIIII')
        data = annealing_method(bVals, b_interval, N, n, dn, h, tryNew, newTransforms, seed, fileName, directory)
