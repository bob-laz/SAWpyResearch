# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 09:01:50 2017

@author: alek
"""

import pandas as pd
import numpy as np
import matplotlib as plt
# mpl.use('Agg')
#import matplotlib.pyplot as plt
from newAlekCode.energyCalc import *


# load previous data table or create a new one
# if that table does not exist.
def load_previous_data_table(location):
    try:
        table = pd.read_csv(location)
    except IOError:
        return request_data_table()
    return table


def request_data_table(index=[]):
    table = pd.DataFrame(index=index,
                         columns=['N', 'N_offset', 'B', 'E', 'weight', 'h', 'mu1', 'mu2', 'ddn', 'dn', 'n', 'seed',
                                  'annealingMethod', 'runWithFirstNewTrans'])
    return table


def request_filament_matrix(N, h):
    matrix = np.zeros(shape=(int(N), 3))
    for iy in range(0, matrix.shape[0]):
        matrix[iy, 0] = iy * h
        matrix[iy, 1] = 0
        matrix[iy, 2] = 0
    return matrix


def show_plot(plot, fig, name='', has_grid=True, has_legend=False, save=False):
    if plot is not None:
        if has_legend: plot.legend()
        if has_grid: plot.grid()
    if save:
        location = "%s.png" % name
        fig.savefig(location, dpi=150, format='png', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show(plot)


def add_all_max_energies_to_plot(table, label, plot):
    N_values = table['N'].unique()
    for N in N_values:
        label = "%s %d" % (label, (N + table["N_offset"].unique()[0]))
        add_max_energy_to_plot((N + table["N_offset"].unique()[0]), label, plot)


def add_max_energy_to_plot(N, label, plot):
    energy = calculate_max_energy(N)
    plot.axhline(y=energy, linestyle='--', label=label)


def calculate_max_energy(N):
    a = np.zeros(shape=(N, 3))
    for iy in range(0, a.shape[0]):
        a[iy, 0] = iy
        a[iy, 1] = 0
        a[iy, 2] = 0
    return energyInChainOPc(a)


def get_exact_energy_data(location, N):
    location_label = "%s/E%d.csv" % (location, N)
    exact_energy_data = pd.read_csv(location_label, header=None)
    exact_energy_data.columns = ['E']
    exact_energy_data['B'] = -100 + (exact_energy_data.index * 0.5)
    return exact_energy_data


def add_exact_energy_to_plot(location, plot):
    for N in range(3, 10):
        label = "N: %d (exact)" % N
        exact_energy_data = get_exact_energy_data(location, N)
        plot.plot(exact_energy_data['B'], exact_energy_data['E'], '--', label=label)


def add_exact_entorpy_to_plot(location, plot):
    if plot is None:
        return
    for N in range(3, 10):
        location_label = "%s/S%d.csv" % (location, N)
        label = "N: %d (exact)" % N
        exact_entropy_data = pd.read_csv(location_label, header=None)
        exact_entropy_data['B'] = -100 + (exact_entropy_data.index * 0.5)
        # exact_entropy_data.sort_values('B', ascending=False)
        plot.plot(exact_entropy_data['B'], exact_entropy_data[0], '--', label=label)
