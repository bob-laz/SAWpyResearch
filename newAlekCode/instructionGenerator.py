#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 14:47:45 2017

@author: alek
"""
from newAlekCode.helpfulFunctions import *
from newAlekCode.compMatXC import *
import numpy
from numba import jit
import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D

# mpl.use('Agg')




# get the current insturction set
def get_instruction_set(state, point_rotations):
    instructions = []
    for index in range(0, len(state)):
        instructions.append(point_rotations[index + 1][state[index]])
    return instructions


# based on the current state find the next state to go to
@jit(nopython=True)
def next_state(state, length_of_point_instructions):
    remainder = 1
    for index in range(len(state) - 1, -1, -1):
        remainder = (state[index] + remainder) % length_of_point_instructions
        if remainder == 0:
            state[index] = 0
            remainder = 1
        else:
            state[index] = remainder
            break
    return state


# testing function to show all states
def generate_all_states(N):
    state = [0 for _ in range(0, N - 1)]
    while True:
        print(state)
        state = next_state(state, 5)
        if sum(state) == 0:
            break


# return a dictionary of point rotations
def generate_point_rotations(N, rotations=['+z', '-z', '+y', '-y']):
    point_rotations = {}
    # go from point 1 to point N+1
    # example:
    #   N=1: ,
    #   N=2: 1,
    #   N=3: 1,2
    #   N=4: 1,2,3
    for point in range(1, N):
        # add each roation to the point
        point_rotation_list = [()]
        for rotation in rotations:
            point_rotation_list.append((point, rotation))
        point_rotations[point] = point_rotation_list
    return point_rotations


# this is a generator function, you can think of it as a stack or list.
def generate_instructions(N, rotations=['+z', '-z', '+y', '-y']):
    # print ('generating instructions')
    # print ('-* N: %d' % N)
    # print ('-* rotations: %s' % rotations)
    point_rotations = generate_point_rotations(N, rotations)
    # print ('-* point_rotations: %s' % point_rotations)
    if N <= 1:
        return [[()]]
    state = [0 for _ in range(0, N - 1)]
    while True:
        instruction = get_instruction_set(state, point_rotations)
        state = next_state(state, len(rotations) + 1)
        yield instruction
        if sum(state) == 0:
            break


def point_rotation(filament, instruction, rotation_matrices):
    center = filament[instruction[0], :].copy()
    matToRot = filament[instruction[0]:filament.shape[0], :]
    mutation = numpy.dot((matToRot - center), rotation_matrices[instruction[1]]) + center
    filament[instruction[0]:filament.shape[0]] = mutation
    return filament


def transform_filament(base_filament, instruction_set, rotation_matrices):
    for instruction in reversed(instruction_set):
        if len(instruction) == 0:
            continue
        base_filament = point_rotation(base_filament, instruction, rotation_matrices)
    return base_filament


def get_filament(N, h, instruction_set, rotation_matrices):
    base_filament = request_filament_matrix(N + 1, h)
    for instruction in reversed(instruction_set):
        print(instruction)
        filament = point_rotation(base_filament, instruction, rotation_matrices)
    return filament


def generate_filaments(N, h):
    rotation_matrices = get_rotation_matrices()
    base_filament = request_filament_matrix(N + 1, h)
    for instruction_set in generate_instructions(N):
        yield transform_filament(base_filament.copy(), instruction_set, rotation_matrices)


def get_rotation_matrices():
    rotyp = numpy.zeros(shape=[3, 3])
    rotyn = rotyp.copy()
    rotzp = rotyp.copy()
    rotzn = rotyp.copy()
    # +y
    rotyp[0] = [0, 0, -1]
    rotyp[1] = [0, 1, 0]
    rotyp[2] = [1, 0, 0]
    # -y
    rotyn[0] = [0, 0, 1]
    rotyn[1] = [0, 1, 0]
    rotyn[2] = [-1, 0, 0]
    # +z
    rotzp[0] = [0, 1, 0]
    rotzp[1] = [-1, 0, 0]
    rotzp[2] = [0, 0, 1]
    # -z
    rotzn[0] = [0, -1, 0]
    rotzn[1] = [1, 0, 0]
    rotzn[2] = [0, 0, 1]

    rotation_matrices = {'+z': rotzp, '-z': rotzn, '+y': rotyp, '-y': rotyn}

    return rotation_matrices


def graph_filament(filament):
    print('')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    ax.set_zlim3d(-5, 5)
    ax.set_xlim3d(-5, 5)
    ax.set_ylim3d(-5, 5)
    # plot out data to the screen
    ax.plot(filament[:, 0], filament[:, 1], filament[:, 2], 'ro-', label='filament segment')
    ax.legend()
    # fig.savefig('E:\PythonWorkSaves\PythonWork_04_06_2016/PosTFigs/config.png')   # save the figure to file
    # plt.close(fig)
    plt.show()


if __name__ == '__main__':
    import time

    # note that the number is in segments NOT points
    N = 8
    h = 1.0
    filaments = []

    print('generating all of the filaments for N=%d' % N)
    start_time = time.time()
    for filament in generate_filaments(N, h):
        intersects_self = checkForSelfIntersection(filament)
        if intersects_self:
            continue
        filaments.append(filament)
        #graph_filament(filament)
    end_time = time.time()
    # print ('filaments: ', filaments)

    print('number of instructions for N=%d: %d' % (N, len(filaments)))
    print('number of possible SAWs: %d' % (len(filaments) * 6))
    print('duration %1.3f seconds' % (end_time - start_time))
