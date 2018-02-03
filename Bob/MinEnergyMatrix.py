import numpy as np


class MinEnergyMatrix:
    """
    Stores the minimum energy value and configurations with the energy for a saw of a particular length
    """

    def __init__(self, min_energy, matrix_config):
        self.total_checked = 0
        self.min_energy = min_energy
        self.matrix_config = matrix_config

    def __str__(self):
        ret_val = 'Min energy: ' + str(self.min_energy) + '\n'
        ret_val += 'Total checked: ' + str(self.total_checked) + '\n'
        ret_val += 'Min configs: ' + str(len(self.matrix_config)) + '\n'
        for k in range(0, len(self.matrix_config)):
            ret_val += np.array2string(self.matrix_config[k], separator=',', precision=0) + ';\n\n'
        return ret_val
