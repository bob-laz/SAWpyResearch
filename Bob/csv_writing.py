import csv
import logging
import os
from Bob.MinEnergyMatrix import MinEnergyMatrix
from collections import Counter

log = logging.getLogger('__main__')


def write_to_file(data_file: str, is_final: bool, length: int, min_energy_object: MinEnergyMatrix, time: str):
    """
    Writes one row of a data file. Technically, it reads all data from the file and writes it back changing just the
    one line to be written.
    :param data_file: csv file to write to
    :param is_final: true if writing to final directory, false if writing to test
    :param length: length of saw
    :param min_energy_object: object holding all minimum energy config info
    :param time: time taken to generate the saw
    """
    new_data = [length, min_energy_object.min_energy, time, min_energy_object.total_checked,
                len(min_energy_object.matrix_config)]
    if is_final:
        data_list = []  # store data currently in csv
        with open(data_file, newline='') as csv_file_read:
            reader = csv.reader(csv_file_read)
            data_list.extend(reader)
            # want to overwrite the N-1 line with new data, preserving the rest of the data
            if len(data_list) >= length:
                log.info("overwriting line %i with new data" % length)
                line_to_override = {length - 1: new_data}
                with open(data_file, 'w', newline='') as csv_file_write:
                    writer = csv.writer(csv_file_write)
                    for line, row in enumerate(data_list):
                        data = line_to_override.get(line, row)
                        writer.writerow(data)
            else:
                log.info("appending new data to eof at line %i" % length)
                with open(data_file, 'a', newline='') as csv_file_write:
                    writer = csv.writer(csv_file_write)
                    writer.writerow(new_data)
        configs_data_file = 'final2/configs/config' + str(length) + '.txt'
        with open(configs_data_file, 'w') as txt_file_write:
            txt_file_write.write(min_energy_object.__str__())
            txt_file_write.write('Energy computation factored:\n')
            counter = 0
            for key, val in energy_of_saw_factored(min_energy_object.matrix_config[0]).items():
                counter += val
                txt_file_write.write(key + ' : ' + str(val) + '\n')
            txt_file_write.write('Total interactions: ' + str(counter))
    else:
        with open(data_file, 'a', newline='') as csv_file_write:
            writer = csv.writer(csv_file_write, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(new_data)
        configs_data_file = 'test/configs/config' + str(length) + '.txt'
        with open(configs_data_file, 'w') as txt_file_write:
            txt_file_write.write(min_energy_object.__str__())
            txt_file_write.write('Energy computation factored:\n')
            counter = 0
            for key, val in energy_of_saw_factored(min_energy_object.matrix_config[0]).items():
                counter += val
                txt_file_write.write(key + ' : ' + str(val) + '\n')
            txt_file_write.write('Total interactions: ' + str(counter))


def setup_csv(data_file: str, is_final: bool, csv_header: []):
    """
    Checks to see if headers are present and writes them if they are for final. Writes headers to the new test file for
    test.
    :param data_file: data file to write to
    :param is_final: true if writing to final directory, false if writing to test
    :param csv_header: the content the csv header should contain
    """
    # if final, check to see if header is present and accurate
    if is_final:
        data_list = []  # store data currently in csv
        if os.path.exists(data_file):
            with open(data_file, newline='') as csv_file_read:
                reader = csv.reader(csv_file_read)
                data_list.extend(reader)
                # if file is empty or doesn't match expected, overwrite header while re-writing the rest of the data
                if not data_list or not data_list[0] == csv_header:
                    line_to_override = {0: csv_header}
                    with open(data_file, 'w', newline='') as csv_file_write:
                        writer = csv.writer(csv_file_write)
                        for line, row in enumerate(data_list):
                            data = line_to_override.get(line, row)
                            writer.writerow(data)
                    log.info("found problem with headers, overwrote new headers")
                else:
                    log.info("headers found to be present and correct, not written")
        else:
            with open(data_file, 'w', newline='') as csv_file_write:
                writer = csv.writer(csv_file_write)
                writer.writerow(csv_header)
            log.info("headers written for new test csv file")
    else:
        with open(data_file, 'w', newline='') as csv_file_write:
            writer = csv.writer(csv_file_write)
            writer.writerow(csv_header)
        log.info("headers written for new test csv file")


def energy_of_saw_factored(saw_matrix: [[int]]):
    """
    Computes the energy of the self-avoiding walk passed into the function as per the equation from Chronin's paper
    :param saw_matrix: an N x 3 matrix of the x, y, z coordinates of each point in the SAW
    :return: a floating point energy value
    """
    midpoints = saw_matrix.copy()
    directions = saw_matrix.copy()
    # calculate direction of each filament and midpoints of N-1 filaments
    for a in range(0, len(midpoints) - 1):
        directions[a] = [directions[a + 1][0] - directions[a][0], directions[a + 1][1] - directions[a][1],
                         directions[a + 1][2] - directions[a][2]]
        midpoints[a] = [(midpoints[a][0] + midpoints[a + 1][0]) / 2.0, (midpoints[a][1] + midpoints[a + 1][1]) / 2.0,
                        (midpoints[a][2] + midpoints[a + 1][2]) / 2.0]
    # total energy in system, used to keep track of energy throughout summations
    total_energy = []
    # outer summation from i = 0 to N-1, computes energy of every combination of filaments and tracks total for SAW
    for n in range(0, len(midpoints) - 1):
        direction_i = directions[n]
        midpoints_i = midpoints[n]
        # inner summation from j = i+1 to N-1
        for j in range(n + 1, len(midpoints) - 1):
            direction_j = directions[j]
            midpoint_j = midpoints[j]
            dot_product = (direction_i[0] * direction_j[0]) + (direction_i[1] * direction_j[1]) + (
                    direction_i[2] * direction_j[2])
            x_dif = (midpoints_i[0] - midpoint_j[0]) ** 2.0
            y_dif = (midpoints_i[1] - midpoint_j[1]) ** 2.0
            z_dif = (midpoints_i[2] - midpoint_j[2]) ** 2.0
            distance_abs = abs((x_dif + y_dif + z_dif))
            total_energy.append(str(dot_product) + ' / sqrt(' + str(distance_abs) + ')')
    return Counter(total_energy)
