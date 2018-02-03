import csv
import logging
import os


log = logging.getLogger('__main__')


def write_to_file(data_file, is_final, length, min_energy, time, total_configs, min_energy_configs):
    """
    Writes one row of a data file. Technically, it reads all data from the file and writes it back changing just the
    one line to be written.
    :param data_file: csv file to write to
    :param is_final: true if writing to final directory, false if writing to test
    :param length: length of saw
    :param min_energy: minimum calculated energy value
    :param time: time taken to generate the saw
    :param total_configs: total configurations checked of this length
    :param min_energy_configs: number of configurations with this minimum energy
    :return:
    """
    new_data = [length, min_energy, time, total_configs, min_energy_configs]
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
    else:
        with open(data_file, 'a', newline='') as csv_file_write:
            writer = csv.writer(csv_file_write, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(new_data)


def setup_csv(data_file, is_final, csv_header):
    """
    Checks to see if headers are present and writes them if they are for final. Writes headers to the new test file for
    test.
    :param data_file: data file to write to
    :param is_final: true if writing to final directory, false if writing to test
    :param csv_header: the content the csv header should contain
    :return:
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
