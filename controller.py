from computations import do_run
from csv_writing import setup_csv
import os
from datetime import datetime
import sys
import logging
import gc


def start(n_0: int, n: int, final: bool, data_file_name: str, run_mode: str, final_directory: str):
    # headers for data csv file
    csv_header = ['length', 'e_m', 'time', 'total configs', 'min energy configs']
    # logging setup
    logging_config = '[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)s] %(message)s'
    logging_time = '%m-%d-%Y %I:%M:%S %p'

    if final:
        directory = final_directory + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        logging.basicConfig(filename=directory + '/running.log', level=logging.INFO, format=logging_config,
                            datefmt=logging_time)
    else:
        date_time = datetime.now().strftime('%m-%d-%Y_%H.%M.%S')
        if run_mode == 'all':
            directory = 'test/' + date_time + '_all_%d/' % n
        elif run_mode == 'subset':
            directory = 'test/' + date_time + '_subset_%d/_%d' % (n_0, n)
        elif run_mode == 'inf':
            directory = 'test/' + date_time + '_inf'
        elif run_mode == 'single':
            directory = 'test/' + date_time + '_single_%d/' % n
        else:
            sys.exit('invalid run mode')
        os.makedirs(directory)
        logging.basicConfig(filename=directory + 'running.log', level=logging.DEBUG, format='%(asctime)s : %(message)s',
                            datefmt='%m-%d-%Y %I:%M:%S %p')

    logging.info("=============================================================================================")
    logging.info("beginning run")
    logging.info("final: %s, N: %i, N_0: %i, run_mode: %s, data_file_name: %s" % (final, n, n_0, run_mode, data_file_name))

    file = directory + data_file_name
    setup_csv(file, final, csv_header)

    # run 2 to N
    if run_mode == 'all':
        for i in range(2, n + 1):
            logging.info("beginning run %i of %i" % (i, n))
            do_run(i, file, directory, final)
            gc.collect()
    # run N_0 to N
    elif run_mode == 'subset':
        for i in range(n_0, n + 1):
            logging.info("beginning run %i of %i" % (i, n))
            do_run(i, file, directory, final)
            gc.collect()
    # run indefinitely
    elif run_mode == 'inf':
        for i in range(2, 1001):
            logging.info("beginning run %i of inf" % i)
            do_run(i, file, directory, final)
            gc.collect()
    # run N
    elif run_mode == 'single':
        logging.info("beginning run %i" % n)
        do_run(n, file, directory, final)
    else:
        logging.error('invalid run mode')
        sys.exit('invalid run mode')

    logging.info("run complete")

    print("Run complete")
