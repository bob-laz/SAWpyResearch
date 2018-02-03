from Bob.computations import do_run
from Bob.csv_writing import setup_csv
import os
from datetime import datetime
import sys
import logging

""" SET THESE
N: max filament length to generate
N_0: min filament length to generate, use with run_mode = 'subset'
run_mode: all runs 2 to N, subset runs N_0 to N, single runs N, inf runs indefinitely
final: true to output to final folder, false to output to test folder
data_file_name: name of data csv file to write to
"""
N = 30
N_0 = 15
run_mode = 'subset'  # all, subset, inf, single
final = True
data_file_name = 'data.csv'
""""""
# headers for data csv file
csv_header = ['length', 'e_m', 'time', 'total configs', 'min energy configs']
# logging setup
logging_config = '[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)s] %(message)s'
logging_time = '%m-%d-%Y %I:%M:%S %p'

if final:
    directory = 'final2/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    logging.basicConfig(filename=directory + '/running.log', level=logging.INFO, format=logging_config,
                        datefmt=logging_time)
else:
    date_time = datetime.now().strftime('%m-%d-%Y_%H.%M.%S')
    if run_mode == 'all':
        directory = 'test/' + date_time + '_all_%d/' % N
    elif run_mode == 'subset':
        directory = 'test/' + date_time + '_subset_%d/_%d' % (N_0, N)
    elif run_mode == 'inf':
        directory = 'test/' + date_time + '_inf'
    elif run_mode == 'single':
        directory = 'test/' + date_time + '_single_%d/' % N
    else:
        sys.exit('invalid run mode')
    os.makedirs(directory)
    logging.basicConfig(filename=directory + 'running.log', level=logging.DEBUG, format='%(asctime)s : %(message)s',
                        datefmt='%m-%d-%Y %I:%M:%S %p')

logging.info("=============================================================================================")
logging.info("beginning run")
logging.info("final: %s, N: %i, N_0: %i, run_mode: %s, data_file_name: %s" % (final, N, N_0, run_mode, data_file_name))

file = directory + data_file_name
setup_csv(file, final, csv_header)

# run 2 to N
if run_mode == 'all':
    for i in range(2, N + 1):
        logging.info("beginning run %i of %i" % (i, N))
        do_run(i, file, directory, final)
# run N_0 to N
elif run_mode == 'subset':
    for i in range(N_0, N + 1):
        logging.info("beginning run %i of %i" % (i, N))
        do_run(i, file, directory, final)
# run indefinitely
elif run_mode == 'inf':
    for i in range(2, 1001):
        logging.info("beginning run %i of inf" % i)
        do_run(i, file, directory, final)
# run N
elif run_mode == 'single':
    logging.info("beginning run %i" % N)
    do_run(N, file, directory, final)
else:
    logging.error('invalid run mode')
    sys.exit('invalid run mode')

logging.info("run complete")
