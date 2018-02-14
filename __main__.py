from controller import start

""" SET THESE
N: max filament length to generate
N_0: min filament length to generate, use with run_mode = 'subset'
run_mode: all runs 2 to N, subset runs N_0 to N, single runs N, inf runs indefinitely
final: true to output to final folder, false to output to test folder
data_file_name: name of data csv file to write to
final_directory: directory to write final output to
"""
N = 12
N_0 = 2
run_mode = 'single'  # all, subset, inf, single
final = False
data_file_name = 'data.csv'
final_directory = 'final_fixed_endpoint'
""""""


start(N_0, N, final, data_file_name, run_mode, final_directory)
