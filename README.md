This code generates self avoiding walks on a cubic lattice. Various methods are implemented for doing so as described below. The goal is to find the minimum energy walk configurations. 

To use:

Open \_\_main__.py and set variables as desired. 

N: max filament length to generate  
N_0: min filament length to generate, use with run_mode = 'subset'  
run_mode: all runs 2 to N, subset runs N_0 to N, single runs N, inf runs indefinitely  
final: true to output to final folder, false to output to test folder  
data_file_name: name of data csv file to write to  
final_directory: directory to write final output to  

Navigate to directory containing pyResearch folder. Run in command line:

```python pyResearch```

Note: project uses Python 3

Other info:

Check do_run method in ```computations.py``` file to see which algorithm is being used. The three methods are:  
-Brute force: fixes direction of first segment to be in positive x direction, uses recursion to generate all possible SAWs and checks for minimum energy, checks 1/6 of total possibilities   
-Fixed endpoint: optimization1 algorithm, uses a fixed beginning and end point and restricts SAW paths to no more than the length remaining from the end point, checks < 1% of total possibilities. We have observed that the endpoint is always a distance of 1 from the beginning for odd lengths and sqrt(2) for even lengths.
-Contained to box: optimization2 algorithm, does not let saw get outside of floor((length+1)^(1/3)), the floor of the cube root of 1 + the length. The pattern we have observed is that the minimum energy configurations fit within this box. 
