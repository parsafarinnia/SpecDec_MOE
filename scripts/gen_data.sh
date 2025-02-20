#!/bin/bash
# This script activates a conda environment, runs the Python allocation script,
# and logs the output to a text file.

# 1. Source conda so we can use 'conda activate'
#    (Adjust the path below to the location where your conda is installed.)
source ~/miniconda3/etc/profile.d/conda.sh

# 2. Activate your conda environment
conda activate specmoe

cd ..
# 3. Define the output directory path
OUTDIR="/work/farinneya/eagle_data_llama3_8B"  # Replace with the actual path of your data output directory

# 4. Run the Python command and log output to a file
#    - '>' redirects standard output (stdout) to the file
#    - '2>&1' redirects standard error (stderr) to stdout, combining both into the same file
python -m eagle.ge_data.allocation --outdir "$OUTDIR" >/home/farinneya/SpecDec_MOE/eagle/logs/allocation_output.log 2>&1

# End of script
