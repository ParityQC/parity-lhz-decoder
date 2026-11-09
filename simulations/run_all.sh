#!/bin/bash
# Reruns the simulations with the compiled decoders, using the commands listed
# in the notebooks. Run it from the repository root:
#
#   caffeinate -i -s bash simulations/run_all.sh
#
# Not included, run them separately:
# - the cat-qubit simulation, which runs until 100 block errors per point (resumable):
#     python simulations/cat_qubit_cln/parallel_simulation.py
# - the random matching and the lower bound of Fig. 4, which its notebook computes when it is run.
set -e

# ---------------------------------------------------------------- code capacity
# Fig. 5
python simulations/code_capacity/logical_error_rate_ism.py --distances 5 11 17 23 29 35 --error-rates 0.05 0.50 0.02 --shots 50000 --no-post-processing --output simulations/code_capacity/Fig-5-LER/data/ism.csv
python simulations/code_capacity/logical_error_rate_ism.py --distances 5 11 17 23 29 35 --error-rates 0.05 0.50 0.02 --shots 50000 --output simulations/code_capacity/Fig-5-LER/data/ism_pp.csv
python simulations/code_capacity/logical_error_rate_gsm.py --distances 5 11 17 23 29 35 --error-rates 0.05 0.50 0.02 --shots 50000 --no-post-processing --output simulations/code_capacity/Fig-5-LER/data/gsm.csv
python simulations/code_capacity/logical_error_rate_gsm.py --distances 5 11 17 23 29 35 --error-rates 0.05 0.50 0.02 --shots 50000 --output simulations/code_capacity/Fig-5-LER/data/gsm_pp.csv

# Fig. 4 (MWPM and ISM)
python simulations/code_capacity/logical_error_rate_gsm.py --distances 11 31 51 --error-rates 0.10 0.51 0.02 --shots 50000 --output simulations/code_capacity/Fig-4-matching-benchmark/data/gsm_pp.csv
python simulations/code_capacity/logical_error_rate_ism.py --distances 11 31 51 --error-rates 0.10 0.51 0.02 --shots 50000 --output simulations/code_capacity/Fig-4-matching-benchmark/data/ism_pp.csv

# Fig. 13
python simulations/code_capacity/post_processing_cycles.py --decoder gsm --distances 7 11 15 19 23 27 31 35 39 --error-rates 0.1 0.21 0.05 --shots 50000 --output simulations/code_capacity/Fig-13-PP-cycles/data/gsm.csv
python simulations/code_capacity/post_processing_cycles.py --decoder ism --distances 7 11 15 19 23 27 31 35 39 --error-rates 0.1 0.21 0.05 --shots 50000 --output simulations/code_capacity/Fig-13-PP-cycles/data/ism.csv

# Fig. 6
python simulations/code_capacity/logical_error_rate_gsm.py --distances 5 7 --error-rates 0.175 0.226 0.005 --shots 50000 --no-post-processing --output simulations/code_capacity/Fig-6-threshold/data/gsm/d05_07.csv
python simulations/code_capacity/logical_error_rate_gsm.py --distances 13 15 --error-rates 0.225 0.276 0.005 --shots 50000 --no-post-processing --output simulations/code_capacity/Fig-6-threshold/data/gsm/d13_15.csv
python simulations/code_capacity/logical_error_rate_gsm.py --distances 21 23 --error-rates 0.255 0.306 0.005 --shots 50000 --no-post-processing --output simulations/code_capacity/Fig-6-threshold/data/gsm/d21_23.csv
python simulations/code_capacity/logical_error_rate_gsm.py --distances 29 31 --error-rates 0.255 0.306 0.005 --shots 50000 --no-post-processing --output simulations/code_capacity/Fig-6-threshold/data/gsm/d29_31.csv
python simulations/code_capacity/logical_error_rate_gsm.py --distances 37 39 --error-rates 0.27 0.321 0.005 --shots 50000 --no-post-processing --output simulations/code_capacity/Fig-6-threshold/data/gsm/d37_39.csv
python simulations/code_capacity/logical_error_rate_gsm.py --distances 45 47 --error-rates 0.28 0.331 0.005 --shots 50000 --no-post-processing --output simulations/code_capacity/Fig-6-threshold/data/gsm/d45_47.csv

python simulations/code_capacity/logical_error_rate_gsm.py --distances 5 7 --error-rates 0.235 0.286 0.005 --shots 50000 --output simulations/code_capacity/Fig-6-threshold/data/gsm_pp/d05_07.csv
python simulations/code_capacity/logical_error_rate_gsm.py --distances 13 15 --error-rates 0.315 0.366 0.005 --shots 50000 --output simulations/code_capacity/Fig-6-threshold/data/gsm_pp/d13_15.csv
python simulations/code_capacity/logical_error_rate_gsm.py --distances 21 23 --error-rates 0.34 0.391 0.005 --shots 50000 --output simulations/code_capacity/Fig-6-threshold/data/gsm_pp/d21_23.csv
python simulations/code_capacity/logical_error_rate_gsm.py --distances 29 31 --error-rates 0.36 0.426 0.005 --shots 50000 --output simulations/code_capacity/Fig-6-threshold/data/gsm_pp/d29_31.csv
python simulations/code_capacity/logical_error_rate_gsm.py --distances 37 39 --error-rates 0.37 0.426 0.005 --shots 100000 --output simulations/code_capacity/Fig-6-threshold/data/gsm_pp/d37_39.csv
python simulations/code_capacity/logical_error_rate_gsm.py --distances 45 47 --error-rates 0.38 0.431 0.005 --shots 100000 --output simulations/code_capacity/Fig-6-threshold/data/gsm_pp/d45_47.csv

python simulations/code_capacity/logical_error_rate_ism.py --distances 5 7 --error-rates 0.235 0.286 0.005 --shots 50000 --output simulations/code_capacity/Fig-6-threshold/data/ism_pp/d05_07.csv
python simulations/code_capacity/logical_error_rate_ism.py --distances 13 15 --error-rates 0.315 0.366 0.005 --shots 50000 --output simulations/code_capacity/Fig-6-threshold/data/ism_pp/d13_15.csv
python simulations/code_capacity/logical_error_rate_ism.py --distances 21 23 --error-rates 0.34 0.391 0.005 --shots 50000 --output simulations/code_capacity/Fig-6-threshold/data/ism_pp/d21_23.csv
python simulations/code_capacity/logical_error_rate_ism.py --distances 29 31 --error-rates 0.36 0.426 0.005 --shots 50000 --output simulations/code_capacity/Fig-6-threshold/data/ism_pp/d29_31.csv
python simulations/code_capacity/logical_error_rate_ism.py --distances 37 39 --error-rates 0.37 0.431 0.005 --shots 100000 --output simulations/code_capacity/Fig-6-threshold/data/ism_pp/d37_39.csv
python simulations/code_capacity/logical_error_rate_ism.py --distances 45 47 --error-rates 0.38 0.431 0.005 --shots 100000 --output simulations/code_capacity/Fig-6-threshold/data/ism_pp/d45_47.csv

# ---------------------------------------------------------------- bit-flip circuit-level noise
# Fig. 10 (also the data of the effect of post-processing)
python simulations/bit_flip_cln/logical_error_rate_gsm.py --distances 3 5 7 9 11 13 15 --error-rates 0.005 0.02 0.001 --shots 100000 --no-post-processing --output simulations/bit_flip_cln/Fig-10-LER/data/gsm.csv
python simulations/bit_flip_cln/logical_error_rate_gsm.py --distances 3 5 7 9 11 13 15 --error-rates 0.005 0.02 0.001 --shots 100000 --output simulations/bit_flip_cln/Fig-10-LER/data/gsm_pp.csv
python simulations/bit_flip_cln/logical_error_rate_ism.py --distances 3 5 7 9 11 13 15 --error-rates 0.005 0.02 0.001 --shots 100000 --no-post-processing --output simulations/bit_flip_cln/Fig-10-LER/data/ism.csv
python simulations/bit_flip_cln/logical_error_rate_ism.py --distances 3 5 7 9 11 13 15 --error-rates 0.005 0.02 0.001 --shots 100000 --output simulations/bit_flip_cln/Fig-10-LER/data/ism_pp.csv

# threshold crossings (also the data of the threshold fit)
python simulations/bit_flip_cln/logical_error_rate_gsm.py --distances 3 5 7 9 11 13 15 17 19 --error-rates 0.010 0.0181 0.0005 --shots 500000 --no-post-processing --output simulations/bit_flip_cln/Fig-threshold-crossings/data/gsm.csv
python simulations/bit_flip_cln/logical_error_rate_gsm.py --distances 3 5 7 9 11 13 15 17 19 --error-rates 0.011 0.0186 0.0005 --shots 500000 --output simulations/bit_flip_cln/Fig-threshold-crossings/data/gsm_pp.csv
python simulations/bit_flip_cln/logical_error_rate_ism.py --distances 3 5 7 9 11 13 15 17 19 --error-rates 0.005 0.0171 0.0005 --shots 500000 --no-post-processing --output simulations/bit_flip_cln/Fig-threshold-crossings/data/ism.csv
python simulations/bit_flip_cln/logical_error_rate_ism.py --distances 3 5 7 9 11 13 15 17 19 --error-rates 0.010 0.0181 0.0005 --shots 500000 --output simulations/bit_flip_cln/Fig-threshold-crossings/data/ism_pp.csv
