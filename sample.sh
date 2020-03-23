#!/bin/bash

#SBATCH --output=output.txt

python3 -u scripts/main_gw_mli.py --task conneau --entreg 10 --maxiter 10 --maxs 5000 --tree s-tree2 --lang_space fr --results_path out9/no-proj-new --option barycenter --initlang random --dim 2times --convergeiter 10



