#!/bin/bash

files_to_run=(
    # "simple_demonstration.py"
    # rotation_demonstration-Adam_01_20.py
    # rotation_demonstration-Adam_05_20.py
    # rotation_demonstration-Adam_10_01.py
    # rotation_demonstration-Adam_10_05.py
    # rotation_demonstration-Adam_10_10.py
    # rotation_demonstration-Adam_10_15.py
    # rotation_demonstration-Adam_10_20.py
    # rotation_demonstration-Adam_15_20.py
    # rotation_demonstration-Adam_20_01.py
    # rotation_demonstration-Adam_20_05.py
    # rotation_demonstration-Adam_20_10.py
    # rotation_demonstration-Adam_20_15.py
    # rotation_demonstration-Adam-p0.py
    # rotation_demonstration-Adam.py
    variable_L_exp_20_10.py
    variable_L_exp.py
)

for file in "${files_to_run[@]}"; do
    echo $(pwd)
    echo "$file"
    python3 "$file"
done