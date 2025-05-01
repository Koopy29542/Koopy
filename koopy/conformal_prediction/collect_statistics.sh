#!/bin/bash

# Define the lists of arguments
arg1_list=("linear" "koopman" "trajectron" "gp" "eigen")
arg2_list=("adaptive" "split")

# Loop through all combinations of arguments
for arg1 in "${arg1_list[@]}"; do
    for arg2 in "${arg2_list[@]}"; do
        echo "Running: python main.py --model $arg1 --cp $arg2"
        python main.py --model "$arg1" --cp "$arg2"
    done
done

