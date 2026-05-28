#!/bin/bash

for i in {0..15}; do
    for j in {0..15}; do
        echo "Running with i=$i, j=$j"
        ./skinny1 "$i" "$j" && python3 step_2_process_ciphertext_2.py && python3 step_3_verify.py "$i" "$j" && rm ./cipher_txt_npy_8r/* 
    done
done
