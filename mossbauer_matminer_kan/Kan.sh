#!/bin/bash

source activate kans

predict_items=('mm' 'efg' 'rto' 'eta' 'hff')
k_values=(3 5)
model_width_values=('4,4,2,1' '4,2,2,1' '4,4,2,1,1')

wait_for_jobs() {
    # shellcheck disable=SC2046
    while [ $(jobs | wc -l) -ge 5 ]; do
        sleep 1
    done
}

for model_width in "${model_width_values[@]}"; do
    for k in "${k_values[@]}"; do
        for predict_item in "${predict_items[@]}"; do
            wait_for_jobs  # Check and wait for available slot
            echo "Running script with predict_item=${predict_item}, k=${k}, model_width=${model_width}"
            log_file="kan_${predict_item}_k_${k}_model_width_${model_width//,/}.log"

            nohup python main.py --predict_item "${predict_item}" --k "${k}" --model_width "${model_width}" > "${log_file}" 2>&1 &
        done
    done

done

wait