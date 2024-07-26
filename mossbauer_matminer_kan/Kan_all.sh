#!/bin/bash

predict_items=('mm' 'efg' 'rto' 'eta' 'hff')
steps_values=(50 100 200)
k_values=(3 5)
grids_values=('3,10' '3,10,20' '3,10,20,50' '3,10,20,50,100')
model_width_values=('4,2,1' '4,4,1' '4,2,2,1' '4,4,2,1' '4,8,2,1' '4,4,2,1,1')

wait_for_jobs() {
    # shellcheck disable=SC2046
    while [ $(jobs | wc -l) -ge 20 ]; do
        sleep 1
    done
}

for model_width in "${model_width_values[@]}"; do
    for steps in "${steps_values[@]}"; do
        for k in "${k_values[@]}"; do
            for grids in "${grids_values[@]}"; do
                for predict_item in "${predict_items[@]}"; do
                    wait_for_jobs  # Check and wait for available slot
                    echo "Running script with predict_item=${predict_item}, steps=${steps}, k=${k}, grids=${grids}, model_width=${model_width}"
                    log_file="kan_${predict_item}_steps_${steps}_k_${k}_grids_${grids//,/}_model_width_${model_width//,/}.log"

                    nohup python main.py --predict_item "${predict_item}" --steps "${steps}" --k "${k}" --grids "${grids}" --model_width "${model_width}" > "${log_file}" 2>&1 &
                done
            done
        done
    done
done

wait