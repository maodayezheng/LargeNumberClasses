#!/usr/bin/env bash
# Pre set up of what to run
DATA_LOCATION=$HOME/python_modules/LargeNumberClasses/bin/ProcessedData
PREFIX=small
# DATA_LOCATION=$HOME/python_modules/LargeNumberClasses/bin/ProcessedFull
# PREFIX=full
DISTORTIONS=(0.0 0.75 1.0)
DI=0
d=${DISTORTIONS[DI]}

FOLDER=${HOME}/models/large/${PREFIX}_${DI}
gru_dim=(128 256 512)
lr=(0.02 0.01 0.005 0.001)
models=(NEG IMP BLA BER ALEX)
for gd in "${gru_dim[@]}"
do
    for i in `seq 0 3`
    do
        for j in `seq 0 3`
        do
            for m in "${models[@]}"
            do
                lrg=${lr[i]}
                lre=${lr[j]}
                PARAM_FILE=${FOLDER}/${m}_${gd}_${i}_${j}/param.npz
                if [ ! -f ${PARAM_FILE} ]
                then
                    echo "Job ${m}_${gd}_${i}_${j} has not finished"
                fi
            done
        done
    done
done