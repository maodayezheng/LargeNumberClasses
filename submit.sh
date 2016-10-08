#!/usr/bin/env bash
FOLDER=${HOME}/models/large/small
BS=100
EPOCHS=1
gru_dim=(128 256 512)
lr=(0.05 0.01 0.005 0.001)
models=(NEG IMP BAL BER ALEX)
for gd in "${gru_dim[@]}"
do
    for i in `seq 0 5`
    do
        for j in `seq 0 5`
        do
            for m in "${models[@]}"
            do
            lrg=${lr[i]}
            lre=${lr[j]}
            SUB_FILE=${FOLDER}/sub/${m}_${gd}_${i}_${j}.sh
            echo "#!/bin/bash
source \${HOME}/vars.sh
# Set up folders and files
FOLDER=${FOLDER}/${m}_${gd}_${i}_${j}
mkdir -p \${FOLDER}
MODEL=${m}
OF=\${FOLDER}/out.txt
EF=\${FOLDER}/err.txt
# Set up command
EXE=\$HOME/python_modules/LargeNumberClasses/bin/experiments.py
COMMAND=\"python3 -u \${EXE} \${MODEL} \${FOLDER} -bs ${BS} -e ${EPOCHS} -gd ${gd} -lrg ${lrg} -lre ${lre} \"
# Set up theano
export THEANO_FLAGS=\"base_compiledir=\${HOME}/jobs/large/${m}/${gd}/${i}/${j}\"
# Echo some useful stuff
echo \"Job ID: \${LSB_BATCH_JID}\" > \${OF}
echo \"My PID: \$\$\" >> \${OF}
echo \"Running command '\${COMMAND} 1>>OF 2>EF'\" >> \${OF}
# Running the command
\${COMMAND} 1>>\${OF} 2>\${EF}
echo \"Finished\" >> \${OF}
# rm -r \${COMPILE_DIR}
" > ${SUB_FILE}
                        echo "bsub -q emerald-k80 -J large/${m}/${gd}/${i}/${j} -o $HOME/jobs/%J.log -e $HOME/jobs/%J.err -W 48:00 < ${SUN_FILE}"
            done
        done
    done
done

