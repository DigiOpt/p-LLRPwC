#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -N TCH_ANCHOR
#$ -q all.q
#$ -pe smp 16
#$ -j y
#$ -o Logs/Tcheby_Anchors_$JOB_ID.log

mkdir -p Logs Outputs

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "========================================"
echo "TCHEBYCHEFF ANCHOR JOB"
echo "Host      : $(hostname)"
echo "Job ID    : $JOB_ID"
echo "NSLOTS    : $NSLOTS"
echo "Start     : $(date)"
echo "========================================"

python3 RUN_TCHEBY_ANCHORS_PROFILE_HPC_PARALLEL.py

status=$?

echo "========================================"
echo "Exit code : $status"
echo "End       : $(date)"
echo "========================================"

exit $status
