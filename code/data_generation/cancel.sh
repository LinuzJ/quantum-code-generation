#!/usr/bin/env bash

JOBID=6444641

for i in {30..299}
do
  echo "Canceling job ${JOBID}_${i}"
  scancel "${JOBID}_${i}"
done