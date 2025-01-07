#!/bin/bash
#
#$ -t 1-3233
#$ -N t_fetch_val
#$ -o out.stdout
#$ -e errors.stderr
#$ -cwd
#$ -l h_vmem=490G
#$ -l h=nlpgrid20
BANK_NAME=Valley_National_Bancorp conda run twitter_fetcher.py $SGE_TASK_ID