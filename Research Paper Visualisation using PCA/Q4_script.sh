#!/bin/bash
#$ -P rse-com6012
#$ -l h_rt=04:00:00  
#$ -pe smp 6
#$ -l rmem=26G
#$ -o ./Output/Q4_output.txt  
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M nsmathew1@shef.ac.uk 
#$ -m ea #Email you when it finished or aborted
#$ -cwd 

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit --driver-memory 26g --executor-memory 20g --master local[6] ./Q4_code.py