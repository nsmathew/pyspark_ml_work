#!/bin/bash
#$ -P rse-com6012
#$ -l h_rt=00:30:00  
#$ -pe smp 4
#$ -l rmem=4G 
#$ -o ./Output/Q1_output.txt  
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M nsmathew1@shef.ac.uk 
#$ -m ea #Email you when it finished or aborted
#$ -cwd 

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit --driver-memory 4g --master local[4] ./Q1_code.py