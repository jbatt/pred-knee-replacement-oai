#Run with current environment (-V) and in the current directory (-cwd)
#$ -V -cwd

#Request some time- min 15 mins - max 48 hours
#$ -l h_rt=06:00:00

#Request GPU
#$ -l coproc_v100=4

#Get email at start and end of the job
#$ -m be

# Set tasks 1-2
#$ -t 1-2

#Now run the job
# module load python/anaconda3
module load cuda
# module load pytorch


infile=$(sed -n -e "$SGE_TASK_ID p" config/config_unet.txt)

python train.py --model unet --hpc-flag 1 < $infile
