#Run with current environment (-V) and in the current directory (-cwd)
#$ -V -cwd

#Request some time- min 15 mins - max 48 hours
#$ -l h_rt=03:00:00

#Request GPU
#$ -l coproc_v100=1

#Get email at start and end of the job
#$ -m be

#Now run the job
# module load python/anaconda3
module load cuda
# module load pytorch

python models/train_manet_multiclass.py 1
