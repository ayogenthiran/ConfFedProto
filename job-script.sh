#!/bin/bash
#
#SBATCH --account=def-kgroling
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=15:0:0    
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=ytelila@uwo.ca
#SBATCH --mail-type=ALL


cd /home/joet/projects/def-kgroling/joet/u-fedproto/
module purge

module load python/3.12 scipy-stack cuda cudnn
module load arrow/16.1.0

source ~/UWO/bin/activate

# MNIST
 # iid
python train.py -device cuda -ufedproto False --data mnist --isiid True --pfl False
python train.py -device cuda -ufedproto False --data mnist --isiid False --pfl False

python train.py -device cuda -ufedproto True --data mnist --isiid True --pfl False
python train.py -device cuda -ufedproto True --data mnist --isiid False --pfl False

python train.py -device cuda -ufedproto False --data mnist --isiid True --pfl True
python train.py -device cuda -ufedproto False --data mnist --isiid False --pfl True


# CIFAR10
 # iid
python train.py -device cuda -ufedproto False --data cifar10 --isiid True --pfl False
python train.py -device cuda -ufedproto False --data cifar10 --isiid False --pfl False

python train.py -device cuda -ufedproto True --data cifar10 --isiid True --pfl False
python train.py -device cuda -ufedproto True --data cifar10 --isiid False --pfl False

python train.py -device cuda -ufedproto False --data cifar10 --isiid True --pfl True
python train.py -device cuda -ufedproto False --data cifar10 --isiid False --pfl True