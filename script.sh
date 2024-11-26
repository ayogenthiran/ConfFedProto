#!/bin/bash

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




