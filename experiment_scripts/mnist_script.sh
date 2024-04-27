#!/bin/bash

for seed in {0..4}
do
  dir="mnist_$seed"
  
 # echo "$seed nonpriv"
 # python3 main.py --dataset=mnist --method=regular --config group_ratios=-1,-1,-1,-1,-1,-1,-1,-1,0.09,-1 --config make_valid_loader=0 --config net=cnn --config hidden_channels=32,16 --config lr=0.01 --config train_batch_size=256 --config valid_batch_size=256 --config test_batch_size=256 --config max_epochs=60 --config logdir=$dir/mnist_nonpriv --config seed=$seed

 # echo "$seed dpsgd"
 # python3 main.py --dataset=mnist --method=dpsgd --config group_ratios=-1,-1,-1,-1,-1,-1,-1,-1,0.09,-1 --config make_valid_loader=0 --config net=cnn --config hidden_channels=32,16 --config lr=0.01 --config train_batch_size=256 --config valid_batch_size=256 --config test_batch_size=256 --config max_epochs=60 --config delta=1e-6 --config noise_multiplier=0.8 --config l2_norm_clip=1 --config logdir=$dir/mnist_dpsgd10 --config seed=$seed

 # echo "$seed dpnsgd"
 # python3 main.py --dataset=mnist --method=dpnsgd --config group_ratios=-1,-1,-1,-1,-1,-1,-1,-1,0.09,-1 --config make_valid_loader=0 --config net=cnn --config hidden_channels=32,16 --config lr=0.01 --config train_batch_size=256 --config valid_batch_size=256 --config test_batch_size=256 --config max_epochs=60 --config delta=1e-6 --config noise_multiplier=0.8 --config l2_norm_clip=1 --config logdir=$dir/mnist_dpnsgd --config seed=$seed

#  echo "$seed dpsgd-f"
#  python3 main.py --dataset=mnist --method=dpsgd-f --config group_ratios=-1,-1,-1,-1,-1,-1,-1,-1,0.09,-1 --config make_valid_loader=0 --config net=cnn --config hidden_channels=32,16 --config lr=0.01 --config train_batch_size=256 --config valid_batch_size=256 --config test_batch_size=256 --config max_epochs=60 --config delta=1e-6 --config noise_multiplier=0.8 --config base_max_grad_norm=1 --config counts_noise_multiplier=8 --config logdir=$dir/mnist_dpsgdf --config seed=$seed

 # echo "$seed dpsgd-global-adapt"
 # python3 main.py --dataset=mnist --method=dpsgd-global-adapt --config group_ratios=-1,-1,-1,-1,-1,-1,-1,-1,0.09,-1 --config make_valid_loader=0 --config net=cnn --config hidden_channels=32,16 --config train_batch_size=256 --config valid_batch_size=256 --config test_batch_size=256 --config max_epochs=60 --config delta=1e-6 --config noise_multiplier=0.8 --config l2_norm_clip=1 --config strict_max_grad_norm=50 --config lr=0.01 --config logdir=$dir/mnist_dpsgdg_adapt --config threshold=0.7 --config seed=$seed

  echo "$seed FDPOG"
 python3 main.py --dataset=mnist --method=IS --config group_ratios=-1,-1,-1,-1,-1,-1,-1,-1,0.09,-1 --config make_valid_loader=0 --config net=cnn --config hidden_channels=32,16 --config lr=0.01 --config train_batch_size=256 --config valid_batch_size=256 --config test_batch_size=256 --config max_epochs=60 --config delta=1e-6 --config noise_multiplier=2 --config l2_norm_clip=1 --config weighted_sampling=True --config sample_rate=0.00467 --config logdir=$dir/mnist_fdpog --config seed=$seed

done
