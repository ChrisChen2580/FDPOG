#!/bin/bash

for seed in {0..4}
do
  dir="mnist_$seed"
  
 # echo "$seed nonpriv"
 # python3 main.py --dataset=mnist --method=regular --config group_ratios=-1,-1,-1,-1,-1,-1,-1,-1,0.09,-1 --config make_valid_loader=0 --config net=cnn --config hidden_channels=32,16 --config lr=0.01 --config train_batch_size=256 --config valid_batch_size=256 --config test_batch_size=256 --config max_epochs=60 --config logdir=$dir/mnist_nonpriv --config seed=$seed

 echo "$seed dpsur"
 python3 main.py --dataset=mnist --method=dpsur --config group_ratios=-1,-1,-1,-1,-1,-1,-1,-1,0.09,-1 --config make_valid_loader=0 --config net=cnn --config hidden_channels=32,16 --config lr=0.01 --config train_batch_size=256 --config valid_batch_size=256 --config test_batch_size=256 --config max_epochs=60 --config delta=1e-6 --config noise_multiplier=0.7 --config l2_norm_clip=1 --config logdir=$dir/mnist_dpsur --config seed=$seed --config C_v=3.0 --config sigma_v=1.5

#  echo "$seed dpsgd"
#  python3 main.py --dataset=mnist --method=dpsgd --config group_ratios=-1,-1,-1,-1,-1,-1,-1,-1,0.09,-1 --config make_valid_loader=0 --config net=cnn --config hidden_channels=32,16 --config lr=0.01 --config train_batch_size=256 --config valid_batch_size=256 --config test_batch_size=256 --config max_epochs=60 --config delta=1e-6 --config noise_multiplier=0.8 --config l2_norm_clip=1 --config logdir=$dir/mnist_dpsgd --config seed=$seed

#  echo "$seed dpsgd-f"
#  python3 main.py --dataset=mnist --method=dpsgd-f --config group_ratios=-1,-1,-1,-1,-1,-1,-1,-1,0.09,-1 --config make_valid_loader=0 --config net=cnn --config hidden_channels=32,16 --config lr=0.01 --config train_batch_size=256 --config valid_batch_size=256 --config test_batch_size=256 --config max_epochs=60 --config delta=1e-6 --config noise_multiplier=0.8 --config base_max_grad_norm=1 --config counts_noise_multiplier=8 --config logdir=$dir/mnist_dpsgdf --config seed=$seed

#  echo "$seed dpsgd-global-adapt"
#  python3 main.py --dataset=mnist --method=dpsgd-global-adapt --config group_ratios=-1,-1,-1,-1,-1,-1,-1,-1,0.09,-1 --config make_valid_loader=0 --config net=cnn --config hidden_channels=32,16 --config train_batch_size=256 --config valid_batch_size=256 --config test_batch_size=256 --config max_epochs=60 --config delta=1e-6 --config noise_multiplier=0.8 --config l2_norm_clip=1 --config strict_max_grad_norm=50 --config lr=0.01 --config logdir=$dir/mnist_dpsgdg_adapt --config threshold=0.7 --config seed=$seed
 
#    echo "$seed dp-is-sgd"
#  python3 main.py --dataset=mnist --method=dp-is-sgd --config group_ratios=-1,-1,-1,-1,-1,-1,-1,-1,0.09,-1 --config make_valid_loader=0 --config net=cnn --config hidden_channels=32,16 --config lr=0.01 --config train_batch_size=256 --config valid_batch_size=256 --config test_batch_size=256 --config max_epochs=60 --config delta=1e-6 --config l2_norm_clip=1 --config weighted_sampling=1 --config sample_rate=0.00467 --config logdir=$dir/mnist_dp_is_sgd --config seed=$seed --config epsilon=5.9

#   echo "$seed FDPOG"
#  python3 main.py --dataset=mnist --method=fdp --config group_ratios=-1,-1,-1,-1,-1,-1,-1,-1,0.09,-1 --config make_valid_loader=0 --config net=cnn --config hidden_channels=32,16 --config lr=0.01 --config train_batch_size=256 --config valid_batch_size=256 --config test_batch_size=256 --config max_epochs=60 --config delta=1e-6 --config noise_multiplier=2 --config l2_norm_clip=1 --config weighted_sampling=2 --config sample_rate=0.00467 --config logdir=$dir/mnist_fdpog --config seed=$seed

done

# for seed in 0
# do
#   dir="mnist_nsgd_$seed"
  
#   echo "$seed nonpriv"
#  python3 main.py --dataset=mnist --method=regular --config group_ratios=-1,-1,-1,-1,-1,-1,-1,-1,0.09,-1 --config make_valid_loader=0 --config net=MNIST_CNN --config hidden_channels=32,16 --config lr=0.005 --config optimizer=momentum --config momentum=0.9 --config train_batch_size=512 --config valid_batch_size=512 --config test_batch_size=512 --config max_epochs=40 --config logdir=$dir/mnist_nonpriv --config seed=$seed

#   echo "$seed auto-s"
#  python3 main.py --dataset=mnist --method=dpnsgd --config group_ratios=-1,-1,-1,-1,-1,-1,-1,-1,0.09,-1 --config make_valid_loader=0 --config net=MNIST_CNN --config lr=0.5 --config optimizer=momentum --config momentum=0.9 --config train_batch_size=512 --config valid_batch_size=512 --config test_batch_size=512 --config max_epochs=40 --config delta=1e-5 --config noise_multiplier=1 --config l2_norm_clip=0.1 --config logdir=$dir/mnist_auto_s --config seed=$seed --config epsilon=3
 
#   echo "$seed DP-IS-SGD"
#  python3 main.py --dataset=mnist --method=dp-is-sgd --config group_ratios=-1,-1,-1,-1,-1,-1,-1,-1,0.09,-1 --config make_valid_loader=0 --config net=MNIST_CNN --config lr=0.5 --config optimizer=momentum --config momentum=0.9 --config train_batch_size=512 --config valid_batch_size=512 --config test_batch_size=512 --config max_epochs=40 --config delta=1e-5 --config noise_multiplier=2 --config l2_norm_clip=0.1 --config weighted_sampling=1 --config logdir=$dir/mnist_DP_IS_SGD --config seed=$seed --config epsilon=3

#    echo "$seed FDPOG"
#  python3 main.py --dataset=mnist --method=fdp --config group_ratios=-1,-1,-1,-1,-1,-1,-1,-1,0.09,-1 --config make_valid_loader=0 --config net=MNIST_CNN --config lr=0.5 --config optimizer=momentum --config momentum=0.9 --config train_batch_size=512 --config valid_batch_size=512 --config test_batch_size=512 --config max_epochs=40 --config delta=1e-5 --config noise_multiplier=2 --config l2_norm_clip=0.1 --config weighted_sampling=2 --config gamma=0.4 --config logdir=$dir/mnist_fdpog --config seed=$seed --config epsilon=3

# done
