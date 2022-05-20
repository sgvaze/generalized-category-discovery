PYTHON='/users/sagar/miniconda3/envs/open_world_prototype/bin/python'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

${PYTHON} -m methods.clustering.extract_features --dataset cifar100 --use_best_model 'True' \
 --warmup_model_dir '/work/sagar/osr_novel_categories/metric_learn_gcd/log/(28.04.2022_|_27.530)/checkpoints/model.pt'