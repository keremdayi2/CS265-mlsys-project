# generic gpu
salloc -p gpu_test -t 0-01:00 --mem 16G --gres=gpu:1

# get h-100
salloc -p seas_gpu -t 0-01:00 --mem 32G --gpus=nvidia_h100_80gb_hbm3

salloc -p seas_gpu -t 0-01:00 --mem 32G --gpus=nvidia_h100_80gb_hbm3
salloc -p seas_gpu -t 0-01:00 --mem 16G --gpus=nvidia_a100-sxm4-40gb




mamba activate cs265

python benchmarks.py --model_idx=0 --batch_idx=3 --mem_cap=10 > logs/benchmark.out 2> logs/stderr.out

python benchmarks.py --model_idx=2 --batch_idx=4 --mem_cap=10 > logs/benchmark_2_4.out 