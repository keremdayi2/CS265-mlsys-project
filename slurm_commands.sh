salloc -p gpu_test -t 0-01:00 --mem 16G --gres=gpu:1

mamba activate cs265

python benchmarks.py --model_idx=2 --batch_idx=3 --mem_cap=10 > logs/benchmark.out