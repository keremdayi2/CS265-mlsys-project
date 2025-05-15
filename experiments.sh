# transformer experiments
python benchmarks.py --model_idx=0 --batch_idx=0 --mem_cap=10 > logs/benchmark_0_0.out
python benchmarks.py --model_idx=0 --batch_idx=1 --mem_cap=10 > logs/benchmark_0_1.out
python benchmarks.py --model_idx=0 --batch_idx=2 --mem_cap=10 > logs/benchmark_0_2.out
python benchmarks.py --model_idx=0 --batch_idx=3 --mem_cap=10 > logs/benchmark_0_3.out

# resnet50 experiments
python benchmarks.py --model_idx=2 --batch_idx=0 --mem_cap=10 > logs/benchmark_2_0.out
python benchmarks.py --model_idx=2 --batch_idx=1 --mem_cap=10 > logs/benchmark_2_1.out
python benchmarks.py --model_idx=2 --batch_idx=2 --mem_cap=10 > logs/benchmark_2_2.out
python benchmarks.py --model_idx=2 --batch_idx=3 --mem_cap=20 > logs/benchmark_2_3.out