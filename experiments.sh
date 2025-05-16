# resnet50 experiments
python benchmarks.py --model_idx=2 --batch_idx=0 --mem_ratio=0.9 > logs/benchmark_2_0.9_0.out
python benchmarks.py --model_idx=2 --batch_idx=1 --mem_ratio=0.9 > logs/benchmark_2_0.9_1.out
python benchmarks.py --model_idx=2 --batch_idx=2 --mem_ratio=0.9 > logs/benchmark_2_0.9_2.out
python benchmarks.py --model_idx=2 --batch_idx=3 --mem_ratio=0.9 > logs/benchmark_2_0.9_3.out
python benchmarks.py --model_idx=2 --batch_idx=4 --mem_ratio=0.9 > logs/benchmark_2_0.9_3.out

python benchmarks.py --model_idx=2 --batch_idx=0 --mem_ratio=0.7 > logs/benchmark_2_0.7_0.out
python benchmarks.py --model_idx=2 --batch_idx=1 --mem_ratio=0.7 > logs/benchmark_2_0.7_1.out
python benchmarks.py --model_idx=2 --batch_idx=2 --mem_ratio=0.7 > logs/benchmark_2_0.7_2.out
python benchmarks.py --model_idx=2 --batch_idx=3 --mem_ratio=0.7 > logs/benchmark_2_0.7_3.out
python benchmarks.py --model_idx=2 --batch_idx=4 --mem_ratio=0.7 > logs/benchmark_2_0.7_3.out

python benchmarks.py --model_idx=2 --batch_idx=0 --mem_ratio=0.5 > logs/benchmark_2_0.5_0.out
python benchmarks.py --model_idx=2 --batch_idx=1 --mem_ratio=0.5 > logs/benchmark_2_0.5_1.out
python benchmarks.py --model_idx=2 --batch_idx=2 --mem_ratio=0.5 > logs/benchmark_2_0.5_2.out
python benchmarks.py --model_idx=2 --batch_idx=3 --mem_ratio=0.5 > logs/benchmark_2_0.5_3.out
python benchmarks.py --model_idx=2 --batch_idx=4 --mem_ratio=0.5 > logs/benchmark_2_0.5_3.out

# # transformer experiments
python benchmarks.py --model_idx=0 --batch_idx=0 --mem_ratio=0.9 > logs/benchmark_0_0.out
python benchmarks.py --model_idx=0 --batch_idx=1 --mem_ratio=0.9 > logs/benchmark_0_1.out
python benchmarks.py --model_idx=0 --batch_idx=2 --mem_ratio=0.9 > logs/benchmark_0_2.out
python benchmarks.py --model_idx=0 --batch_idx=3 --mem_ratio=0.9 > logs/benchmark_0_3.out

python benchmarks.py --model_idx=0 --batch_idx=0 --mem_ratio=0.7 > logs/benchmark_0_0.out
python benchmarks.py --model_idx=0 --batch_idx=1 --mem_ratio=0.7 > logs/benchmark_0_1.out
python benchmarks.py --model_idx=0 --batch_idx=2 --mem_ratio=0.7 > logs/benchmark_0_2.out
python benchmarks.py --model_idx=0 --batch_idx=3 --mem_ratio=0.7 > logs/benchmark_0_3.out

python benchmarks.py --model_idx=0 --batch_idx=0 --mem_ratio=0.5 > logs/benchmark_0_0.out
python benchmarks.py --model_idx=0 --batch_idx=1 --mem_ratio=0.5 > logs/benchmark_0_1.out
python benchmarks.py --model_idx=0 --batch_idx=2 --mem_ratio=0.5 > logs/benchmark_0_2.out
python benchmarks.py --model_idx=0 --batch_idx=3 --mem_ratio=0.5 > logs/benchmark_0_3.out