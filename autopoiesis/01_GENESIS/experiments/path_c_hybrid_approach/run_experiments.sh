#!/bin/bash
# Run all Path C experiments

# Activate virtual environment
source /Users/say/Documents/GitHub/ai/08_GENESIS/venv/bin/activate
cd /Users/say/Documents/GitHub/ai/08_GENESIS/experiments/path_c_hybrid_approach

# Detect device
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
elif python -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    DEVICE="mps"
else
    DEVICE="cpu"
fi

echo "Using device: $DEVICE"
echo "======================================"

# Run MNIST ablation study
echo "Starting MNIST ablation study..."
python experiment.py \
    --dataset mnist \
    --n_trials 5 \
    --n_epochs 20 \
    --device $DEVICE \
    --save_dir ./results

# Run MNIST robustness evaluation
echo ""
echo "Starting MNIST robustness evaluation..."
python robustness.py \
    --dataset mnist \
    --n_epochs 20 \
    --device $DEVICE \
    --save_dir ./results

# Generate visualizations
echo ""
echo "Generating visualizations..."
ABLATION_FILE=$(ls -t ./results/ablation_mnist_*.json 2>/dev/null | head -1)
ROBUST_FILE=$(ls -t ./results/robustness_mnist_*.json 2>/dev/null | head -1)

if [ -n "$ABLATION_FILE" ]; then
    python visualize.py --ablation "$ABLATION_FILE" --robustness "$ROBUST_FILE" --output ./results
fi

echo ""
echo "======================================"
echo "MNIST experiments complete!"
echo "Results saved to ./results/"

# Optional: Run CIFAR-10 (uncomment if needed)
# echo ""
# echo "Starting CIFAR-10 experiments..."
# python experiment.py \
#     --dataset cifar10 \
#     --n_trials 5 \
#     --n_epochs 30 \
#     --device $DEVICE \
#     --save_dir ./results
