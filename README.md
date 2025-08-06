# NanoKimi-vs.-NanoGPT

## Overview

This project compares two transformer-based language models, NanoKimi and NanoGPT, to evaluate the advantages of the Muon optimizer used in NanoKimi against the standard AdamW optimizer used in NanoGPT. NanoKimi incorporates a Mixture of Experts (MoE) architecture, optimized with Muon for its hidden layers, while NanoGPT uses a simpler feedforward architecture with AdamW. The comparison focuses on performance (validation loss, perplexity), training/inference speed, memory usage, and parameter efficiency, using a character-level language modeling task on a provided text dataset (input.txt).

The code trains both models, measures key metrics, and visualizes results in a comparison graph (comparison_graph.png). The results highlight NanoKimi’s superior performance due to the Muon optimizer and MoE.
-------
## Setup Instructions

1). Prerequisites
Python 3.8+
PyTorch (torch)
NumPy (numpy)
Matplotlib (matplotlib)
psutil (psutil)
A text file named input.txt containing the training dataset, which is provided in dataset floder.
CUDA-enabled GPU (optional, falls back to CPU if unavailable)

2). Installation

1. Clone the repository:
git clone <repository-url>
cd <repository-directory>

2. Install dependencies:
pip install torch numpy matplotlib psutil

3. Ensure input.txt is in the project directory with sufficient text data for training.

3). Running the Code
Execute the script to train both models and generate the comparison graph:
python model_comparison.py
The script outputs training progress, final metrics, and saves a visualization to comparison_graph.png.
------
## Code Structure

The code is organized into the following components:
1). Data Loading and Preprocessing:
    1. Loads input.txt and creates a character-level vocabulary.
    2. Splits data into 90% training and 10% validation sets.
    3. Implements get_batch for generating input-target pairs.



2). Model Architecture:
    1. NanoGPT: Standard transformer with multi-head self-attention and feedforward layers, optimized with AdamW.
    2. NanoKimi: Transformer with MoE layers (4 experts) and multi-head self-attention, using Muon for hidden layers and AdamW for others.
    3. Shared components: SwiGLU activation, Head, MultiHeadAttention, Block, and LanguageModel classes.



3). Muon Optimizer:
Custom TrueMuon optimizer with matrix_sign for directional gradient updates, applied to NanoKimi’s hidden layers.

4). Training and Evaluation:
    1. Trains both models for 10,000 iterations, computing validation loss every 100 iterations.
    2. Measures training time, inference time, memory usage, and parameter efficiency.



Visualization(Output):
Generates bar plots comparing validation loss, training time, memory usage, and inference time.
<img width="1016" height="375" alt="image" src="https://github.com/user-attachments/assets/76ca595c-8c85-4004-9257-1a298de65049" />
<img width="800" height="1200" alt="comparison_graph" src="https://github.com/user-attachments/assets/f96e95e9-e93f-444c-ada0-c2b65365ee50" />

------
## NanoKimi K2: Design and Advantages
NanoKimi K2 (referred to as NanoKimi in the code) is designed to leverage the Muon optimizer and Mixture of Experts (MoE) architecture for enhanced performance in language modeling. Key features include:

1). Mixture of Experts (MoE) Architecture
Structure: Each transformer block in NanoKimi replaces the standard feedforward layer with an MoELayer containing 4 experts. A gating network (self.gate) computes softmax scores to weight expert outputs dynamically.



Implementation:

class MoELayer(nn.Module):
    def __init__(self, dim, num_experts=4):
        super().__init__()
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts)
    def forward(self, x):
        scores = F.softmax(self.gate(x), dim=-1)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=-1)
        scores = scores.unsqueeze(-2)
        return (expert_outs * scores).sum(-1)



Benefit: MoE allows NanoKimi to specialize computations across experts, improving modeling capacity and achieving lower validation loss compared to NanoGPT’s feedforward layers.

2). Muon Optimizer
Purpose: The TrueMuon optimizer applies directional updates using matrix_sign to stabilize training for high-dimensional parameters in NanoKimi’s MoE and attention layers.

Implementation:

class TrueMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, beta=0.9, weight_decay=0.01):
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
        super().__init__(params, defaults)
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.grad.ndim < 2:
                    continue
                grad = p.grad
                state = self.state[p]
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p)
                m = state['momentum']
                m.mul_(beta).add_(grad, alpha=(1-beta))
                X = beta * m + grad
                O = matrix_sign(X)
                if wd != 0:
                    O = O.add(p, alpha=wd)
                p.add_(O, alpha=-lr)



3). Advantages:
Improved Convergence: Muon’s directional updates stabilize training for MoE layers, leading to lower validation loss (1.6756 vs. 1.7285).
Sparse Optimization: Well-suited for MoE’s sparse gradients, enhancing performance over AdamW.
Trade-off: Increased computational cost due to matrix_sign (5 matrix operations), resulting in slower training (0.1264s vs. 0.0508s).

4). Key Parameters
Embedding Dimension: 64 (n_embd)
Layers: 4 (n_layer)
Heads: 4 (n_head)
Experts: 4 (num_experts)
Parameter Count: 0.28M, higher than NanoGPT’s 0.21M due to MoE layers.
----
## Results

The comparison results highlight NanoKimi’s advantages and trade-offs:
    1. Validation Loss: NanoKimi (1.6756) outperforms NanoGPT (1.7285), indicating better language modeling accuracy.
    2. Perplexity: NanoKimi (5.34) vs. NanoGPT (5.63), reflecting Muon’s role in achieving better predictions.
    3. Training Time: NanoKimi (0.1264s) is ~2.5x slower than NanoGPT (0.0508s) due to Muon’s computations and MoE complexity.
    4. Inference Time: NanoKimi (0.0081s) is ~1.35x slower than NanoGPT (0.0060s).
    5. Memory Usage: NanoKimi (0.01 MB) uses less memory than NanoGPT (0.03 MB), benefiting from MoE’s sparse activation.
    6. Parameter Efficiency: NanoKimi (19.21) is less efficient than NanoGPT (26.85) due to higher parameter count.
    7. Parameter Count: NanoKimi (0.28M) vs. NanoGPT (0.21M).

* Key Insight: The Muon optimizer enables NanoKimi to achieve superior performance (lower loss/perplexity) by stabilizing MoE training, but at the cost of slower training and inference. Lower memory usage makes NanoKimi suitable for resource-constrained settings.
-----
## Limitations and Future Work

1. Enhancements: Add top-k gating to MoELayer for faster computation and expert pruning for dynamic parameter reduction.
2. Resource Limitation: Due to computational constraints (e.g., limited GPU memory and processing power), the model could not be developed to run at full capacity. This restricted the scale of NanoKimi’s Mixture of Experts (MoE) architecture and the number of training iterations, potentially limiting its performance
Contributing
----
Contributions are welcome! Please submit issues or pull requests for bug fixes, optimizations, or additional features.
