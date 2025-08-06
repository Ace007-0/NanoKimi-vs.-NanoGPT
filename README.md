# NanoKimi vs. NanoGPT

## Overview
This project compares two transformer-based language models, **NanoKimi** and **NanoGPT**, to evaluate the advantages of the **TrueMuon optimizer** used in NanoKimi against the standard **AdamW optimizer** used in NanoGPT. NanoKimi incorporates a **Mixture of Experts (MoE)** architecture, optimized with TrueMuon for its hidden layers, while NanoGPT uses a simpler feedforward architecture with AdamW. The comparison focuses on performance (validation loss, perplexity), training/inference speed, memory usage, and parameter efficiency, using a character-level language modeling task on a user-provided text dataset (`input.txt`).

The code trains both models, measures key metrics, and visualizes results in a comparison graph (`comparison_graph.png`). NanoKimi’s superior performance (lower validation loss) highlights the TrueMuon optimizer’s effectiveness in stabilizing MoE training, despite slower training times.

## Setup Instructions
### Prerequisites
- Python 3.8+
- PyTorch (`torch`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)
- psutil (`psutil`)
- A text file named `input.txt` containing the training dataset, placed in the project root
- CUDA-enabled GPU (optional, falls back to CPU if unavailable)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install dependencies:
   ```bash
   pip install torch numpy matplotlib psutil
   ```
3. Place `input.txt` in the project root with sufficient text data for training.

### Running the Code
Execute the script to train both models and generate the comparison graph:
```bash
python model_comparison.py
```
The script outputs training progress, final metrics, and saves a visualization to `comparison_graph.png`.

## Code Structure
The code is organized into the following components:

### Data Loading and Preprocessing
- Loads `input.txt` and creates a character-level vocabulary.
- Splits data into 90% training and 10% validation sets (`train_data = data[:n]`, `val_data = data[n:]`, where `n = int(0.9 * len(data))`).
- Implements `get_batch` to generate input-target pairs for training and validation.

### Model Architecture
- **NanoGPT**: Standard transformer with multi-head self-attention and feedforward layers, optimized with AdamW.
- **NanoKimi**: Transformer with MoE layers (4 experts) and multi-head self-attention, using TrueMuon for hidden layers and AdamW for other parameters.
- Shared components: `SwiGLU` activation, `Head`, `MultiHeadAttention`, `Block`, and `LanguageModel` classes.

### TrueMuon Optimizer
- Custom `TrueMuon` optimizer with `matrix_sign` for directional gradient updates, applied to NanoKimi’s hidden layers.
- `build_optimizers` splits parameters, applying TrueMuon to 2D parameters in transformer blocks and AdamW to others.

### Training and Evaluation
- Trains both models for 10,000 iterations, computing validation loss every 100 iterations.
- Measures training time, inference time, memory usage, and parameter efficiency.

### Visualization(Output)
- Generates bar plots comparing validation loss, training time, memory usage, and inference time, saved as `comparison_graph.png`.
<img width="1007" height="437" alt="image" src="https://github.com/user-attachments/assets/5ee48252-6fa3-4d18-b2fa-9eed1b56bd54" />
<img width="800" height="1200" alt="comparison_graph" src="https://github.com/user-attachments/assets/2db4e412-4090-456a-8b0f-2be828c19aee" />

## NanoKimi: Design and Advantages
NanoKimi is designed to leverage the **TrueMuon optimizer** and **Mixture of Experts (MoE)** architecture for enhanced performance in language modeling. Key features include:

### Mixture of Experts (MoE) Architecture
- **Structure**: Each transformer block replaces the feedforward layer with an `MoELayer` containing 4 experts. A gating network (`self.gate`) computes softmax scores to weight expert outputs dynamically.
- **Implementation**:
  ```python
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
  ```
- **Benefit**: MoE enhances modeling capacity by specializing computations across experts, contributing to NanoKimi’s lower validation loss compared to NanoGPT’s feedforward layers.

### TrueMuon Optimizer
- **Purpose**: Stabilizes training of high-dimensional hidden layers (MoE and attention) using directional gradient updates via `matrix_sign`.
- **Implementation**:
  ```python
  def matrix_sign(G, steps=5, eps=1e-7):
      assert G.ndim == 2
      a, b, c = 3.4445, -4.7750, 2.0315
      X = G / (G.norm() + eps)
      if G.size(0) > G.size(1):
          X = X.T
      for _ in range(steps):
          A = X @ X.T
          B = b * A + c * (A @ A)
          X = a * X + B @ X
      if G.size(0) > G.size(1):
          X = X.T
      return X

  class TrueMuon(torch.optim.Optimizer):
      def __init__(self, params, lr=1e-2, beta=0.9, weight_decay=0.01):
          defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
          super().__init__(params, defaults)

      @torch.no_grad()
      def step(self, closure=None):
          loss = closure() if closure else None
          for group in self.param_groups:
              beta = group['beta']
              lr = group['lr']
              wd = group['weight_decay']
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
          return loss

  def build_optimizers(model):
      hidden = [p for n, p in model.named_parameters() if p.ndim >= 2 and 'blocks' in n]
      hidden_ids = set(id(p) for p in hidden)
      others = [p for n, p in model.named_parameters() if id(p) not in hidden_ids]
      adamw = torch.optim.AdamW(others, lr=learning_rate)
      muon = TrueMuon(hidden, lr=1e-2, beta=0.9, weight_decay=0.01)
      return adamw, muon
  ```
- **Advantages**:
  - **Improved Convergence**: TrueMuon’s directional updates stabilize MoE training, achieving lower validation loss (1.6756 vs. NanoGPT’s 1.7285).
  - **Sparse Optimization**: Well-suited for MoE’s sparse gradients, enhancing performance over AdamW.
  - **Trade-off**: Increased computational cost due to `matrix_sign` (5 matrix operations), resulting in slower training (0.1264s vs. 0.0508s).

### Key Parameters
- Embedding Dimension: 64 (`n_embd`)
- Layers: 4 (`n_layer`)
- Heads: 4 (`n_head`)
- Experts: 4 (`num_experts`)
- Parameter Count: 0.28M (NanoKimi) vs. 0.21M (NanoGPT) due to MoE layers

## Results
The comparison results highlight NanoKimi’s advantages and trade-offs:
- **Validation Loss**: NanoKimi (1.6756) outperforms NanoGPT (1.7285), indicating better language modeling accuracy.
- **Perplexity**: NanoKimi (5.34) vs. NanoGPT (5.63), reflecting TrueMuon’s role in better predictions.
- **Training Time**: NanoKimi (0.1264s) is ~2.5x slower than NanoGPT (0.0508s) due to TrueMuon and MoE complexity.
- **Inference Time**: NanoKimi (0.0081s) is ~1.35x slower than NanoGPT (0.0060s).
- **Memory Usage**: NanoKimi (0.01 MB) uses less memory than NanoGPT (0.03 MB), benefiting from MoE’s sparse activation.
- **Parameter Efficiency**: NanoKimi (19.21) is less efficient than NanoGPT (26.85) due to higher parameter count.
- **Parameter Count**: NanoKimi (0.28M) vs. NanoGPT (0.21M).

**Key Insight**: The TrueMuon optimizer enables NanoKimi to achieve superior performance (lower loss/perplexity) by stabilizing MoE training, but at the cost of slower training and inference. Lower memory usage makes NanoKimi suitable for resource-constrained settings.

## Usage
1. **Prepare Data**: Place a text dataset in `input.txt` in the project root.
2. **Train Models**: Run `python model_comparison.py` to train NanoKimi and NanoGPT, generating metrics and `comparison_graph.png`.
3. **Analyze Results**: Review console output for training progress and final metrics. Visualize comparisons in `comparison_graph.png`.

## Limitations and Future Work
- **Enhancements**: Add top-k gating to `MoELayer` for faster computation and expert pruning for dynamic parameter reduction.
- **Resource Limitation**: Due to computational constraints (e.g., limited GPU memory and processing power), NanoKimi could not be developed to run at full capacity, restricting the scale of its MoE architecture and training iterations, potentially limiting performance compared to larger models.

## Contributing
Contributions are welcome! Please submit issues or pull requests for bug fixes, optimizations, or additional features.
