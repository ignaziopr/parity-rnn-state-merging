# Streaming‐Parity Paper Replication

### By Ignazio Perez Romero | igp4@cornell.edu | https://github.com/ignaziopr

## Purpose

This project reproduces the key empirical findings from **“Algorithm Development in Neural Networks: Insights from the Streaming Parity Task”**.

- **Demonstrate** how a simple RNN trained on short binary‐parity sequences first memorizes them (low train loss, chance val loss), then undergoes a sudden phase‐transition—via implicit state‐merger—to perfect generalization on arbitrarily long sequences.
- **Extract** and **minimize** the learned DFA via clustering and Hopcroft’s algorithm.
- **Empirically verify** the theory’s merger‐threshold predictions (loss curves, heatmaps, ODE fits).

## Figures

## Setup

1. **Clone this repository**

   ```bash
   git clone <repo_url>
   cd streaming_parity_replication
   ```

2. **Create & activate a Conda environment**

   This project uses a Python virtual environment called `parity-rnn-state-merge`.

   ```bash
   python -m venv parity-rnn-state-merge
   ```

   To activate the virtual environment:

   **On macOS/Linux:**

   ```bash
   source parity-rnn-state-merge/bin/activate
   ```

   **On Windows:**

   ```bash
   parity-rnn-state-merge\Scripts\activate
   ```

   After activation, your terminal prompt should show `(parity-rnn-state-merge)` at the beginning.

   ```bash
   which python  # Should point to the venv Python (macOS/Linux)
   where python  # Should point to the venv Python (Windows)
   ```

   Install project dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   Alternative manual installation:

   ```bash
   pip install torch torchvision torchaudio
   pip install numpy matplotlib seaborn pandas
   pip install scikit-learn graphviz
   pip install jupyter notebook
   ```

   Deactivating the Environment:

   ```bash
   deactivate
   ```

## Data Creation

1. Overview
   The `data_production.ipynb` notebook generates training and validation datasets for the streaming parity task, where networks must output 0 for even parity (even number of 1s) and 1 for odd parity.

   ### The Streaming Parity Task

   - **Input**: Binary sequences of varying lengths (e.g., [1, 0, 1, 1, 0])
   - **Output**: 0 if even number of 1s, 1 if odd number of 1s
   - **Challenge**: Networks must generalize to sequences thousands of times longer than training data

   This task tests whether networks learn the underlying algorithm versus memorizing patterns.

2. Data Collection Strategy

   Training Data

   - **All possible binary sequences** from length 1 to 10
   - Total: 2¹ + 2² + ... + 2¹⁰ = 2,046 sequences
   - Exhaustive coverage ensures complete learning foundation

   Validation Data

   - **100 random sequences** at each length: 25, 50, 100, 250, 1000, 2500, 10000
   - Tests generalization to sequences much longer than training (max training = 10)
   - Fixed random seed (0) for reproducibility

3. Running

   Execute all cells in `src/data_production.ipynb`

4. Output & Format

   - `../data/training/train_data.npz`: All 2,046 training sequences
   - `../data/validation/val_data_{length}.npz`: 100 validation sequences per target length
   - **Sequences**: Arrays of 0s and 1s
   - **Labels**: Arrays (0=even, 1=odd parity)
   - **Storage**: Compressed `.npz` format

5. Importance for Replication
   A. **Complete training set**: Uses ALL short sequences (not random sampling)
   B. **Specific validation lengths**: Tests different aspects of generalization
   C. **Reproducible randomness**: Fixed seed ensures consistent results
   D. **Algorithm vs. memorization**: Design specifically tests whether networks learn the parity algorithm

## RNN Architecture & Training

1. Architecture Overview

   The ParityRNN model implements a simple recurrent neural network designed to learn the streaming parity algorithm. The architecture closely follows the paper's specifications with minor optimizations for training stability.

   **Model Components**
   RNN Layer: Single fully connected recurrent layer with 100 hidden ReLU units
   Readout Layer: Linear layer mapping hidden states to 2-dimensional output (even/odd parity)
   Input: 2-dimensional one-hot encoding (for bits 0 and 1)
   Output: 2-dimensional logits (converted to probabilities via softmax)

   **Key Architecture Details**
   pythonclass ParityRNN(nn.Module):
   def **init**(self, input_size=2, hidden_size=100, output_size=2):
   self.rnn = nn.RNN(input_size, hidden_size, nonlinearity='relu', batch_first=True)
   self.readout = nn.Linear(hidden_size, output_size)

2. Differences from Paper

   While the core architecture matches Appendix C.1, several modifications were made for training stability:
   Xavier Initialization: Gain set to 0.99 instead of 0.1 (paper's original)
   Learning Rate: 0.05 instead of 0.02 (higher for faster convergence)
   Gradient Clipping: Added clip*grad_value*(0.5) to prevent exploding gradients
   Adaptive Batch Sizes: Variable validation batch sizes based on sequence length

3. Training Configuration

   Hyperparameters
   hidden_size = 100 # Hidden ReLU units
   input_size = 2 # One-hot binary input
   output_size = 2 # One-hot parity output  
   learning_rate = 0.05 # SGD learning rate
   batch_size = 128 # Training batch size
   num_epochs = 1000 # Training epochs
   optimizer = SGD(momentum=0.0, weight_decay=0.0)
   criterion = MSELoss() # Mean squared error loss

4. Training Challenges and Solutions

   Vanishing Gradients
   Problem: Original paper's Xavier gain (0.1) caused vanishing gradients, preventing learning.
   Solution: Increased initialization gain to 0.99 while maintaining small weights for the merger effect.
   Training Instability
   Problem: Large gradients during early training caused instability.
   Solution: Added gradient clipping (clip_value=0.5) to stabilize training.
   Memory Efficiency
   Problem: Long validation sequences (up to 10,000 tokens) caused memory issues.
   Solution: Implemented adaptive batch sizes: Length ≤25: batch_size=100, Length ≤50: batch_size=50,Length >50: batch_size=25

5. Training Results

   The model demonstrates the paper's key finding - a sharp phase transition from random performance to perfect generalization:

   Epochs 1-500: Random performance (~50% accuracy, loss ~0.25)
   Epoch ~600: Sudden transition begins
   Epochs 700+: Perfect performance (100% accuracy, loss ~0.0001)

6. How to Run Training

   Execute all cells in `src/train_parity_rnn.ipynb`
