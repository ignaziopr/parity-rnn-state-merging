# Streaming‐Parity Paper Replication

## Purpose

This project reproduces the key empirical findings from **“Algorithm Development in Neural Networks: Insights from the Streaming Parity Task”**.

- **Demonstrate** how a simple RNN trained on short binary‐parity sequences first memorizes them (low train loss, chance val loss), then undergoes a sudden phase‐transition—via implicit state‐merger—to perfect generalization on arbitrarily long sequences.
- **Extract** and **minimize** the learned DFA via clustering and Hopcroft’s algorithm.
- **Empirically verify** the theory’s merger‐threshold predictions (loss curves, heatmaps, ODE fits).

## Setup

1. **Clone this repository**

   ```bash
   git clone <repo_url>
   cd streaming_parity_replication
   ```

2. **Create & activate a Conda environment**

   ```
   conda create -n streaming_parity python=3.9 -y
   conda activate streaming_parity
   ```

3. **Install dependencies**

   ```
   pip install -r requirements.txt
   ```

4. **Run the main experiments**

   ```
   # Train the RNN and log losses
   python train_parity.py \
   --train-max-len 10 \
   --val-lens 25 100 250 1000 2500 10000 \
   --hidden-size 100 \
   --epochs 1000 \
   --batch-size 64 \
   --lr 1e-3
   ```

5. **Extract & visualize the learned automaton**

   ```
   python extract_dfa.py \
   --model-path checkpoints/parity_rnn.pt \
   --eps 1e-2 \
   --output-graph dfa_raw.dot
   dot -Tpng dfa_raw.dot -o dfa_raw.png

   ```

## Data Collection

TODO
