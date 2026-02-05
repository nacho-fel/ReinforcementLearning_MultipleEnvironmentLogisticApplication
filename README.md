# Reinforcement Learning: Applying core RL concepts across multiple environments

This repository contains a final Reinforcement Learning project where RL techniques are applied to multiple environments with different objectives and difficulty levels. The work includes:

- Continuous navigation using SARSA with Tile Coding for discretization and generalization.
- A warehouse task using a DQN agent (Stable-Baselines3), including feature engineering, reward design, and training tracking with TensorBoard.

---

## System Overview

The project is organized into two main parts:

### 1) Navigation with Tile Coding + SARSA
Given a continuous state space, Tile Coding is used to convert states into a sparse (binary) representation that enables learning with a linear function approximator. On top of this encoding, an on-policy SARSA agent is trained with a decaying epsilon-greedy exploration strategy to learn a policy that reaches the goal while avoiding collisions.

Reward shaping (summary):
- Reaching the goal: +1
- Collision: -1
- Moving/step progress: 0

### 2) Warehouse task with DQN
A warehouse-like environment is used, with fixed shelves and a goal that depends on the scenario. DQN is chosen due to its ability to learn non-linear approximations without explicit discretization and for convenient implementation with Stable-Baselines3.

This part includes:
- Hyperparameter selection/tuning (MLP 2x128, exploration schedule, gamma, learning rate, replay buffer, batch size, etc.).
- Feature engineering (shelf-related information, distances, normalized direction to the goal, etc.).
- Reward design and anti-loop penalties (memory of recent actions/states and penalties for repetitive behavior).
- TensorBoard tracking to monitor rewards, episode lengths, success rate, and exploration.

---

## Repository Structure

```bash
RL-Final-Project/

├── navigation_tile_coding/            # Part 1: Continuous navigation
│   ├── sarsa_agent.py                 # SARSA agent implementation
│   ├── tile_coding.py                 # Tile Coding (sparse features)
│   ├── env_navigation.py              # Navigation environment (walls/obstacles/goal)
│   └── train_sarsa.py                 # Training and evaluation script
│
├── warehouse_dqn/                     # Part 2: Warehouse task
│   ├── env_warehouse.py               # Warehouse environment (scenario-dependent goals)
│   ├── features.py                    # Feature engineering (distances, direction, shelves...)
│   ├── rewards.py                     # Reward function + anti-loop logic
│   ├── train_dqn.py                   # DQN training (Stable-Baselines3)
│   └── evaluate.py                    # Evaluation and metrics
│
├── logs/                              # TensorBoard logs (generated during training)
├── results/                           # Plots/outputs (generated during training)
│
└── Informe PF RL.pdf                  # Project report
