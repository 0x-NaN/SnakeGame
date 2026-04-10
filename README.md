# 🐍 Snake RL: PPO Implementation

A high-performance Snake game environment built with Python and Pygame, featuring an integrated Reinforcement Learning (RL) agent using the **Proximal Policy Optimization (PPO)** algorithm.

---

## 🛠 Tech Stack
- **Language:** Python 3.10+
- **GUI/Game Engine:** Pygame
- **RL Framework:** Gymnasium (formerly OpenAI Gym)
- **RL Library:** Stable-Baselines3 (PyTorch backend)
- **Monitoring:** TensorBoard

---

## 🚀 Getting Started

### 1. Installation
```bash
pip install pygame gymnasium stable-baselines3 shimmy tensorboard
```

### 2. Usage Modes
The project supports three distinct execution modes via `snake_game.py`:

| Mode | Command | Description |
|------|---------|-------------|
| **Human** | `python snake_game.py` | Play manually with Arrow Keys. |
| **Train** | `python snake_game.py train` | Train a new PPO agent (Headless). |
| **Train (Vis)** | `python snake_game.py train --visualize` | Watch the agent learn in real-time. |
| **Resume** | `python snake_game.py train --continue` | Resume training from `models/ppo_snake.zip`. |
| **AI Play** | `python snake_game.py play` | Watch the best-trained model play 5 episodes. |

**Visualization Controls (during `--visualize`):**
- `UP/DOWN`: Adjust Tick Rate (FPS).
- `0`: Max Speed (1000 FPS).
- `1`: Normal Speed (30 FPS).

---

## 🧠 Reinforcement Learning Design

### Algorithm: PPO (Proximal Policy Optimization)
- **Policy:** `MlpPolicy` (Multi-layer Perceptron).
- **Architecture:** Efficient MLP optimized for a 14-feature relative observation vector.

### Observation Space (14-element Vector)
The agent perceives its surroundings through a compact relative state representation:

1.  **Danger straight, right, left (3 bits):** Binary flags for immediate collision.
2.  **Direction (4 bits):** [Left, Right, Up, Down] (Current absolute movement).
3.  **Food Location (4 bits):** [Left, Right, Up, Down] (Relative to head).
4.  **Future View (3 floats):** Normalized distances (0 to 1) to the nearest obstacle (wall or body) in the Straight, Right, and Left relative directions.

### Reward System
| Event | Reward | Rationale |
|-------|--------|-----------|
| **Eat Food** | `+10` | Primary objective. |
| **Collision** | `-10` | Terminal penalty (Wall or Body). |
| **Survival** | `0` | Default reward per step. |

---

## ⚙️ Constants & Assumptions
- **Grid Size:** 600x400 pixels.
- **Block Size:** 20px (All coordinates are snapped to this grid).
- **Stagnation Timeout:** Episodes end if the snake takes `100 * len(snake)` steps without eating.
- **Model Path:** Saved to `models/ppo_snake.zip`.
- **Logs:** TensorBoard logs stored in `ppo_snake_tensorboard/`.

---

## 📊 Monitoring
To view training metrics (Reward, Loss, Episode Length):
```bash
tensorboard --logdir=ppo_snake_tensorboard
```
Then visit `http://localhost:6006` in your browser.
