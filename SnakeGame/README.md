# 🐍 Snake Game: Logic, Python Implementation & RL Integration Guide

This document breaks down the core logic of the Snake game, explains how it's built with Python & Pygame, and provides a step-by-step roadmap to train a Reinforcement Learning (RL) agent to play it autonomously.

---

## 1. 🧠 Core Game Logic

The game follows a classic grid-based simulation loop:

| Component | Logic |
|-----------|-------|
| **Grid System** | The screen is divided into `block_size` (20px) cells. All positions are snapped to this grid. |
| **Snake Representation** | A Python list of `(x, y)` tuples. The last element is always the **head**. |
| **Movement** | Each frame, a new head is calculated based on `direction`. It's appended to the list. If no food is eaten, the tail (`pop(0)`) is removed, creating the illusion of movement. |
| **Food Spawning** | Random grid coordinates are generated until one falls outside the snake's body. |
| **Collision Detection** | Fails if the new head: <br>• Goes outside `[0, WIDTH]` or `[0, HEIGHT]` (wall)<br>• Overlaps with `self.snake[:-1]` (self) |
| **Game Loop** | `Event Polling → State Update → Rendering → Frame Cap` runs continuously until quit or collision. |

---

## 2. 🐍 Python & Pygame Implementation Details

### Why Python + Pygame?
- **Python** offers fast prototyping, readable syntax, and a massive ML ecosystem.
- **Pygame** handles low-level tasks: window creation, input polling, 2D drawing, and frame timing (`pygame.time.Clock`).

### Architecture Highlights
```python
class SnakeGame:
    def __init__(self): ...      # Initial state
    def update_game(self): ...   # Pure logic (movement, collision, scoring)
    def draw_game(self): ...     # Pure rendering (Pygame draw calls)
```
- **Separation of Concerns**: Logic and rendering are decoupled, making it easier to swap keyboard input for an AI agent later.
- **Frame Rate Control**: `clock.tick(speed)` ensures consistent gameplay regardless of hardware.
- **Python Features Used**: Lists for dynamic body tracking, tuples for immutable coordinates, f-strings for UI, and `sys.exit()` for clean termination.

> 💡 *Note: The original snippet had indentation errors, missing color definitions, and a broken `if __name__` guard. These were corrected for Python 3.14 compatibility.*

---

## 3. 🤖 Integrating Reinforcement Learning (RL)

To make the snake play itself, we frame it as a **Markov Decision Process (MDP)** and train an RL agent using libraries like `gymnasium` and `stable-baselines3`.

### 3.1 Environment Wrapper (Gymnasium-Compatible)
Pygame is too slow for direct RL training. We wrap the game logic into a standard environment that exposes `reset()`, `step()`, and `render()`.

```python
import gymnasium as gym
import numpy as np

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.action_space = gym.spaces.Discrete(4)  # UP, DOWN, LEFT, RIGHT
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)
        self.render_mode = render_mode
        self.game = SnakeGame()  # Reuse existing logic class

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.restart()
        return self._get_obs(), {}

    def step(self, action):
        # Map action to direction
        dirs = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        # Prevent 180° turns
        if dirs[action] != self.game._opposite(self.game.direction):
            self.game.direction = dirs[action]
            
        self.game.update_game()
        reward, done = self._calculate_reward()
        return self._get_obs(), reward, done, False, {}

    def _calculate_reward(self):
        head = self.game.snake[-1]
        # +10 for food, -10 for death, -0.1 per step, +0.1 for moving toward food
        reward = -0.1
        if head == self.game.food:
            reward += 10
        if done := (len(self.game.snake) == 0):  # Assuming game_over sets flag or raises
            reward -= 10
        return reward, done

    def _get_obs(self):
        # Normalized state: [head_x, head_y, food_x, food_y, danger_straight, danger_right, danger_left, dir_x, dir_y, ...]
        return np.random.rand(11).astype(np.float32)  # Placeholder

    def render(self):
        if self.render_mode == "human":
            self.game.draw_game()
```

### 3.2 MDP Design
| Component | Design Choice |
|-----------|---------------|
| **State Space** | `Box(11,)` or `Box(640, 480, 3)` for pixels. Structured vectors train faster. Include: head position, food position, danger in 3 directions, current direction, distance to food. |
| **Action Space** | `Discrete(4)` → `[UP, DOWN, LEFT, RIGHT]` |
| **Reward Function** | Crucial for convergence:<br>`+10` eat food<br>`-10` die<br>`-0.1` per step (prevents looping)<br>`+0.1` step toward food (dense reward shaping) |
| **Terminal Condition** | Wall/self collision or max steps reached. |

### 3.3 Recommended Algorithms
| Algorithm | Pros | Cons |
|-----------|------|------|
| **DQN** (Deep Q-Network) | Simple, works well with discrete actions, well-documented for Snake | Can be unstable with sparse rewards |
| **PPO** (Proximal Policy Optimization) | Stable, sample-efficient, modern default | Slightly more hyperparameter tuning |
| **A2C** | Good balance of speed & stability | May struggle with long-horizon credit assignment |

📦 **Recommended Stack**: `gymnasium` + `stable-baselines3` (PyTorch) + `tensorboard` for monitoring.

### 3.4 Step-by-Step Integration Guide
1. **Decouple Logic from Rendering**: Ensure `update_game()` runs without `pygame.display.flip()` during training.
2. **Wrap as Gym Env**: Implement `reset()`, `step()`, `_get_obs()`, `_calculate_reward()`.
3. **Headless Training**: Run training without `render_mode="human"` to maximize FPS (1000+ steps/sec).
4. **Train**:
   ```python
   from stable_baselines3 import PPO
   env = SnakeEnv()
   model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
   model.learn(total_timesteps=500_000)
   model.save("snake_ppo")
   ```
5. **Inference & Visualization**:
   ```python
   env = SnakeEnv(render_mode="human")
   model = PPO.load("snake_ppo")
   obs, _ = env.reset()
   for _ in range(1000):
       action, _ = model.predict(obs, deterministic=True)
       obs, reward, done, _, _ = env.step(action)
       env.render()
       if done: obs, _ = env.reset()
   ```

### 3.5 Training Tips & Common Pitfalls
| Issue | Solution |
|-------|----------|
| **Agent spins in circles** | Add step penalty (`-0.1`) + reward for moving toward food |
| **Slow training** | Use headless mode, vectorized environments (`SubprocVecEnv`), or frame skipping |
| **Forgets how to eat** | Use reward clipping, increase food reward, or curriculum learning (start small grid) |
| **Action latency** | Ensure `clock.tick()` matches RL step rate during inference |

---

## 📚 Next Steps & Resources
- 📖 **Gymnasium Docs**: https://gymnasium.farama.org/
- 🤖 **Stable Baselines3**: https://stable-baselines3.readthedocs.io/
- 🧪 **Open-Source Snake RL**: Search `snake-gymnasium` or `rl-snake` on GitHub for reference implementations.
- 🚀 **Advanced**: Add curriculum learning (grow grid as score increases), imitation learning from human replays, or multi-agent competitive modes.