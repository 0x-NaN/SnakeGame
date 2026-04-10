| File | Feature | Original Repo | Your Version | Benefit |
|------|---------|--------------|--------------|---------|
| **`train.py`** | Environment Setup | Single `SnakeEnv()` | `make_vec_env(SnakeEnv, n_envs=8)` | ⚡ **8x faster** experience collection |
| **`train.py`** | Training Steps | `500_000` timesteps | `1_000_000` timesteps | 🎯 More learning for complex state space |
| **`train.py`** | PPO `n_steps` | `2048` | `4096` | 📊 Larger rollouts → more stable updates |
| **`train.py`** | PPO `batch_size` | `64` | `128` | 📈 Better gradient estimation |
| **`train.py`** | Entropy Coefficient | ❌ Not set (default `0.0`) | ✅ `ent_coef=0.01` | 🔄 Prevents early policy collapse / looping |
| **`train.py`** | Callbacks | ❌ None | ✅ `EvalCallback` + `CheckpointCallback` | 💾 Auto-saves best model + crash recovery |
| **`train.py`** | Model Saving | Final model only | ✅ Final + **Best-by-reward** in `models/best/` | 🏆 Always retains top-performing checkpoint |
| **`train.py`** | Eval Environment | ❌ None | ✅ Separate `eval_env` with `render_mode=None` | 📊 Unbiased performance tracking |
| **`play.py`** | Model Loading | Hardcoded `models/ppo_snake` | ✅ Checks `models/best/best_model.zip` first | 🏆 Loads best performer automatically |
| **`play.py`** | Episode Stats | Prints score per episode | ✅ **Avg / Max / Min** summary at end | 📊 Clear performance overview |
| **`play.py`** | Error Handling | Basic | ✅ Try/except + user-friendly message | 🔧 Better UX when model missing |
| **`snake_game.py`** | Observation Space | 14-dimensional | ✅ **15-dimensional** | 🧠 Extra context for decision-making |
| **`snake_game.py`** | Danger Detection | 1-step lookahead only | ✅ **+ 2-step lookahead** (`point_ll`, `point_rr`, etc.) | 🚨 Anticipates traps earlier |
| **`snake_game.py`** | Food Distance | Boolean only (`food_left/right/up/down`) | ✅ **+ Normalized continuous** (`food_dist_x`, `food_dist_y`) | 🎯 Finer-grained navigation signal |
| **`snake_game.py`** | Snake Length | ❌ Not included | ✅ **Normalized length** feature | 📏 Helps policy adapt to growing snake |
| **`snake_game.py`** | Reward Function | Sparse: `+10` (food) / `-10` (death) / `0` (else) | ✅ **+ Distance shaping**: `±0.1` per step toward/away from food | 🧭 Guides learning between food pickups |
| **`snake_game.py`** | Stagnation Limit | `100 * len(snake)` | ✅ `max(300, 100 * len(snake))` | ⏱️ Prevents premature termination on short snakes |
| **`snake_game.py`** | Render FPS Control | Fixed or manual `time.sleep()` | ✅ Dynamic FPS via keyboard (`↑`/`↓`/`0`/`1`) | 🎮 Smoother, adjustable playback |
---
