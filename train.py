from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from snake_game import SnakeEnv
import os

def train(visualize=False, fps=30, continue_training=False):
    model_dir = "models"
    model_path = f"{model_dir}/ppo_snake"
    best_model_path = f"{model_dir}/best"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # --- Vectorized environments (8 parallel) for faster experience collection ---
    # Note: render_mode not supported with make_vec_env; watch via play mode instead
    if visualize:
        print("Note: Visualization disabled during vectorized training. Use play mode to watch.")
    env = make_vec_env(SnakeEnv, n_envs=8)

    # --- Separate eval environment (single, no rendering) ---
    eval_env = SnakeEnv(render_mode=None)

    # --- Callbacks ---
    # Saves best model based on mean reward over eval episodes
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_path,
        log_path="./logs/eval/",
        eval_freq=10000,          # evaluate every 10k steps
        n_eval_episodes=10,       # average over 10 episodes
        deterministic=True,
        render=False,
        verbose=1,
    )

    # Saves checkpoint every 50k steps so you don't lose progress on crash
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=f"{model_dir}/checkpoints/",
        name_prefix="ppo_snake",
        verbose=1,
    )

    if continue_training and os.path.exists(f"{model_path}.zip"):
        print(f"Loading existing model from {model_path}...")
        model = PPO.load(model_path, env=env)
    else:
        print("Creating new model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./ppo_snake_tensorboard/",
            # --- Core hyperparameters ---
            learning_rate=3e-4,
            n_steps=4096,         # more rollout before each update (was 2048)
            batch_size=128,       # larger minibatch (was 64)
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,      # explicit (was default)
            clip_range=0.2,       # explicit (was default)
            # --- Key addition: entropy bonus ---
            # Prevents policy from collapsing to a deterministic loop too early
            # Critical for pushing past the score ~20 plateau
            ent_coef=0.01,
        )

    print("Starting training...")
    TIMESTEPS = 1_000_000  # bumped from 500k — expanded state space needs more steps
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=not continue_training,
        tb_log_name="PPO",
        callback=[eval_callback, checkpoint_callback],
    )

    # Save the final model
    model.save(model_path)
    print(f"Final model saved to {model_path}")
    print(f"Best model (by eval reward) saved to {best_model_path}/best_model")

if __name__ == "__main__":
    train()
