from stable_baselines3 import PPO
from snake_game import SnakeEnv
import os

def train(visualize=False, fps=30, continue_training=False):
    # Create environment
    render_mode = "human" if visualize else None
    env = SnakeEnv(render_mode=render_mode, fps=fps)
    
    model_dir = "models"
    model_path = f"{model_dir}/ppo_snake"
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if continue_training and os.path.exists(f"{model_path}.zip"):
        print(f"Loading existing model from {model_path}...")
        model = PPO.load(model_path, env=env, device="cpu")
    else:
        print("Creating new model with MlpPolicy...")
        # Define model with MlpPolicy for flattened grid state
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            tensorboard_log="./ppo_snake_tensorboard/",
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            device="cpu"
        )

    print("Starting training...")
    # Train for 500,000 steps
    # Note: For a quick test, you can reduce this number
    TIMESTEPS = 500000
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    
    # Save the model
    model.save(f"{model_dir}/ppo_snake")
    print(f"Model saved to {model_dir}/ppo_snake")

if __name__ == "__main__":
    train()
