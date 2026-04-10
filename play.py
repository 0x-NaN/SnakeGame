from stable_baselines3 import PPO
from snake_game import SnakeEnv
import time

def play():
    # Load environment with human rendering
    env = SnakeEnv(render_mode="human")
    
    # Load trained model
    model_path = "models/ppo_snake"
    try:
        model = PPO.load(model_path)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"No model found at {model_path}. Please train first.")
        return

    # Run game episodes
    episodes = 5
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        score = 0
        print(f"Starting Episode {ep+1}")
        
        while not done:
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            score = info.get("score", 0)
            
            # Slow down for visualization
            time.sleep(0.05)
            
        print(f"Episode {ep+1} Finished. Score: {score}")

if __name__ == "__main__":
    play()
