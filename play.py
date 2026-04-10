from stable_baselines3 import PPO
from snake_game import SnakeEnv

def play():
    env = SnakeEnv(render_mode="human")

    # Prefer the best model (saved by EvalCallback) over the final model
    best_model_path = "models/best/best_model"
    final_model_path = "models/ppo_snake"

    model_path = best_model_path if __import__("os").path.exists(f"{best_model_path}.zip") else final_model_path

    try:
        model = PPO.load(model_path)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"No model found. Please train first.")
        return

    episodes = 5
    scores = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        score = 0
        print(f"Starting Episode {ep+1}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            score = info.get("score", 0)

        scores.append(score)
        print(f"Episode {ep+1} Finished. Score: {score}")

    print(f"\nAvg Score: {sum(scores)/len(scores):.1f} | Max: {max(scores)} | Min: {min(scores)}")

if __name__ == "__main__":
    play()
