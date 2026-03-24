from stable_baselines3 import PPO

model_path = "/home/arthur/rl_train/logs/ppo_runs/20260316_143302/best_model/policy.pth"

try:
    model = PPO.load(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")