import carla
import time
import numpy as np
from stable_baselines3 import PPO
from custom_carla_env import CarlaEnv 

def run_evaluation(town="Town01", weather_id=0, model_path="ppo_carla_agent"):
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    # Initialize CARLA environment
    env = CarlaEnv(client=client, town=town, weather_id=weather_id, eval_mode=True)
    model = PPO.load(model_path)

    obs = env.reset()
    done = False
    episode_reward = 0
    step_count = 0
    collisions = 0
    start_time = time.time()

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        collisions = info.get("collisions", 0)
        step_count += 1

    env.close()

    episode_duration = time.time() - start_time
    print(f"Evaluation Completed in {step_count} steps")
    print(f"Total Reward: {episode_reward:.2f}")
    print(f"Collisions: {collisions}")
    print(f"Episode Time: {episode_duration:.2f} sec")

    with open("eval_metrics.csv", "a") as f:
        f.write(f"{town},{weather_id},{step_count},{episode_reward:.2f},{collisions},{episode_duration:.2f}\n")

if __name__ == "__main__":
    run_evaluation()
