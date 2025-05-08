import os, yaml, gym, numpy as np
import carla

from envs.custom_carla_env import CarlaEnv
from predictors import qcnet_wrapper
from ppo.callbacks import SaveOnBestTrainingRewardCallback

USE_CUSTOM = False  # Set True to use custom PPO implementation

if not USE_CUSTOM:
    from stable_baselines3 import PPO
else:
    from ppo.custom_ppo.ppo_agent import PPOAgent as CustomPPO

def make_env(client, cfg, k=6, horizon=12):
    base_env = CarlaEnv(client, town=cfg['town'], weather_id=cfg['weather_id'])
    class PredictorWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            pred_dim = 10 * k * horizon * 2
            low = np.concatenate([self.observation_space.low,
                                  np.full(pred_dim, -1e4)])
            high = np.concatenate([self.observation_space.high,
                                   np.full(pred_dim, 1e4)])
            self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        def _concat(self, obs):
            traj, _ = qcnet_wrapper.predict(self.env.world, k=k, horizon=horizon)
            return np.concatenate([obs, traj.flatten()])
        def reset(self, **kwargs):
            return self._concat(self.env.reset(**kwargs))
        def step(self, action):
            obs, reward, done, info = self.env.step(action)
            return self._concat(obs), reward, done, info
    return PredictorWrapper(base_env)

def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)

    env = make_env(client, cfg)

    if USE_CUSTOM:
        agent = CustomPPO(env.observation_space.shape[0],
                          env.action_space.shape[0])
        obs = env.reset()
        for t in range(cfg['total_steps']):
            action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            agent.buffer.rewards.append(reward)
            agent.buffer.dones.append(done)
            if done:
                obs = env.reset()
            if (t + 1) % cfg['n_steps'] == 0:
                agent.update()
        import torch
        torch.save(agent.policy.state_dict(), "custom_ppo_carla.pt")
    else:
        model = PPO("MlpPolicy", env,
                    learning_rate=cfg['lr'],
                    n_steps=cfg['n_steps'],
                    batch_size=cfg['batch_size'],
                    clip_range=cfg['clip_range'],
                    gamma=cfg['gamma'],
                    verbose=1,
                    tensorboard_log="./tb_logs")
        callback = SaveOnBestTrainingRewardCallback(5000, "./models")
        model.learn(total_timesteps=cfg['total_steps'], callback=callback)
        model.save("ppo_carla_agent")
    env.close()

if __name__ == "__main__":
    main()
