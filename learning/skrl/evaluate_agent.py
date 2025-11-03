import numpy as np
import torch
def evaluate_policy(policy, env, num_envs, num_episodes):
    """Evaluate policy on environment for a given number of episodes with extended metrics"""
    
    obs, info = env.reset()
    states = None
    episode_starts = np.ones((num_envs,), dtype=bool)
    episode_count = 0
    success_count = 0
    collision_count = 0
    timeout_count = 0
    
    # Basic metrics
    episode_rewards = []
    episode_lengths = []
    # 累积奖励
    cumulative_rewards = torch.zeros((num_envs, 1), dtype=torch.float32).to(obs.device)
    cumulative_lengths = torch.zeros((num_envs, ), dtype=torch.int32).to(obs.device)
    # Extended metrics - similar to custom_callback
    episode_cross_track_errors = []
    episode_accelerations = []
    episode_near_collision_ratio = []

    step_count = 0
    while episode_count < num_episodes:
        with torch.inference_mode():
            actions = policy.act(obs, timestep=0, timesteps=0)[0]
            obs, reward, terminated, truncated, info = env.step(actions)
        step_count += 1
        done = terminated | truncated
        cumulative_rewards += reward
        cumulative_lengths += 1
        if done.any():
            for i, done_ in enumerate(done):
                if done_:
                    if cumulative_lengths[i].item() < 20: # 10s
                        # 对于长度非常短的eposide不统计，可能是task初始化就很危险
                        cumulative_rewards[i] = 0
                        cumulative_lengths[i] = 0
                        continue
                    episode_count += 1
                    ep_length = cumulative_lengths[i].item()
                    ep_reward = cumulative_rewards[i].item()
                    cumulative_rewards[i] = 0
                    cumulative_lengths[i] = 0
                    episode_rewards.append(ep_reward)
                    episode_lengths.append(ep_length)
                    
                    # Classify episode result
                    Done_reason = 'Timeout'
                    if info['goal_reached'][i].item():
                        success_count += 1
                        Done_reason = 'Success'
                    elif info['collision'][i].item():
                        collision_count += 1
                        Done_reason = 'Collision'
                    else:
                        timeout_count += 1
                    print(f'Episode {episode_count} {Done_reason} in {ep_length} steps, \
                    reward={ep_reward:.4f}')
        episode_starts = done
    # Calculate basic metrics
    success_rate = success_count / num_episodes
    collision_rate = collision_count / num_episodes
    timeout_rate = timeout_count / num_episodes
    episode_length = np.mean(episode_lengths)
    episode_reward = np.mean(episode_rewards)
    
    mean_cross_track_error = np.mean(episode_cross_track_errors) if len(episode_cross_track_errors) > 0 else 0.0
    mean_acceleration = np.mean(episode_accelerations) if len(episode_accelerations) > 0 else 0.0
    mean_near_collision_ratio = np.mean(episode_near_collision_ratio) if len(episode_near_collision_ratio) > 0 else 0.0
    
    print("="*60)
    print(f"Success rate: {success_rate:.4f}")
    print(f"Collision rate: {collision_rate:.4f}")
    print(f"Timeout rate: {timeout_rate:.4f}")
    print(f"Episode length: {episode_length:.4f} +/- {np.std(episode_lengths):.4f}")
    print(f"Episode reward: {episode_reward:.4f} +/- {np.std(episode_rewards):.4f}")
    print(f"Mean cross-track error: {mean_cross_track_error:.4f}")
    print(f"Mean acceleration: {mean_acceleration:.4f}")
    print(f"Mean near-collision ratio: {mean_near_collision_ratio:.4f}")
    print("="*60)

    evaluate_results = {
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_rate,
        "episode_length": {"mean": episode_length, "std": np.std(episode_lengths)},
        "episode_reward": {"mean": episode_reward, "std": np.std(episode_rewards)},
        "mean_cross_track_error": mean_cross_track_error,
        "mean_acceleration": mean_acceleration,
        "mean_near_collision_ratio": mean_near_collision_ratio,
        "total_episodes": num_episodes,
        "success_count": success_count,
        "collision_count": collision_count,
        "timeout_count": timeout_count
    }   
    return evaluate_results

