import numpy as np

def evaluate_policy(policy, env, num_envs, num_episodes, new_logger):
    """Evaluate policy on environment for a given number of episodes with extended metrics"""
    
    obs = env.reset()
    states = None
    episode_starts = np.ones((num_envs,), dtype=bool)
    episode_count = 0
    success_count = 0
    collision_count = 0
    timeout_count = 0
    
    # Basic metrics
    episode_rewards = []
    episode_lengths = []
    
    # Extended metrics - similar to custom_callback
    episode_cross_track_errors = []
    episode_accelerations = []
    episode_near_collision_ratio = []

    step_count = 0
    while episode_count < num_episodes:
        # Model-based policies have simple predict interface
        action, states = policy.predict(obs, 
                                    state=states, 
                                    episode_start=episode_starts, 
                                    deterministic=True)
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        if done.any():
            for i, done_ in enumerate(done):
                if done_:
                    episode_count += 1
                    ep_length = info[i]['episode']['l']
                    ep_reward = info[i]['episode']['r']
                    
                    episode_rewards.append(ep_reward)
                    episode_lengths.append(ep_length)
                 
                    # Extract cross-track error if available
                    cross_track_error = 0.0
                    if 'episode_cross_error' in info[i]:  # Note: typo in original code
                        cross_track_error = info[i]['episode_cross_error'].item()
                    
                    episode_cross_track_errors.append(cross_track_error)

                    near_collision_ratio = 0.0
                    if 'episode_near_collision_ratio' in info[i]:
                        near_collision_ratio = info[i]['episode_near_collision_ratio'].item()
                    episode_near_collision_ratio.append(near_collision_ratio)

                    
                    # Extract acceleration if available
                    acceleration = 0.0
                    if 'episode_acceleration' in info[i]:
                        acceleration = info[i]['episode_acceleration'].item()
                    episode_accelerations.append(acceleration)
                    
                    # Classify episode result
                    Done_reason = 'Timeout'
                    if info[i]['goal_reached']:
                        success_count += 1
                        Done_reason = 'Success'
                    elif info[i]['collision']:
                        collision_count += 1
                        Done_reason = 'Collision'
                    else:
                        timeout_count += 1
                    new_logger.info(f'Episode {episode_count} {Done_reason} in {ep_length} steps, \
                    reward={ep_reward:.4f}, cross_error={cross_track_error:.4f}, near_collision_ratio={near_collision_ratio:.4f}')
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
    
    new_logger.info("="*60)
    new_logger.info(f"Success rate: {success_rate:.4f}")
    new_logger.info(f"Collision rate: {collision_rate:.4f}")
    new_logger.info(f"Timeout rate: {timeout_rate:.4f}")
    new_logger.info(f"Episode length: {episode_length:.4f} +/- {np.std(episode_lengths):.4f}")
    new_logger.info(f"Episode reward: {episode_reward:.4f} +/- {np.std(episode_rewards):.4f}")
    new_logger.info(f"Mean cross-track error: {mean_cross_track_error:.4f}")
    new_logger.info(f"Mean acceleration: {mean_acceleration:.4f}")
    new_logger.info(f"Mean near-collision ratio: {mean_near_collision_ratio:.4f}")
    new_logger.info("="*60)

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

