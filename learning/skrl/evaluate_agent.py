import numpy as np
import torch
from collections import defaultdict
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
    # Dynamic metrics tracking (collected only at episode end)
    metrics_names = set()
    metrics_values_by_name = defaultdict(list)

    def _to_scalar(x):
        """Best-effort conversion of various values to float scalar."""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            if x.numel() == 1:
                return float(x.item())
            return float(x.reshape(-1)[0].item())
        if isinstance(x, np.ndarray):
            if x.size == 1:
                return float(x.item())
            return float(x.reshape(-1)[0].item())
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        try:
            return float(x)  # fall back for types convertible to float
        except Exception:
            return None

    def _get_env_value(v, idx):
        """Get the value for env index idx from v (supports tensor/ndarray/list/tuple/scalar)."""
        try:
            if isinstance(v, (torch.Tensor, np.ndarray, list, tuple)):
                elem = v[idx]
                return _to_scalar(elem)
            # scalar broadcast
            return _to_scalar(v)
        except Exception:
            # try squeezing then indexing
            try:
                if isinstance(v, torch.Tensor):
                    v2 = v.squeeze()
                    elem = v2[idx] if v2.ndim >= 1 else v2
                    return _to_scalar(elem)
                if isinstance(v, np.ndarray):
                    v2 = np.squeeze(v)
                    elem = v2[idx] if v2.ndim >= 1 else v2
                    return _to_scalar(elem)
            except Exception:
                return None
        return None

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
                    # Classify episode result
                    do_record = False
                    Done_reason = 'Timeout'
                    if info['goal_reached'][i].item():
                        success_count += 1
                        do_record = True
                        Done_reason = 'Success'
                    elif info['collision'][i].item():
                        collision_count += 1
                        Done_reason = 'Collision'
                    else:
                        timeout_count += 1
                    if cumulative_lengths[i].item() < 20: # 10s
                        # 对于长度非常短的eposide不统计，可能是task初始化就很危险
                        cumulative_rewards[i] = 0
                        cumulative_lengths[i] = 0
                        continue
                    # Collect dynamic metrics only from this step's info for env i
                    if do_record:
                        for k, v in info.items():
                            if isinstance(k, str) and k.startswith('metrics/'):
                                name = k.split('/', 1)[1]
                                metrics_names.add(name)
                                val = _get_env_value(v, i)
                                if val is not None and np.isfinite(val):
                                    metrics_values_by_name[name].append(val)
                    episode_count += 1
                    ep_length = cumulative_lengths[i].item()
                    ep_reward = cumulative_rewards[i].item()
                    cumulative_rewards[i] = 0
                    cumulative_lengths[i] = 0
                    if do_record:
                        episode_rewards.append(ep_reward)
                        episode_lengths.append(ep_length)
                    

                    print(f'Episode {episode_count} {Done_reason} in {ep_length} steps, reward={ep_reward:.4f}', end='')
                    # for name, values in metrics_values_by_name.items():
                    #     print(f", {name}={np.mean(values):.4f}", end='')
                    print()
        episode_starts = done
    # Calculate basic metrics
    success_rate = success_count / episode_count
    collision_rate = collision_count / episode_count
    timeout_rate = timeout_count / episode_count
    episode_length = np.mean(episode_lengths)
    episode_reward = np.mean(episode_rewards)
    # Aggregate dynamic metrics
    metrics_summary = {}
    for name, values in metrics_values_by_name.items():
        if len(values) > 0:
            metrics_summary[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values))
            }
        else:
            metrics_summary[name] = {
                "mean": 0.0,
                "std": 0.0
            }
    
    print("="*60)
    print(f"Success rate: {success_rate:.4f}")
    print(f"Collision rate: {collision_rate:.4f}")
    print(f"Timeout rate: {timeout_rate:.4f}")
    print(f"Episode length: {episode_length:.4f} +/- {np.std(episode_lengths):.4f}")
    print(f"Episode reward: {episode_reward:.4f} +/- {np.std(episode_rewards):.4f}")
    # Print dynamic metrics summary
    if metrics_summary:
        for name in sorted(metrics_summary.keys()):
            m = metrics_summary[name]
            print(f"Metric {name}: {m['mean']:.4f} +/- {m['std']:.4f}")
    print("="*60)

    evaluate_results = {
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_rate,
        "episode_length": {"mean": episode_length, "std": np.std(episode_lengths)},
        "episode_reward": {"mean": episode_reward, "std": np.std(episode_rewards)},
        "total_episodes": num_episodes,
        "success_count": success_count,
        "collision_count": collision_count,
        "timeout_count": timeout_count
    }   
    # Attach dynamic metrics and keep backward compatibility for common keys
    evaluate_results["metrics"] = metrics_summary
    for legacy_key in ("mean_cross_track_error", "mean_acceleration", "mean_near_collision_ratio"):
        if legacy_key in metrics_summary:
            evaluate_results[legacy_key] = metrics_summary[legacy_key]["mean"]
    return evaluate_results

