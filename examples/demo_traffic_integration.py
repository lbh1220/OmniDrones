#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Demo script showing how to integrate TrafficSimulator with OmniDrones environments.

This example demonstrates:
1. How to add traffic simulation to an existing RL environment
2. How to access traffic aircraft states
3. How to perform collision detection with traffic aircraft
4. How to configure traffic parameters
"""

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from omni_drones import init_simulation_app


@hydra.main(version_base=None, config_path=".", config_name="demo_traffic")
def main(cfg: DictConfig):
    """Main function for traffic integration demo."""
    
    print("Starting traffic integration demo...")
    print(f"Configuration loaded successfully")
    
    # Initialize simulation
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    
    # Create the traffic-aware environment
    from omni_drones.envs.traffic.traffic_hover import TrafficAwareHover
    env = TrafficAwareHover(cfg, headless=False)
    
    print("Traffic integration demo initialized!")
    print(f"Environment has {cfg.env.num_envs} agents")
    print(f"Traffic simulation has {cfg.traffic.num_drones} aircraft")
    print(f"Traffic area bounds: {cfg.traffic.area_bounds}")
    
    # Run demonstration
    try:
        # Reset environment
        tensordict = env.reset()
        print("Environment reset completed")
        import time
        # Run simulation steps
        start_time = time.time()
        for step in range(50000):
            # Random actions for demo (in real use, these would come from your RL agent)
            action_dim = 4  # placeholder
            actions = torch.randn(cfg.env.num_envs, 1, action_dim, device=env.device) * 0.1
            
            # Step the environment
            tensordict["agents", "action"] = actions
            tensordict = env.step(tensordict)
            
            # # Every 50 steps, print traffic information
            # if step % 50 == 0:
            #     traffic_info = env.get_traffic_info()
            #     print(f"Step {step}: {traffic_info['num_drones']} traffic aircraft active")
                
            #     # Demo collision detection
            #     if "positions" in traffic_info and len(traffic_info["positions"]) > 0:
            #         # Use dummy agent positions for demo
            #         dummy_agent_pos = torch.zeros(cfg.env.num_envs, 3, device=env.device)
            #         collisions = env.check_traffic_collision(dummy_agent_pos, safety_radius=5.0)
            #         if collisions.any():
            #             print(f"  Warning: {collisions.sum()} agents in collision zone!")
            if step % 100 == 0:
                end_time = time.time()
                print(f"Step {step} time: {(end_time - start_time)/100} seconds")
                start_time = end_time
            # Check if any environment needs reset
            if tensordict.get("terminated", torch.zeros_like(tensordict.get("truncated"))).any():
                env_mask = (tensordict.get("terminated") | tensordict.get("truncated")).squeeze(-1)
                tensordict = env.reset(tensordict.select(*env_mask.nonzero().squeeze(-1)))
    
    except KeyboardInterrupt:
        print("Demo interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("Closing simulation...")
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
