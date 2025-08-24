
from dataclasses import dataclass
from typing import Dict

@dataclass
class TrafficConfig:
    """Configuration for traffic simulation."""
    
    # Basic settings
    num_drones: int = 10
    drone_model: str = "crazyflie"  # drone model to use
    
    # Flight parameters
    flight_height: float = 5.0  # meters
    max_speed: float = 3.0  # m/s
    arrival_threshold: float = 1.0  # meters
    
    # Area bounds
    area_bounds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.area_bounds is None:
            self.area_bounds = {
                "xmin": -20.0,
                "xmax": 20.0,
                "ymin": -20.0,
                "ymax": 20.0
            }
