import numpy as np
import copy
class BaseConfig(object):
    def __init__(self):
        pass
class ArgsConfig:
    def __init__(self):
        self.output_dir = 'trained_models/my_model'
        self.resume = False
        self.load_path = 'trained_models/sb3_model/my_model/checkpoints/best_model.zip'
        self.overwrite = True
        self.num_threads = 1
        self.phase = 'test'
        self.cuda_deterministic = False
        self.no_cuda = False
        self.seed = 425
        self.num_processes = 16
        self.num_mini_batch = 4
        self.num_steps = 128
        self.recurrent_policy = True
        self.ppo_epoch = 4
        self.clip_param = 0.10
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.lr = 4e-5
        self.eps = 1e-5
        self.alpha = 0.99
        self.gamma = 0.99
        self.max_grad_norm = 0.5
        self.num_env_steps = int(100e6)
        self.use_linear_lr_decay = True
        self.use_cosine_lr_decay = False
        self.min_lr = 5e-6
        self.algo = 'ppo'
        self.save_interval = 100
        self.use_gae = True
        self.gae_lambda = 0.95
        self.log_interval = 1
        self.use_proper_time_limits = False
        self.human_node_rnn_size = 128
        self.human_human_edge_rnn_size = 256
        self.aux_loss = False
        self.human_node_input_size = 3
        self.human_human_edge_input_size = 2
        self.human_node_output_size = 256
        self.robot_node_input_size = 7
        self.human_node_embedding_size = 64
        self.human_human_edge_embedding_size = 64
        self.attention_size = 64
        self.seq_length = self.num_steps
        self.use_self_attn = True
        self.use_hr_attn = True
        self.env_name = 'AirspaceSimCity-v0'
        self.sort_humans = True
        self.cuda = True
        self.action_space_type = "beta" # discrete or beta or Gaussian

class Config(object):
    # for now, import all args from arguments.py
    args = ArgsConfig()


    training = BaseConfig()
    training.device = "cuda:0" if args.cuda else "cpu"

    # general configs for OpenAI gym env
    env = BaseConfig()
    env.time_limit = 750
    env.time_step = 0.2
    env.val_size = 100
    env.test_size = 500
    # if randomize human behaviors, set to True, else set to False
    env.randomize_attributes = False
    env.num_processes = args.num_processes
    # record robot states and actions an episode for system identification in sim2real
    env.record = False
    env.load_act = False

    # 添加新的配置项来控制normalization

    env.use_obs_norm = False  # 是否对observation进行标准化
    env.use_reward_norm = True  # 是否对reward进行标准化
    env.norm_clip = None  # 标准化后的裁剪范围
    env.norm_epsilon = 1e-8  # 避免除零的小值

    obs = BaseConfig()
    obs.use_norm = True # 这个是控制的env中的use_norm_dis
    obs.scale = 10 # 这个是距离归一化后的上限
    obs.use_relative_pos = True
    obs.use_norm_radius = False # 这个与norm_dis不能同时用
    obs.use_norm_vel = False
    obs.extend_obs_radius = True


    # config for reward function
    reward = BaseConfig()
    reward.success_reward = 15
    reward.collision_penalty = -20*0.8 # collision_penalty should be negative
    reward.pot_factor = 1.0 # potential reward for approaching goal
    reward.time_penalty = 0 # time_penalty*timestep, time_penalty should be negative
    reward.future_penalty = 0.0 # use for collision with future state, should be nagative
    reward.future_evtol_penalty = -0.8 # use for collision with future state, should be nagative
    reward.future_uav_penalty = -1.0 # use for collision with future state, should be nagative
    # discomfort distance
    reward.discomfort_dist = 0
    reward.discomfort_penalty_factor = 0
    reward.gamma = 0.99

    # config for simulation
    sim = BaseConfig()
    sim.circle_radius = 50 * np.sqrt(2)
    sim.arena_size = 50
    sim.human_num = 20
    sim.evtol_num = 3
    sim.uav_num = sim.human_num - sim.evtol_num
    # actual human num in each timestep, in [human_num-human_num_range, human_num+human_num_range]
    sim.human_num_range = 0
    sim.predict_steps = 5
    # 'const_vel': constant velocity model,
    # 'truth': ground truth future traj (with info in robot's fov)
    # 'inferred': inferred future traj from GST network
    # 'none': no prediction
    # sim.predict_method = 'truth'
    # sim.predict_method = 'inferred'
    sim.predict_method = 'const_vel'
    # render the simulation during training or not
    sim.render = False
    sim.evtol_turn_radius = 100

    # Static obstacle and local map configurations
    sim.use_local_map = False  # Enable local map observations
    sim.local_map_size = 20   # Size of local map observation (20x20 grid)
    sim.grid_size = 0.5       # Size of each grid cell in meters
    # Convert circle_radius to integer grid dimensions
    sim.map_width = int(sim.circle_radius * 2 / sim.grid_size)   # Width of the map in cells
    sim.map_height = int(sim.circle_radius * 2 / sim.grid_size)  # Height of the map in cells
    sim.min_building_size = 10  # Minimum building size in cells
    sim.max_building_size = 25  # Maximum building size in cells
    sim.coverage = 0.0
    # sim.num_buildings = 10    # Number of buildings to generate
    # sim.min_building_spacing = 3  # Minimum space between buildings in cells
    sim.show_safety_margin = True  # Show safety margins in visualization

    # Path planning parameters
    sim.path_resolution = 0.5  # Resolution for path planning in meters
    sim.path_smoothing = True  # Whether to smooth generated paths

    sim.use_curriculum = False
    sim.curriculum_type = 'human_num' # 'radius' or 'human_num'
    sim.curriculum_stage = 0
    sim.curriculum_radius = [sim.circle_radius*0.25,
                             sim.circle_radius*0.5,
                             sim.circle_radius*0.75,
                             sim.circle_radius]
    sim.curriculum_human_num = [[int(sim.human_num*0.25), int(sim.evtol_num*0.25)],
                                [int(sim.human_num*0.5 ), int(sim.evtol_num*0.5)],
                                [int(sim.human_num*0.75), int(sim.evtol_num*0.75)],
                                [sim.human_num, sim.evtol_num]]
    sim.curriculum_success_rate = 0.8
    # # Sensor ranges (ensure they cover local map area if needed)
    # robot.sensor_range = max(70, sim.local_map_size * sim.grid_size / 2)
    # evtol.sensor_range = max(1000, sim.local_map_size * sim.grid_size / 2)
    # uav.sensor_range = max(70, sim.local_map_size * sim.grid_size / 2)

    # for save_traj only
    render_traj = False
    save_slides = False
    save_path = None

    # whether wrap the vec env with VecPretextNormalize class
    # = True only if we are using a network for human trajectory prediction (sim.predict_method = 'inferred')
    if sim.predict_method == 'inferred':
        env.use_wrapper = True
    else:
        env.use_wrapper = False

    # human config
    humans = BaseConfig()
    humans.visible = True
    # orca or social_force for now
    humans.policy = "orca"
    humans.radius = 0.3
    humans.v_pref = 1
    humans.sensor = "coordinates"
    humans.kinematics = "holonomic"
    # FOV = this values * PI
    humans.FOV = 2.

    # a human may change its goal before it reaches its old goal
    # if randomize human behaviors, set to True, else set to False
    humans.random_goal_changing = True
    humans.goal_change_chance = 0.5

    # a human may change its goal after it reaches its old goal
    humans.end_goal_changing = True
    humans.end_goal_change_chance = 1.0

    # a human may change its radius and/or v_pref after it reaches its current goal
    humans.random_radii = False
    humans.random_v_pref = False

    # one human may have a random chance to be blind to other agents at every time step
    humans.random_unobservability = False
    humans.unobservable_chance = 0.3

    humans.random_policy_changing = False

    # robot config
    robot = BaseConfig()
    # whether robot is visible to humans (whether humans respond to the robot's motion)
    robot.visible = False
    # For baseline: srnn; our method: selfAttn_merge_srnn
    robot.policy = 'selfAttn_merge_srnn'
    # robot.policy = 'follow_path'
    robot.radius = 1.0
    robot.v_pref = 1.0
    robot.sensor = "coordinates"
    # FOV = this values * PI
    robot.FOV = 2
    # radius of perception range
    robot.sensor_range = 100
    # sensor_range设置小的，说明robot看做UAV，但是感知evtol的时候，依然用evtol的那个距离
    # sensor_range设置为大的1000m，说明robot看做eVTOL # 但是暂时不考虑eVTOL主动避让UAV的情况，其实没意义

    # action space of the robot
    action_space = BaseConfig()
    # holonomic or unicycle
    action_space.kinematics = "holonomic"
    action_space.action_space_type = "Discrete"  # 改为Discrete
    action_space.action_space_num_per_dim = 7 # 每个方向上有9个action
    # action_space.kinematics = "unicycle"


    # 在我的配置中, robot依然作为训练的agent的配置存在, human配置不再使用
    # 转而分别设置evtol和uav的配置

    # uav config
    uav = BaseConfig()
    uav.visible = True
    # orca or social_force for now
    uav.policy = "nhorca"
    uav.radius = 1.0
    uav.v_pref = 1.0
    uav.sensor = "coordinates"
    uav.kinematics = "holonomic"
    # FOV = this values * PI
    uav.FOV = 2
    uav.sensor_range = 25

    # a UAV may change its goal before it reaches its old goal
    # if randomize UAV behaviors, set to True, else set to False
    uav.random_goal_changing = True
    uav.goal_change_chance = 0.2

    # a UAV may change its goal after it reaches its old goal
    uav.end_goal_changing = True
    uav.end_goal_change_chance = 1.0

    # a UAV may change its radius and/or v_pref after it reaches its current goal
    uav.random_radii = False
    uav.random_v_pref = False

    # one uav may have a random chance to be blind to other agents at every time step
    uav.random_unobservability = False
    uav.unobservable_chance = 0.3

    uav.random_policy_changing = False

    evtol = copy.deepcopy(uav)

    evtol.policy = "follow_path"
    evtol.kinematics = "unicycle"
    evtol.radius = 10.0
    evtol.v_pref = 2.0
    evtol.random_goal_changing = False
    evtol.end_goal_changing = True # 开启下一次的任务
    evtol.end_goal_change_chance = 1.0
    evtol.random_radii = False
    evtol.random_v_pref = False
    evtol.delta_theta_bound = np.pi / 3.0
    evtol.sensor_range = 100

    # config for ORCA
    orca = BaseConfig()
    orca.neighbor_dist = uav.sensor_range
    orca.safety_space = 5
    orca.time_horizon = 5
    orca.time_horizon_obst = 5

    # config for free flight
    freeflight = BaseConfig()
    freeflight.k1 = 1.0
    freeflight.k2 = 2.0
    freeflight.epsilon = 0.00001
    freeflight.epsilon_s = 0.00001
    freeflight.d1 = 4.0
    freeflight.d2 = 5.0
    freeflight.l_i = 2.0

    # config for social force
    sf = BaseConfig()
    sf.A = 2.
    sf.B = 1
    sf.KI = 1

    # config for data collection for training the GST predictor
    data = BaseConfig()
    data.tot_steps = 40000
    data.render = False
    data.collect_train_data = False
    data.num_processes = 5
    data.data_save_dir = 'gst_updated/datasets/orca_20humans_no_rand'
    # number of seconds between each position in traj pred model
    data.pred_timestep = 2.0

    # config for the GST predictor
    pred = BaseConfig()
    # see 'gst_updated/results/README.md' for how to set this variable
    # If randomized humans: gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj
    # else: gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000/sj
    pred.model_dir = 'gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj'

    # LIDAR config
    lidar = BaseConfig()
    # angular resolution (offset angle between neighboring rays) in degrees
    lidar.angular_res = 5
    # range in meters
    lidar.range = 10

    # config for sim2real
    sim2real = BaseConfig()
    # use dummy robot and human states or not
    sim2real.use_dummy_detect = True
    sim2real.record = False
    sim2real.load_act = False
    sim2real.ROSStepInterval = 0.03
    sim2real.fixed_time_interval = 0.1
    sim2real.use_fixed_time_interval = True

    if sim.predict_method == 'inferred' and env.use_wrapper == False:
        raise ValueError("If using inferred prediction, you must wrap the envs!")
    if sim.predict_method != 'inferred' and env.use_wrapper:
        raise ValueError("If not using inferred prediction, you must NOT wrap the envs!")

