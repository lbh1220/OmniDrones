# traffic simulator

新增update_grid_map, 将env中的grid map给两个manager，通过调用manamger的update_grid map

两个target generator因为需要做一些安全性判断和轨迹搜索

把manager中的grid map和global planner传进去

# state

中增加一个新的命名空间，填充grid map的信息，仿照env的state

## set_waypoints_for_aircraft

这个函数仅在evtol manager中使用，不需要动，保留现在的状态

drones manager如果要设置waypoints, 用env内差不多的方式，直接在manager中做赋值，并且不需要填充actual length之后的部分

## 新增update_navigation_state_vectorized函数

逻辑完全模仿env.mdp.state中的update_navigation_state_vectorized函数，为drones更新制导信息

# drones manager

update_grid_map

从simulator那边拿到grid map, 然后原地extended, 在manager中仅保留extended之后的map; 应该还需要同时更新给global planner

原地更新build bbox的内容

_generate_random_position

需要对安全性进行检查，

initialize_targets

增加对target安全性的检查，检查是否安全，

generate_targets

drones的目标点生成在candidate_targets附近，需要检查这个过程生成的target是否安全

check collision

需要增加这个函数，检查是否与evtol发生碰撞，drones彼此之间是否碰撞，是否侵入了危险的grid中

返回一个collision mask

_post_physics_step

先_update_state_manager，然后检测碰撞拿到collision mask

处理那些collision mask的飞机，不能调用_reset_idx，因为_reset_idx会重置traffic中的所有飞机，重新生成位置，模仿multirotor base中的_reset_idx，重置对应的速度acc, torques之类的;  还需要重新为他们生成targets, 整体参考reset的逻辑，但是不需要更新state,也不需要initialize_targets

set_initial_targets

改名为set_targets(drones_ids=None)

修改这个函数的接口，输入drones_ids = None, 如果是None就更新所有飞机，如果不是None就更新特定的飞机

pre_physics_step中，if num_arrived>0的话，就可以调用新的set_targets()来更新部分飞机的状态了

reset_drones

重置输入的idx(多个)飞机，内部模仿multirotor base中的逻辑，env_ids是0, 需要重置的是[env_ids][idx]上的所有内容，因为他是一个[env_num, drones_num, x]的结构

omni_drones/robots/drone/multirotor.py

我已经写了雏形了

update_global_path(drones_ids)

本次新增的关键函数，核心是计算从state.start_positions到state.target_positions的路径，使用global planner

生成的路径仿照city_nav_env的方式赋值给state

将其放在由set_initial_targets更改的新的set_targets函数靠后的位置处

_pre_physics_step

计算velocity_commands的指令整体大改，调用新的update_navigation_state_vectorized函数来计算出local goal, 以local goal作为velocity_commands的方向

local goal通常是有一定距离的，也就是update_navigation_state时候输入的lookahead_distance， 当终点里的很近的时候local goal的距离将会小于lookahead_distance，利用这个与local goal之间的距离作为阈值，出发与原来类似的减速逻辑

# evtol manager

update_grid_map, 

从simulator那边拿到grid map, 然后原地extended, 在manager中仅保留extended之后的map, 应该还需要同时更新给global planner

generate_course_with_smooth_trajectory

修改inter_points中的逻辑，应当用global planner来生成中间点

evtol manager不需要用张量描述所有waypoints,因为他是依靠smooth path运行的，smoothpath是一个list of numpy, evtol manager的更新是遍历做的，

整个evtol manager中不需要追求并行化的张量运算，所以不需要做太多的修改