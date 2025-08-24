# 总体需求
本文档旨在设计并实现一个集成于 Isaac Lab 强化学习环境中的动态交通仿真模块 TrafficSimulator。该模块的核心目标是复现并改进原 AirSim 交通仿真器的基础功能，使其完全原生于 Isaac Sim 的矢量化、GPU 加速框架。

此模块将作为一个独立的 Python 类，在 RLEnv 中被实例化和管理。它将负责在仿真世界中生成、管理并驱动一组背景交通无人机（Traffic UAVs），并为RL环境中的智能体（Ego-Agents）提供一个动态、可交互的背景。

我需要将原来适用于airsim的air traffic simulator中的核心功能air_traffic_simulator/reference/airsim_traffic_simulator/simulator.py
迁移到isaac sim中

希望将其作为一个environment的插件存在，在我定义RL env的时候，可以实例化这个类，然后和整个RL env一起更新

这个类中需要定义一些接口，可以访问air traffic中的飞机状态等，方便RL env中访问信息或者设置某些信息

# 设计方案

## 设计要求

原来airsim的traffic功能复杂，包括两类飞机，uav和evtol,同时uav还有是否有planner的两种

目前的设计方案可以暂时不考虑planner,也不需要用astar做搜索，仅仅考虑点到点的直线路径即可

目前的设计方案中暂时不实现evtol,只实现uav即可

可以暂时取消airsim traffic中的map manager,暂时不考虑地图，考虑完全空旷的场景

与airsim中保持一致，所有飞机在同一个高度层运行

需要单独给traffic设置一个配置的命名空间，可以命名为traffic,其中包括飞机数量，飞机模型，所用控制器，飞行范围，飞行高度，飞行速度之类的基本配置

整体迁移的功能并不复杂，关键要考虑的是，现在是在isaac sim中，基本所有变量都是张量，计算要满足张量的要求，可以批量计算的尽量批量计算以发挥isaac sim的优势

## aircraft的实现

核心的aircraft靠omni_drones的MultirotorBase类omni_drones/robots/drone/multirotor.py
来实现，采用与omni drones相似的配置，初始化这个类

但是不要采用原来的命名空间，原来的飞机都是在/World/envs/env_0下的，这样环境克隆的时候会复制出来; 

现在air traffic的飞机不需要被复制，所有env使用同样的traffic的飞机，将其命名在/World/Traffic下，这样就不会被复制了

所有的飞机依然采用速度控制器来实现，也就是通过某种policy输出一个速度指令给飞机来执行。因此可以考虑用/home/liang/Projects/OmniDrones/omni_drones/controllers/lee_position_controller.py中的控制器

目前不需要考虑躲避功能的实现，也即不需要实现与ORCA相关的功能，所有飞机径直飞向target goal即可

特别注意！！目前只是一个初步的transfer,复杂的功能不需要实现，我现在需要的就是一个不会被env复制，在isaac sim中朝着各自的goal飞行的飞机集群即可，之后的复杂功能我会逐渐复现，而不是现在

请你在设计时考虑好整个项目的架构，方便我未来扩展功能，没有必要完全照搬airsim的架构，因为他的版本比较久，设计可能比较混乱。现在趁此转移到isaac sim的机会，可以重新设计架构

## 接口

整个traffic simulator的插件（类）需要有一些接口方便外部的env访问一些信息

具体包括访问所有飞机的位置，速度，半径之类的状态;

最好还有一个API接口，输入外部env中的飞机的位置，计算是否与simulator中的飞机发生碰撞（radius小），其输入可能是batch类型的外部env的所有的飞机，以及一个radius（假设外部所有飞机的半径相同），然后通过高效的tensor张量计算与simulator中的飞机做距离是否小于radius的检查; radius其实是safety space的概念，可以用safetyspace表达，而不是之前的radius容易引起歧义