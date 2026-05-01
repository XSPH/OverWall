from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class GO2WRoughCfg(LeggedRobotCfg):
    class terrain(LeggedRobotCfg.terrain):
        # Low-wall terrain is sampled from the discrete-obstacle band.
        # Increase this ratio to focus more on wall-crossing.
        low_wall_share_in_discrete = 0.6
        # Wall geometry in meters.
        low_wall_height = 0.30
        low_wall_thickness = 0.5
        low_wall_side_margin = 0.4

    class commands(LeggedRobotCfg.commands):
        max_ang_vel_yaw = 3.0

        class ranges(LeggedRobotCfg.commands.ranges):
            ang_vel_yaw = [-1.0, 1.0]
            limit_vel_yaw = [-2.0, 2.0]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.5]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.,   # [rad]
            'RL_hip_joint': 0.,   # [rad]
            'FR_hip_joint': 0.,  # [rad]
            'RR_hip_joint': 0.,   # [rad]


            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]

            'FL_foot_joint': 0.,   # [rad]
            'RL_foot_joint': 0.,   # [rad]
            'FR_foot_joint': 0.,   # [rad]
            'RR_foot_joint': 0.,   # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip_joint': 40.,'thigh_joint':40.,'calf_joint':40.,'foot_joint':0}  # [N*m/rad]
        damping = {'hip_joint': 1.0,'thigh_joint':1.0,'calf_joint':1.0,'foot_joint':0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/TOE_dog3/urdf/dog.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2w/urdf/go2w.urdf'

        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/wheel_dog_/urdf/wheel_dog.urdf'


        name = "go2w"
        foot_name = "foot"
        # foot_name = "wheel_solid"

        penalize_contacts_on = ["thigh","motor" "calf","base","hip"]
        terminate_after_contacts_on = ["base"]    
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        
    class rewards(LeggedRobotCfg.rewards):
        class scales:
            termination = -0.
            tracking_lin_vel = 2.
            tracking_ang_vel = 1.
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -0.42
            # large_orientation = -1

            torques = -1e-5         # 扭矩惩罚，鼓励节能
            torques_wheel = -1e-6   # 轮子扭矩惩罚，鼓励轮子节能
            power = -2e-5       
            power_wheel = -2e-6
            dof_vel = -1e-4         # 关节速度惩罚，鼓励平滑动作
            dof_vel_wheel = -5e-7

            # stumble = -0.1
            # feet_regulation = -0.05     # 鼓励足部保持在合理位置，避免过度伸展或收缩
            dof_acc = -2.5e-7
            dof_acc_wheel = -2.5e-9

            base_height = -10.        #维持身体高度 -1
            # feet_air_time = 0.9     #足部离地时间  1
            collision = -0.88          #碰撞惩罚
            # feet_stumble = -0.2
            # feet_height = -0.5
            feet_contact_forces = -1.5e-4   #接触力惩罚，鼓励轻柔接触
            # trap_static = -2.
            hip_limit = -10.
            #low_height
            # thigh_low = -5
            # calf_low = -5

            action_rate = -0.01
            action_smoothness = -0.001
            stand_still = -0.7 
            # stand_still_vel = -2

            dof_pos_limits = -2
            dof_vel_limits = -2
            torque_limits = -2

            # power_distribution = -1e-5
            # trot_gait = -0.05
            centripetal = -5.

        # if true negative total rewards are clipped at zero (avoids early
        # termination problems)
        only_positive_rewards = False
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        base_height_target = 0.44

        soft_dof_pos_limit = 0.8  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.8
        soft_torque_limit = 0.8
        max_orientation = 60
        max_contact_force = 60.  # forces above this value are penalized



class GO2WRoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'reinforce'

        experiment_name = 'go2_load_teacher_student_phase_model_a'
