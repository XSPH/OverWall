from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class GO2WRoughCfg(LeggedRobotCfg):
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        selected = False

    #     # === wall-crossing terrain ===
    #     num_rows = 10
    #     num_cols = 10
    #     terrain_proportions = [0.0, 0.0, 0.0, 0.0, 1.0]
    #     low_wall_share_in_discrete = 1.0
    #     low_wall_curriculum = True
    #     low_wall_height_min = 0.0
    #     low_wall_height_max = 0.45
    #     low_wall_thickness_min = 0.30
    #     low_wall_thickness_max = 0.05
    #     low_wall_side_margin = 0.0

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        # zero_command_prob = 0.0  # wall: never stand still

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1, 1]
            lin_vel_y = [-1, 1]
            ang_vel_yaw = [-1, 1]
            heading = [-3.14, 3.14]

            # # wall: forward-only
            # lin_vel_x = [0.3, 2.0]
            # lin_vel_y = [0.0, 0.0]
            # ang_vel_yaw = [0.0, 0.0]
            # heading = [0.0, 0.0]

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

        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/wheel_dog2/urdf/wheel_dog2.urdf'


        name = "go2w"
        foot_name = "foot"
        # foot_name = "wheel_solid"

        penalize_contacts_on = ["thigh","motor", "calf","base","hip"]
        terminate_after_contacts_on = []    
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        
    class rewards(LeggedRobotCfg.rewards):
        class scales:
            # === core locomotion ===
            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.5
            lin_vel_z = -1.0
            ang_vel_xy = -0.05

            # === wall crossing ===
            # wall_front_lift = 2.0
            # wall_progress = 1.0
            # wall_crossed = 10.0
            # wall_height_gain = 0.5

            # === energy efficiency ===
            torques = -1e-5
            torques_wheel = -1e-7
            # power = -2e-5
            # power_wheel = -2e-6
            dof_vel = -1e-5
            dof_vel_wheel = -5e-7
            dof_acc = -2.5e-7
            dof_acc_wheel = -2.5e-9

            # === smoothness ===
            action_rate = -0.01
            action_smoothness = -0.001

            # === constraints ===
            dof_pos_limits = -10
            dof_vel_limits = -2
            torque_limits = -2
            orientation = -0.5
            base_height = -10
            stand_still = -1.0
            collision = -1
            feet_contact_forces = -1e-4
            hip_limit = -1

            # === gait quality ===
            feet_air_time = 1.0
            stumble = -0.02
            feet_stumble = -0.2
            # feet_regulation = -0.05

            termination = 0.0

        only_positive_rewards = False
        tracking_sigma = 0.25            # tracking reward = exp(-error^2/sigma)
        base_height_target = 0.43
        soft_dof_pos_limit = 0.9         # percentage of urdf limits
        max_orientation = 300
        max_contact_force = 300.         # forces above this value are penalized



class GO2WRoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'reinforce'

        experiment_name = 'go2_load_teacher_student_phase_model_a'
