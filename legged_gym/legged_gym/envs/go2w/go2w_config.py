from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class GO2WRoughCfg(LeggedRobotCfg):
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'heightfield'
        curriculum = True
        selected = False

        num_rows = 10   # 10 difficulty levels (one per row)
        num_cols = 10   # 10 parallel envs per level

        # All terrain cells are low walls
        terrain_proportions = [0.0, 0.0, 0.0, 0.0, 1.0]
        low_wall_share_in_discrete = 1.0

        # Enable difficulty-based wall height
        low_wall_curriculum = True
        low_wall_height_min = 0.10   # level 1  → 10 cm
        low_wall_height_max = 0.45   # level 10 → 45 cm
        low_wall_thickness = 0.05    # always 5 cm
        low_wall_side_margin = 0.5

    class commands(LeggedRobotCfg.commands):
        # max_ang_vel_yaw = 3.0

        class ranges(LeggedRobotCfg.commands.ranges):
            ang_vel_yaw = [-3.5, 3.5]
            limit_vel_yaw = [-3.5, 3.5]
            lin_vel_x = [-2.0,3.0]  # min max [m/s]
            lin_vel_y = [-2.0, 3.0]  # min max [m/s]
            # ang_vel_yaw = [-0.4, 0.4]    # min max [rad/s]
            heading = [-3.14, 3.14]
            limit_vel_x = [-2.0, 3.0]
            limit_vel_y = [-2.5, 3.0]
            # limit_vel_yaw = [-1.0, 1.0]
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
        terminate_after_contacts_on = []    
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        
    class rewards(LeggedRobotCfg.rewards):
        class scales:
            termination = -0.
            tracking_lin_vel = 2.
            tracking_ang_vel = 1.
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            torques = -1e-5         # 扭矩惩罚，鼓励节能
            torques_wheel = -1e-6   # 轮子扭矩惩罚，鼓励轮子节能
            power = -2e-5       
            power_wheel = -2e-6
            dof_vel = -1e-4         # 关节速度惩罚，鼓励平滑动作
            dof_vel_wheel = -5e-7
            dof_acc = -2.5e-7
            dof_acc_wheel = -2.5e-9
            collision = -0.88     
            feet_contact_forces = -1.5e-4   #接触力惩罚，鼓励轻柔接触
            hip_limit = -0.00
            action_rate = -0.01
            action_smoothness = -0.001
            stand_still = -1.
            dof_pos_limits = -0.1
            dof_vel_limits = -2
            torque_limits = -2

        # if true negative total rewards are clipped at zero (avoids early
        # termination problems)
        only_positive_rewards = False
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        base_height_target = 0.43


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
