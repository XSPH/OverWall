from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class GO2WRoughCfg(LeggedRobotCfg):
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        curriculum = True
        selected = False

        num_rows = 10   # 10 difficulty levels (one per row)
        num_cols = 10   # 10 parallel envs per level

        # All terrain cells are low walls
        terrain_proportions = [0.0, 0.0, 0.0, 0.0, 1.0]
        low_wall_share_in_discrete = 1.0

        # Enable difficulty-based wall height
        low_wall_curriculum = True
        low_wall_height_min = 0.0    # level 0 → flat (learn forward first)
        low_wall_height_max = 0.45   # level 9 → 35 cm (covers 30cm target)
        low_wall_thickness_min = 0.30   # level 0 → 30 cm (thick, easy)
        low_wall_thickness_max = 0.05   # level 9 → 5 cm (thin, target)
        low_wall_side_margin = 0.0   # wall spans full width, no gaps to bypass

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        zero_command_prob = 0.0  # never stand still — always move forward

        class ranges(LeggedRobotCfg.commands.ranges):
            # forward-only: narrow lateral + yaw, broad forward range
            lin_vel_x = [0.3, 2.0]     # only forward [m/s]
            lin_vel_y = [0.0, 0.0]     # zero lateral — pure forward
            ang_vel_yaw = [0.0, 0.0]   # zero yaw — no turning
            heading = [0.0, 0.0]
            limit_vel_x = [0.3, 2.0]
            limit_vel_y = [0.0, 0.0]
            limit_vel_yaw = [0.0, 0.0]
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
            tracking_lin_vel = 2.0        # main driving signal
            tracking_ang_vel = 0.       # reduced — forward is priority
            lin_vel_z = -1.0              # penalize vertical bounce
            ang_vel_xy = -0.05            # penalize roll/pitch rate

            # === wall crossing ===
            # wall_front_lift = 2.0         # reward lifting front wheels to wall height
            # wall_progress = 1.0           # reward COM forward progress when near wall
            # wall_crossed = 10.0           # sparse bonus for crossing the wall center
            # wall_height_gain = 0.5        # reward base height approaching wall height

            # === energy efficiency (keep low) ===
            torques = -1e-5               # penalize leg torque
            torques_wheel = -1e-6         # penalize wheel torque (weaker)
            power = -2e-5                 # penalize leg power
            power_wheel = -2e-6           # penalize wheel power (weaker)
            dof_vel = -1e-4               # penalize leg joint velocity
            dof_vel_wheel = -5e-7         # penalize wheel velocity (weaker)
            dof_acc = -2.5e-7             # penalize leg acceleration
            dof_acc_wheel = -2.5e-9       # penalize wheel acceleration (weaker)

            # === smoothness ===
            action_rate = -0.01           # penalize action changes
            action_smoothness = -0.001    # penalize action jitter

            # === constraints (relaxed for climbing) ===
            dof_pos_limits = -10          # hard limit penalty
            dof_vel_limits = -2           # velocity limit penalty
            torque_limits = -2            # torque limit penalty
            orientation = -0.02           # relaxed: allow ~30° tilt during climb
            base_height = -0.5            # relaxed: height changes during climb
            stand_still = -1.0            # penalize motion at zero command
            collision = -0.2              # relaxed: thighs may touch wall
            feet_contact_forces = -1e-4   # light contact penalty
            hip_limit = -0.01             # small hip centering

            # === disabled (conflict with wall crossing) ===
            stumble = 0.0               # feet WILL hit vertical wall face
            feet_stumble = 0.0          # same reason
            feet_regulation = 0.0       # leg height pattern breaks during climb
            feet_air_time = 0.0         # gait pattern changes during climb
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
