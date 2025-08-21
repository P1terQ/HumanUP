# Usage Instructions
## Training && Playing Policy
First, please go to the scripts folder
``` bash
cd legged_gym/legged_gym/scripts
```
### 1. Stage I Discovery Policy Training
#### 1.1 Getting Up Policy
- Training:
``` bash
bash run.sh g1waist [your_exp_desc] [device]
# e.g. bash run.sh g1waist stage1_get_up cuda:0
```
- Evaluation:
``` bash
bash eval.sh g1waist [your_exp_desc] [checkpoint]
# e.g. bash eval.sh g1waist stage1_get_up -1
```

#### 1.2 Rolling Over Policy
- Training:
``` bash
bash run.sh g1waistroll [your_exp_desc] [device]
# bash run.sh g1waistroll stage1_roll_over cuda:0
```
- Evaluation:
``` bash
bash eval.sh g1waistroll [your_exp_desc] [checkpoint]
# e.g. bash eval.sh g1waist stage1_roll_over -1
```

For the main training args:
+ `--debug` disables wandb and sets the number of environments to 64, which is useful for debugging;
+ `--fix_action_std` fixes the action std, this is useful for stablizing training;
+ `--resume` indicates whether to resume from the previous experiment;
+ `--resumeid` specifies the exptid to resume from (if resume is set true);

For the main evaluation args:
+ `--record_video` allows you to record video headlessly, this is useful for sever users;
+ `--checkpoint [int]` specifies the checkpoint to load, this is default set as -1, which is the latest one;
+ `--use_jit` use jit model to play;
+ `--teleop_mode` allows the user to control the robot with the keyboard;


### 2. Stage II Deployable Policy Training
#### 2.1 Log the Stage I policy trajectory
```bash
sh log.sh g1waistroll [your_exp_desc] [checkpoint]  # getting up policy
sh log.sh g1waist [your_exp_desc] [checkpoint]  # rolling over policy
```
Then, please put all trajectories under the `simulation/legged_gym/logs/env_logs`, the structure looks like:
```bash
.
└── env_logs
    ├── getup_traj
    │   ├── dof_pos_all.pkl
    │   └── head_height_all.pkl
    └── rollover_traj
        ├── dof_pos_all.pkl
        ├── head_height_all.pkl
        └── projected_gravity_all.pkl
```

To help further development over our HumanUP, we provide our discovered trajectories on [Google Drive](https://drive.google.com/drive/folders/1kRSGkMDnqsX6OLr7-8OM5R6bF9mn84sK?usp=sharing). Feel free to download it to directly train Stage II policy.

#### 2.2 Getting Up Tracking
- Training:
``` bash
bash run_track.sh g1waist [your_exp_desc] [device] [traj_name]
# e.g. bash run_track.sh g1waist stage2_get_up cuda:0 getup_traj
```
- Evaluation:
``` bash
bash eval_track.sh g1waist [your_exp_desc] [checkpoint] [traj_name]
# e.g. bash eval_track.sh g1waist stage2_get_up -1 getup_traj
```

#### 2.3 Rolling Over Tracking
- Training:
``` bash
bash run_track.sh g1waistroll [your_exp_desc] [device] [traj_name]
# bash run_track.sh g1waistroll stage2_roll_over cuda:0 rollover_traj
```
- Evaluation:
``` bash
bash eval_track.sh g1waistroll [your_exp_desc] [checkpoint] [traj_name]
# e.g. bash eval_track.sh g1waist stage2_roll_over -1 rollover_traj
```

## 3. Save jit model
```bash
# bash to_jit.sh g1waist [your_exp_desc] # e.g. bash to_jit.sh g1waist pretrained_exp
python save_jit.py --proj_name g1waist_track --exptid [your_exp_desc] --checkpoint [checkpoint] --robot g1
# e.g. python save_jit.py --proj_name g1waist_track --exptid stage2_get_up --checkpoint -1 --robot g1
```

You can specify which checkpoint exactly to save by adding `--checkpoint [int]` to the command, this is default set as -1, which is the latest one.

You can display the jit policy by adding `--use_jit` in the eval script.

# Notes
There are some useful notes:

## Simulation Frequency
The simulation frequency has a huge impact on the performance of the policy. Most existing codebases for humanoid robots or quadruped robots use a sim frequency of 200Hz. This is enough for locomotion tasks like walking. For the getting up policy learning, we use a higher frequency of 1k Hz (`dt=0.001`). Although you can train a reasonable policy in simulation under 200Hz, but it will not work in the real world.

## Collision Mesh
For the G1 humanoid robots, we have customized the original G1's collision mesh to simplified and modified collision mesh so that we can accelerate training and improve Sim2Real performance. 
- **[g1_29dof_fixedwrist_custom_collision.urdf](./legged_gym/resources/robots/g1_modified/g1_29dof_fixedwrist_custom_collision.urdf)**:
Simplified collision mesh, 23 DoFs G1 with wrists' DoFs removed.
- **[g1_29dof_fixedwrist_custom_collision_with_head.urdf](./legged_gym/resources/robots/g1_modified/g1_29dof_fixedwrist_custom_collision_with_head.urdf)**: 
Simplified collision mesh with the head (better for training rolling over), 23 DoFs G1 with wrists' DoFs removed.
- **[g1_29dof_fixedwrist_custom_collision.urdf](./legged_gym/resources/robots/g1_modified/g1_29dof_fixedwrist_full_collision.urdf)**:
Full collision mesh, 23 DoFs G1 with wrists' DoFs removed.