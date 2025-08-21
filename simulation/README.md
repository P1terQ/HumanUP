# Usage Instructions for <HumanUp>
## Training && Playing Policy
First, please go to the scripts folder
``` bash
cd legged_gym/legged_gym/scripts
```
### 1. Stage I Discovery Policy Training
#### 1.1 Getting Up Policy
- Training:
``` bash
cd simulation/legged_gym/legged_gym/scripts
python train.py --task=go2up  
```
- Evaluation:
``` bash
cd simulation/legged_gym/legged_gym/scripts
python play.py --task=go2up  
```

#### 1.2 Rolling Over Policy
- Training:
``` bash
cd simulation/legged_gym/legged_gym/scripts
python train.py --task=go2rollup
```
- Evaluation:
``` bash
cd simulation/legged_gym/legged_gym/scripts
python play.py --task=go2rollup
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
cd simulation/legged_gym/legged_gym/scripts
python log_traj.py 
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

#### 2.2 Getting Up Tracking
- Training:
``` bash
cd simulation/legged_gym/legged_gym/scripts
python train.py --task=go2_track
```
- Evaluation:
``` bash
cd simulation/legged_gym/legged_gym/scripts
python play.py --task=go2_track
```

#### 2.3 Rolling Over Tracking
- Training:
``` bash
cd simulation/legged_gym/legged_gym/scripts
python train.py --task=go2roll_track
```
- Evaluation:
``` bash
cd simulation/legged_gym/legged_gym/scripts
python play.py --task=go2roll_track
```

# Usage Instructions for <Host>
