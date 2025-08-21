# Installation Instruction
This document provides the instructions on codebase installation. We recommend using [Anaconda](https://www.anaconda.com/) to simplify the process.

## Create a Conda Enviroment
```bash
conda create -n humanup python=3.8
conda activate humanup
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
## IsaacGym
Download [IsaacGym Preview 4.0](https://developer.nvidia.com/isaac-gym) from [Google Drive](https://drive.google.com/file/d/1YEsZPtmdzQbSePX0WMhdf0565XwBaIFi/view?usp=sharing), then install it by running
```bash
cd isaacgym/python && pip install -e .
```
**Note:** NVIDIA preserves all rights of [IsaacGym](https://developer.nvidia.com/isaac-gym).
After installing IsaacGym, please make sure it is working by running,
```bash
# this example can only be ran with a monitor 
python examples/joint_monkey.py
```

## RSL RL
Install `rsl_rl` by running
```bash
cd ../../rsl_rl && pip install -e .
```

## Legged Gym and Other
Install `legged_gym` and other dependencies
```bash
cd ../legged_gym && pip install -e .
pip install "numpy==1.23.0" pydelatin wandb tqdm opencv-python pymeshlab ipdb pyfqmr flask dill gdown hydra-core mujoco mujoco-python-viewer loguru
pip install -r requirements.txt
pip install imageio[ffmpeg]
```
If you cannot install `imageio[ffmpeg]`, please run
```bash
pip install imageio imageio-ffmpeg
```
Next, please follow the running instruction to test a running.