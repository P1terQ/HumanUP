<h1 align="center">Learning Getting-Up Policies for <br>Real-World Humanoid Robots
</h1>


<p align="center">
    <a href="https://xialin-he.github.io/"><strong>Xialin He<sup>*,1</sup></strong></a>
    |
    <a href="https://runpeidong.web.illinois.edu/"><strong>Runpei Dong<sup>*,1</sup></strong></a>
    |
    <a href="https://zixuan417.github.io/"><strong>Zixuan Chen<sup>2</sup></strong></a>
    |
    <a href="https://saurabhg.web.illinois.edu/"><strong>Saurabh Gupta<sup>1</sup></strong></a>
    <br>
    <sup>1</sup>University of Illinois Urbana-Champaign
    &nbsp
    <sup>2</sup>Simon Fraser University
    <br>
    * Equal Contribution
    <h3 align="center"><strong><span style="color: #9C2018;">RSS 2025</span></strong></h3>
</p>

<p align="center">
    <a href="https://humanoid-getup.github.io/"><img src="https://img.shields.io/badge/Project-Page-green.svg"></a>
    <a href="https://arxiv.org/abs/2502.12152"><img src="https://img.shields.io/badge/Paper-PDF-orange.svg"></a>
    <a href="https://www.youtube.com/watch?v=NPMUlf9soL0&t=5s"><img src="https://img.shields.io/badge/Youtube-Video-red.svg"></a>
    <a href="https://www.bilibili.com/video/BV1pJA8epExA/?share_source=copy_web&vd_source=fc2c8db46b5b7afce6757bf7c9ccec0b"><img src="https://img.shields.io/badge/Bilibili-Video-blue.svg"></a>
    <a href="https://github.com/RunpeiDong/humanup/blob/master/LICENSE"><img src="https://img.shields.io/badge/Code%20License-Apache_2.0-yellow.svg"></a>
    <img src="./poster.gif" width="80%">
</p>

## News
- 🎉 Apr 2025: HumanUP has been accepted by RSS 2025

## HumanUP
**[HumanUP](https://arxiv.org/abs/2502.12152)** is an RL learning framework for training humanoid robots to get up from supine (i.e., lying face up) or prone (i.e., lying face down) poses. This codebase is initially built for the code release of this **[HumanUP](https://arxiv.org/abs/2502.12152)** paper, which supports simulation training of **Unitree G1** humanoid robot. The simulation training is based on **Isaac Gym**.

## Installation
See [installation instructions](./docs/install.md).

## Getting Started
See [usage instructions](./simulation/README.md).

## Change Logs
See [changelogs](./docs/changelog.md).


## Acknowledgements
+ We would like to thank all the authors in this project, this project cannot be finished without your efforts!
+ Our simulation environment implementation is based on [legged_gym](https://github.com/leggedrobotics/legged_gym), and the rl algorithm implementation is based on [rsl_rl](https://github.com/leggedrobotics/rsl_rl).
+ [Smooth-Humanoid-Locomotion](https://github.com/zixuan417/smooth-humanoid-locomotion) also provide lots of insights.

## Citation
If you find this work useful, please consider citing:
```
@article{humanup25,
  title={Learning Getting-Up Policies for Real-World Humanoid Robots},
  author={He, Xialin and Dong, Runpei and Chen, Zixuan and Gupta, Saurabh},
  journal={arXiv preprint arXiv:2502.12152},
  year={2025}
}
```

