<p align="center">
  <h1 align="center"><ins>LoD-Loc v3</ins>:<br>Generalized Aerial Localization in Dense Cities using Instance Silhouette Alignment</h1>
  <p align="center">
    <h>Shuaibang&nbsp;Peng</h>
    ·
    <h>Juelin&nbsp;Zhu</h>
    ·
    <h>Xia&nbsp;Li</h>
    ·
    <h>Shen&nbsp;Yan</h>
    ·
    <h>Kun&nbsp;Yang</h>
    ·
    <h>Maojun&nbsp;Zhang</h>
    ·
    <h>Yu&nbsp;Liu</h>
  </p>
  <h2 align="center">CVPR 2026</h2>

  <h3 align="center">
    <a href="https://nudt-sawlab.github.io/LoD-Locv3/">Project Page</a>
    | <a href="https://arxiv.org/pdf/2603.19609">Paper</a>
    | <a href="https://www.youtube.com/watch?v=pU2_JjncLPg">Demo</a>
  </h3>
  <div align="center"></div>
</p>
<p align="center">
    <a href="assets/teaser.png"><img src="assets/teaser.png" alt="teaser" width="100%"></a>
    <br>
    <em>LoD-Loc v3 addresses cross-scene generalization and ambiguity in dense urban aerial localization through instance silhouette alignment.</em>
</p>

This repository is an implementation of the paper "LoD-Loc v3: Generalized Aerial Localization in Dense Cities using Instance Silhouette Alignment".

## Important Things

We highly appreciate the research community's interest in the LoD-Loc v3 project. Please note that the OSG rendering technique described in the paper involves project intellectual property constraints. Consequently, we have implemented a Blender-based rendering pipeline as a substitute in this codebase.

## Overview

- Main public entry: `refine_blender_Japan_07.sh`
- This repository contains code, configs, and documentation
- Large experiment assets are distributed separately through [GitHub Releases](https://github.com/pppppsb/LoD-Locv3/releases)

## Repository Structure

- `config/`: rendering and experiment configurations
- `gloc/`, `gloc_roma/`, `lib/`, `utils/`: localization and rendering code
- `assets/`: figures used by the README and project page
- `refine_blender_Japan_07.sh`: public test entry
- `README_CN.md`: Chinese project introduction

## Environment

Linux or WSL is recommended.

Suggested setup:

```bash
conda create -n lodlocv3 python=3.10 -y
conda activate lodlocv3
pip install -r requirements.txt
```

You will also need:

- a PyTorch build compatible with your CUDA environment
- Blender 3.3 or a compatible version

By default, `config/config_RealTime_render_1_Japan_07.json` uses `blender` as the executable name. If Blender is not in your `PATH`, replace that field with the absolute path to your Blender executable.

## Quick Start

1. Install the required environment and dependencies.
2. Download `data.zip`, `Ins_data.zip`, and `model.zip` from [GitHub Releases](https://github.com/pppppsb/LoD-Locv3/releases).
3. Extract them to the repository root so that `data/`, `Ins_data/`, and `model/` are created in place.
4. Run:

```bash
bash ./refine_blender_Japan_07.sh
```

This entry script runs:

- `refine_pose_realtime_area.py`
- `refine_pose_realtime_score.py`

## Required External Assets

The `Japan_07` experiment expects the following assets:

- `data/UAVD4L-LoD/Japan_07/GPS_pose_new_all.txt`
- `data/blender_origin_zero.xml`
- `Ins_data/Japan_07/PT_640_360_09091800/conf_0.3`
- `model/Japan/Japan_07/Tokeyo_Dingmu_viewpoints_topology_clustered.blend`

These files are distributed through GitHub Releases instead of normal Git history.

## Acknowledgments

Our implementation is mainly based on the following repositories. Thanks to their authors.

- [LoD-Loc v2](https://github.com/VictorZoo/LoD-Loc-v2)
- [RSPrompter](https://github.com/KyanChen/RSPrompter)

## Citation

If this project helps your research, please cite:

```bibtex
@article{peng2026lodlocv3,
  title={{LoD-Loc} v3: Generalized Aerial Localization in Dense Cities using Instance Silhouette Alignment},
  author={Peng, Shuaibang and Zhu, Juelin and Li, Xia and Yang, Kun and Zhang, Maojun and Liu, Yu and Yan, Shen},
  journal={arXiv preprint arXiv:2603.19609},
  year={2026}
}
```

## License

This repository is released under the MIT License. See the `LICENSE` file for details.
