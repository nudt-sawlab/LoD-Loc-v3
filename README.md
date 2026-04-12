# LoD-Loc v3: Generalized Aerial Localization in Dense Cities using Instance Silhouette Alignment

[Project Page](https://pppppsb.github.io/LoD-Locv3/) | [Chinese README](README_CN.md) | [Repository](https://github.com/pppppsb/LoD-Locv3)

LoD-Loc v3 is a research codebase for aerial visual localization over low-detail city models, with a focus on cross-scene generalization and instance silhouette alignment in dense urban environments.

![LoD-Loc v3 teaser](assets/teaser.png)

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
  title={LoD-Loc v3: Generalized Aerial Localization in Dense Cities using Instance Silhouette Alignment},
  author={Peng, Shuaibang and Zhu, Juelin and Li, Xia and Yan, Shen and Yang, Kun and Zhang, Maojun and Liu, Yu},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## License

This repository is released under the MIT License. See the `LICENSE` file for details.
