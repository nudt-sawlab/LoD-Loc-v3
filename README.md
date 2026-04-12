# LoD-Loc v3: Generalized Aerial Localization in Dense Cities using Instance Silhouette Alignment

[Project Page](https://pppppsb.github.io/LoD-Locv3/) | [Chinese README](README_CN.md) | [Repository](https://github.com/pppppsb/LoD-Locv3)

LoD-Loc v3 is a research codebase for aerial visual localization over low-detail city models. Compared with LoD-Loc v2, this version focuses on:

- better cross-scene generalization
- reducing pose ambiguity in dense urban regions through instance silhouette alignment

This public repository is organized with `LoD-Loc-v2-main` as the reference style: code, configs, documentation, and figures stay in GitHub, while large local experiment assets are distributed separately.

![LoD-Loc v3 teaser](assets/teaser.png)

## Overview

- Main public entry: `refine_blender_Japan_07.sh`
- This repository does not directly include large local experiment assets such as `data/`, `Ins_data/`, or `model/`
- Full reproduction requires downloading those assets separately and placing them into the expected local paths

## Repository Structure

- `config/`: rendering and experiment configurations
- `gloc/`, `gloc_roma/`, `lib/`, `utils/`: localization and rendering code
- `assets/`: figures used by the README and project page
- `refine_blender_Japan_07.sh`: public test entry
- `README_CN.md`: Chinese project introduction
- `OPEN_SOURCE_GUIDE_CN.md`: Chinese upload and release guide

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

## Quick Test

The recommended public entry is:

```bash
bash ./refine_blender_Japan_07.sh
```

It runs:

- `refine_pose_realtime_area.py`
- `refine_pose_realtime_score.py`

## External Assets

The following assets are required for the `Japan_07` experiment but are intentionally not stored in the Git repository:

- `data/UAVD4L-LoD/Japan_07/GPS_pose_new_all.txt`
- `data/blender_origin_zero.xml`
- `Ins_data/Japan_07/PT_640_360_09091800/conf_0.3`
- `model/Japan/Japan_07/Tokeyo_Dingmu_viewpoints_topology_clustered.blend`

These assets are intended to be distributed through GitHub Releases instead of normal Git history.

## GitHub Releases

The large experiment assets should be published as release attachments in this repository:

- `data/`
- `Ins_data/`
- `model/`

Recommended practice:

1. Compress each asset group separately, for example `data.zip`, `Ins_data.zip`, and `model.zip`
2. Create a GitHub Release for a tagged version of the code
3. Upload the zip files as release assets
4. Mention the expected extraction paths in the release notes

This keeps the Git repository lightweight while still allowing full local reproduction.

## Project Maintenance

Recommended workflow:

1. Keep `LoD-Loc v3_all` as your full local experiment workspace
2. Keep `LoD-Loc v3` as the public code repository
3. Publish code changes from `LoD-Loc v3`
4. Publish large assets through GitHub Releases

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

Before making the repository fully public, add a formal `LICENSE` file. For research code, `MIT` is usually the simplest choice; `Apache-2.0` is also a good option if you want explicit patent-related terms.
