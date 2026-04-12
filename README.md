# LoD-Loc v3：基于实例轮廓对齐的密集城市无人机定位

[项目主页](https://pppppsb.github.io/LoD-Locv3/)

LoD-Loc v3 是一个面向低细节城市场景模型的无人机视觉定位研究代码仓库。相较于 LoD-Loc v2，本版本重点解决两个问题：

- 跨场景泛化能力不足
- 密集城市建筑区域中的定位歧义问题

本仓库的公开整理方式参考了 `LoD-Loc-v2-main`：保留代码、配置、说明文档和示意图片，将本地实验数据、大型模型文件和运行缓存从公开仓库中分离。

![LoD-Loc v3 teaser](assets/teaser.png)

## 仓库说明

- 当前公开主入口是 `refine_blender_Japan_07.sh`
- 当前公开版本默认不附带 `data/`、`Ins_data/`、`model/` 等大体积实验资源
- 如果你要完整复现实验，需要手动准备这些本地资源，并按下文的目录结构放置

## 目录结构

- `config/`：渲染和实验配置文件
- `gloc/`、`gloc_roma/`、`lib/`、`utils/`：核心定位与渲染代码
- `assets/`：README 和项目页使用的示意图
- `refine_blender_Japan_07.sh`：公开测试入口
- `OPEN_SOURCE_GUIDE_CN.md`：中文上传与发布步骤说明

## 环境配置

建议在 Linux 或 WSL 环境中运行。

推荐安装方式：

```bash
conda create -n lodlocv3 python=3.10 -y
conda activate lodlocv3
pip install -r requirements.txt
```

你还需要：

- 与 CUDA 环境匹配的 PyTorch
- Blender 3.3 或兼容版本

默认情况下，`config/config_RealTime_render_1_Japan_07.json` 中的 `blender_path` 使用的是 `blender`，因此需要 Blender 已经加入系统环境变量，或者你自行把该字段改成 Blender 可执行文件的绝对路径。

## 快速测试

当前公开仓库推荐使用下面这个入口：

```bash
bash ./refine_blender_Japan_07.sh
```

它会依次运行：

- `refine_pose_realtime_area.py`
- `refine_pose_realtime_score.py`

## 需要你本地额外准备的资源

下面这些内容默认不随公开仓库一起上传，但如果你要完整运行 `Japan_07` 实验，就需要自己准备：

- `data/UAVD4L-LoD/Japan_07/GPS_pose_new_all.txt`
- `data/blender_origin_zero.xml`
- `Ins_data/Japan_07/PT_640_360_09091800/conf_0.3`
- `model/Japan/Japan_07/Tokeyo_Dingmu_viewpoints_topology_clustered.blend`

也就是说，公开仓库是“代码仓库”，完整实验依赖的本地大文件需要你单独保存和管理。

## 为什么不直接把大文件放进 GitHub

因为这类文件通常存在下面几个问题：

- 体积很大，容易超过 GitHub 普通仓库的限制
- 其中一部分只适合本地实验，不适合长期版本管理
- 数据、模型、`.blend` 场景文件通常更适合走网盘、GitHub Releases 或 Git LFS

## 开源发布建议

如果你是这个项目的维护者，建议这样管理：

1. `LoD-Loc v3_all` 保留完整实验环境
2. `LoD-Loc v3` 作为公开代码仓库
3. 后续只把 `LoD-Loc v3` 上传到 GitHub
4. 将数据、实例分割结果和 Blender 大文件单独分发

具体上传步骤请看 [OPEN_SOURCE_GUIDE_CN.md](OPEN_SOURCE_GUIDE_CN.md)。

## 致谢

LoD-Loc v3 延续了 LoD-Loc / MC-Loc 的研究路线，并采用了 Blender 渲染流程来支持公开代码整理与复现。

## 引用

如果这份代码或思路对你的研究有帮助，请引用：

```bibtex
@article{peng2026lodlocv3,
  title={LoD-Loc v3: Generalized Aerial Localization in Dense Cities using Instance Silhouette Alignment},
  author={Peng, Shuaibang and Zhu, Juelin and Li, Xia and Yan, Shen and Yang, Kun and Zhang, Maojun and Liu, Yu},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## 许可证

在真正公开 GitHub 仓库之前，建议补一个正式的 `LICENSE` 文件。对科研代码来说，`MIT` 是最常见也最省心的选择；如果你希望保留更明确的专利授权条款，可以选择 `Apache-2.0`。
