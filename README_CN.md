<p align="center">
  <h1 align="center"><ins>LoD-Loc v3</ins>:<br>基于实例轮廓对齐的密集城市无人机定位</h1>
  <p align="center">
    <h>Peng&nbsp;Shuaibang</h>
    ·
    <h>Zhu&nbsp;Juelin</h>
    ·
    <h>Li&nbsp;Xia</h>
    ·
    <h>Yan&nbsp;Shen</h>
    ·
    <h>Yang&nbsp;Kun</h>
    ·
    <h>Zhang&nbsp;Maojun</h>
    ·
    <h>Liu&nbsp;Yu</h>
  </p>
  <h2 align="center">CVPR 2026</h2>

  <h3 align="center">
    <a href="https://pppppsb.github.io/LoD-Locv3/">项目主页</a>
    | <a href="README.md">English README</a>
    | <a href="https://github.com/pppppsb/LoD-Locv3/releases">Releases</a>
  </h3>
  <div align="center"></div>
</p>
<p align="center">
    <a href="assets/teaser.png"><img src="assets/teaser.png" alt="teaser" width="100%"></a>
    <br>
    <em>LoD-Loc v3 通过实例轮廓对齐，面向密集城市无人机定位中的跨场景泛化与歧义问题提出解决方案。</em>
</p>

本仓库是论文 “LoD-Loc v3: Generalized Aerial Localization in Dense Cities using Instance Silhouette Alignment” 的实现代码。

## 重要说明

我们非常感谢研究社区对 LoD-Loc v2 项目的关注。需要说明的是，论文中涉及的 OSG 渲染技术受到项目知识产权限制，因此本代码仓库采用了基于 Blender 的渲染流程作为替代实现。

## 仓库说明

- 当前公开主入口是 `refine_blender_Japan_07.sh`
- 当前仓库包含代码、配置和说明文档
- 大体积实验资源通过 [GitHub Releases](https://github.com/pppppsb/LoD-Locv3/releases) 单独提供

## 目录结构

- `config/`：渲染和实验配置文件
- `gloc/`、`gloc_roma/`、`lib/`、`utils/`：核心定位与渲染代码
- `assets/`：README 和项目页使用的示意图
- `refine_blender_Japan_07.sh`：公开测试入口
- `README_CN.md`：中文说明文档

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

## 快速开始

1. 安装环境和依赖。
2. 从 [GitHub Releases](https://github.com/pppppsb/LoD-Locv3/releases) 下载 `data.zip`、`Ins_data.zip` 和 `model.zip`。
3. 将它们解压到仓库根目录，得到 `data/`、`Ins_data/` 和 `model/`。
4. 运行：

```bash
bash ./refine_blender_Japan_07.sh
```

该入口脚本会依次运行：

- `refine_pose_realtime_area.py`
- `refine_pose_realtime_score.py`

## 所需外部资源

运行 `Japan_07` 实验需要以下资源：

- `data/UAVD4L-LoD/Japan_07/GPS_pose_new_all.txt`
- `data/blender_origin_zero.xml`
- `Ins_data/Japan_07/PT_640_360_09091800/conf_0.3`
- `model/Japan/Japan_07/Tokeyo_Dingmu_viewpoints_topology_clustered.blend`

这些文件通过 GitHub Releases 提供，而不直接放入 Git 历史。

## 致谢

本项目实现主要基于以下开源仓库，感谢原作者的优秀工作：

- [LoD-Loc v2](https://github.com/VictorZoo/LoD-Loc-v2)
- [RSPrompter](https://github.com/KyanChen/RSPrompter)

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

本仓库采用 MIT License，详细内容请见 `LICENSE` 文件。
