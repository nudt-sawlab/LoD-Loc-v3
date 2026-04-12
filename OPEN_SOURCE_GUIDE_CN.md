# LoD-Loc v3 中文后续操作指南

这份文档对应你当前已经完成的状态。

## 当前状态

现在你的目录已经分成两套：

- `C:\Users\27490\Desktop\kaiyuan\LoD-Loc v3_all`
  说明：完整实验目录，保留你本地实验要用的数据、实例分割结果和 Blender 场景文件
- `C:\Users\27490\Desktop\kaiyuan\LoD-Loc v3`
  说明：公开版代码目录，已经初始化 Git，并已推送到 GitHub

当前 GitHub 仓库地址：

- `https://github.com/pppppsb/LoD-Locv3`

也就是说，代码仓库这一步已经完成了。你接下来要做的重点，不再是“上传代码”，而是：

1. 补 `LICENSE`
2. 用 GitHub Releases 分发 `data/`、`Ins_data/`、`model/`
3. 检查仓库首页展示是否符合预期

## 第一步：检查 GitHub 仓库首页

先打开：

- `https://github.com/pppppsb/LoD-Locv3`

确认下面几项：

1. 首页默认显示的是英文 `README.md`
2. README 中有到 `README_CN.md` 的链接
3. 仓库里没有 `data/`、`Ins_data/`、`model/` 这些大目录
4. `refine_blender_Japan_07.sh` 还在
5. `config/config_RealTime_render_1_Japan_07.json` 是公开版路径

## 第二步：补许可证

当前仓库还没有正式的 `LICENSE` 文件。

推荐两个选择：

- `MIT`
- `Apache-2.0`

如果你想省事、兼容性高，优先选 `MIT`。

如果你愿意，我下一步可以直接帮你补 `LICENSE`，然后再次推送。

## 第三步：从 `LoD-Loc v3_all` 打包大资源

因为你已经决定把下面三个目录通过 GitHub Releases 分发：

- `data/`
- `Ins_data/`
- `model/`

建议你在 `LoD-Loc v3_all` 中分别打包为：

```text
data.zip
Ins_data.zip
model.zip
```

推荐打包时保持目录层级不变，这样别人下载后可以直接解压到仓库根目录。

## 第四步：在 GitHub 上创建 Release

在仓库页面里操作：

1. 点击 `Releases`
2. 点击 `Draft a new release`
3. 填写版本号，例如 `v1.0.0`
4. Release title 可以写：

```text
LoD-Loc v3 Code and Assets v1.0.0
```

5. 上传：
   - `data.zip`
   - `Ins_data.zip`
   - `model.zip`

## 第五步：写 Release 说明

建议你在 Release 说明中写清楚下面几点：

1. `data.zip` 解压到仓库根目录后得到 `data/`
2. `Ins_data.zip` 解压到仓库根目录后得到 `Ins_data/`
3. `model.zip` 解压到仓库根目录后得到 `model/`
4. `refine_blender_Japan_07.sh` 依赖这些资源
5. 推荐运行环境是 Linux 或 WSL

你可以直接参考下面这段英文：

```text
This release provides the external assets required for the public LoD-Loc v3 code repository.

After downloading, extract:
- data.zip to ./data
- Ins_data.zip to ./Ins_data
- model.zip to ./model

The Japan_07 public entry script `refine_blender_Japan_07.sh` depends on these assets.
Linux or WSL is recommended for reproduction.
```

## 第六步：后续代码怎么维护

以后你继续做实验时，建议一直保持下面这个分工：

1. 在 `LoD-Loc v3_all` 中继续做完整实验
2. 需要公开时，把整理好的代码同步到 `LoD-Loc v3`
3. 只从 `LoD-Loc v3` 推送 GitHub
4. 大资源只走 GitHub Releases，不进入 Git 仓库历史

## 我现在还能继续帮你做什么

我可以继续帮你做下面两件事中的任意一件：

1. 直接在 `LoD-Loc v3` 里补一个 `LICENSE`
2. 帮你再写一份 GitHub Release 的英文说明稿，方便你直接复制粘贴
