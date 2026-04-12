# LoD-Loc v3 中文上传指南

这份文档是按你当前本地目录状态写的。

现在你的两个目录已经分开了：

- `C:\Users\27490\Desktop\kaiyuan\LoD-Loc v3_all`
  说明：完整实验目录，保留了你本地实验要用的数据、实例分割结果和 Blender 场景文件
- `C:\Users\27490\Desktop\kaiyuan\LoD-Loc v3`
  说明：公开版代码目录，后续上传 GitHub 只操作这个目录

后面所有 GitHub 上传操作，都在 `LoD-Loc v3` 里完成，不要在 `LoD-Loc v3_all` 里做。

## 第一步：确认当前公开目录内容是否正确

进入公开目录：

```powershell
cd "C:\Users\27490\Desktop\kaiyuan\LoD-Loc v3"
```

你要确认这里面主要是这些内容：

- 代码目录：`config/`、`gloc/`、`gloc_roma/`、`lib/`、`utils/`
- 文档文件：`README.md`、`OPEN_SOURCE_GUIDE_CN.md`
- 入口脚本：`refine_blender_Japan_07.sh`
- 配置和依赖：`requirements.txt`、`configs.py`、`parse_args.py` 等

这里不应该再有：

- `data/`
- `Ins_data/`
- `model/`
- `logs/`
- `blender_realtime_temp/`
- `__pycache__/`

## 第二步：补许可证

在真正公开前，建议补一个 `LICENSE` 文件。

推荐两个选择：

- `MIT`
- `Apache-2.0`

如果你想省事、兼容性高，优先选 `MIT`。

如果你愿意，我下一步可以直接帮你在这个公开目录里生成 `LICENSE`。

## 第三步：初始化 Git 仓库

在 `LoD-Loc v3` 目录下执行：

```powershell
git init
git branch -M main
```

然后查看状态：

```powershell
git status
```

## 第四步：把公开目录内容加入 Git

执行：

```powershell
git add .
git status
```

这一步你主要看两件事：

1. 被加入的确实是公开代码目录中的内容
2. 没有把你本地实验大文件重新带进来

## 第五步：创建第一次提交

如果你的 Git 已经设置过用户名和邮箱，就直接执行：

```powershell
git commit -m "Initial open-source release for LoD-Loc v3"
```

如果 Git 提示没有用户名和邮箱，就先执行：

```powershell
git config --global user.name "你的GitHub用户名"
git config --global user.email "你的GitHub邮箱"
```

然后再提交一次。

## 第六步：在 GitHub 网页上创建空仓库

因为这台机器上没有安装 `gh` 命令行工具，所以最稳的方式是你在浏览器里手动创建一个空仓库。

建议仓库名：

- `LoD-Loc-v3`

注意事项：

- 创建空仓库时，不要勾选自动生成 README
- 不要勾选 `.gitignore`
- 不要勾选 License

因为这些内容我们本地已经准备好了。

## 第七步：绑定远程仓库

在 GitHub 创建空仓库后，复制它给你的 HTTPS 地址，例如：

```text
https://github.com/你的用户名/LoD-Loc-v3.git
```

然后在本地执行：

```powershell
git remote add origin https://github.com/你的用户名/LoD-Loc-v3.git
git push -u origin main
```

## 第八步：如果 GitHub 要求登录

首次推送时，GitHub 可能会要求你认证。

常见情况：

1. 已经在本机配置过 GitHub 凭证
  说明：会直接推送成功
2. 需要浏览器登录
  说明：按照弹出的页面完成登录即可
3. 需要 Personal Access Token
  说明：去 GitHub 设置里创建 Token，再作为密码使用

## 第九步：上传完成后检查

上传成功后，你到 GitHub 仓库首页检查下面几项：

1. 首页是否正常显示中文 `README.md`
2. 是否只包含代码、配置、图片和文档
3. 是否没有把 `data/`、`Ins_data/`、`model/` 这些大目录传上去
4. `refine_blender_Japan_07.sh` 是否还在
5. `config/config_RealTime_render_1_Japan_07.json` 是否已经是公开版路径

## 第十步：后续如何维护

以后你继续做实验时：

- 在 `LoD-Loc v3_all` 里工作
- 需要公开时，把整理后的代码同步到 `LoD-Loc v3`
- 只从 `LoD-Loc v3` 推送 GitHub

不要反过来直接在 GitHub 公开目录里堆实验数据。

## 我现在还能继续帮你做什么

我可以继续帮你做下面两件事中的任意一件：

1. 直接在 `LoD-Loc v3` 里补一个 `LICENSE`
2. 直接在 `LoD-Loc v3` 里初始化 Git，并把第一次提交也做掉

如果你愿意继续，我下一步建议先做第 1 件，再做第 2 件。
