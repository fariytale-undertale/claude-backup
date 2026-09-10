---
name: cleanup
description: 任务完成后清理非必要文件（备份/临时/中间版本/缓存），先归档 push 到 GitHub 备份仓库（git@github.com:fariytale-undertale/claude-backup.git）成功后再删本地。用户说「清理」「清除临时文件」「cleanup」、或任务收尾需要收拾现场时使用。
---

# cleanup — 先传后删的清理

把项目里「非任务完成必要」的文件（缓存、备份、中间版本、临时脚本）归档到备份仓库
并 push 上 GitHub，**push 成功后才删本地**。备份安全网永不失效。

## 工具

- 脚本：`cleanup.py`（本 skill 目录）
- 备份仓库本地根：`D:/code-backup`（git 仓库，remote `origin` → claude-backup）
- Python 3.12：`C:/Users/17186/AppData/Local/Programs/Python/Python312/python.exe`

## 流程

### 1. 扫描（只读，必须先行）

```bash
PY312="C:/Users/17186/AppData/Local/Programs/Python/Python312/python.exe"
"$PY312" "C:/Users/17186/.claude/skills/cleanup/cleanup.py" scan --dir "<项目目录>"
```

输出两档清单：
- **缓存类**：可直接清理（__pycache__、latex 编译中间、.tmp 等）
- **备份/中间版类**：*.bak、*~、*_old、*_draft、*_v2、tmp_*、scratch*、*.log 等，**必须经用户确认**

### 2. 用户确认

把两档清单原样展示给用户（缓存类标注「将直接清理」，备份类逐条列给用户过目）。
只有用户点头后才继续。若用户对某个文件有疑虑（可能是最终版/有引用），从清单中排除。

### 3. 执行（先传后删）

```bash
"$PY312" "C:/Users/17186/.claude/skills/cleanup/cleanup.py" run --dir "<项目目录>" --yes-backup
```

- 缓存类 + 已确认的备份类一并归档 → commit → push
- **push 失败：脚本报错并保留本地文件**，如实报告原因（网络/远端冲突），不得强行删除
- push 成功：脚本删除本地原文件

## 安全规则

1. **先传后删**：任何删除都发生在 `git push origin main` 成功之后。
2. **备份类必过确认**：没有用户明确同意，绝不带 `--yes-backup` 删除 *.bak/*_old/*_draft 类文件。
3. **不动备份仓库自身**：绝不清理 `D:/code-backup`，它是安全网。
4. **跳过依赖目录**：node_modules、venv、site-packages、.git 永不触碰。
5. **push 失败不删**：远端备份不可用时，本地文件一律保留，报告阻塞原因。

## 收尾报告

完成后报告三件事：
- 清理了多少项、释放多少空间
- 远端备份位置（`D:/code-backup/cleanup/<日期>/<项目>/` 已 push 到 GitHub）
- 若要恢复，`git clone git@github.com:fariytale-undertale/claude-backup.git` 即可
