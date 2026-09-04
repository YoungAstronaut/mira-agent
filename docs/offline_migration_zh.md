# MIRA Agent 移动硬盘离线迁移指南

[返回中文 README](../README_zh.md) | [English README](../README.md)

本文说明如何通过移动硬盘，把 MIRA Agent 的 Docker 运行环境、Git 仓库、Slime submodule、模型和数据迁移到正式训练服务器。目标是在正式机器不重新构建 Python/CUDA 用户态环境，并且在断网条件下也能恢复完整工程。

## 迁移边界

移动硬盘可以携带：

- `mira-agent/slime-runtime:20260903-cu129`：Slime、PyTorch、SGLang、Ray 和 Megatron 运行环境；
- `python:3.11-slim`：RL 在线 Python 工具沙箱；
- MIRA Agent Git 历史和当前 commit；
- Slime submodule 的 Git 历史和固定 commit；
- Hugging Face 模型、可选的 Megatron `torch_dist` checkpoint，以及原始数据。

移动硬盘不能替代正式机器上的：

- NVIDIA 驱动；
- Docker Engine；
- NVIDIA Container Toolkit；
- 足够的本地 Docker 数据空间；
- 可供训练独占使用的 GPU。

不要直接复制 `/var/lib/docker`，也不要使用 `docker export` 导出镜像。跨机器迁移镜像应使用 `docker save` 和 `docker load`。

## 当前固定版本

| 组件 | 固定标识 |
|---|---|
| MIRA Agent 训练镜像 | `mira-agent/slime-runtime:20260903-cu129` |
| 训练镜像 ID | `sha256:39be6cbb00f9b6770e664ace0c7b9f5ecff2977a1a205e7926a720f906fbc62c` |
| Python 沙箱镜像 | `python:3.11-slim` |
| Python 沙箱镜像 ID | `sha256:be1575ed968de893bd54f4c56315ff7c4736ce522c1bca08fd521731aafc0d76` |
| Slime commit | `4c193f1f37509cca70f0e88807a9305b70f63f4e` |
| Qwen3-8B-Base revision | `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` |
| Qwen3-8B revision | `b968826d9c46dd6066d109eabc6255188de91218` |

镜像在 Docker 中展开后约占 67 GB，两套 Hugging Face 模型约占 31 GB，原始数据约占 291 MB。若同时携带两套 `torch_dist` checkpoint，建议移动硬盘至少预留 150 GB；为日志、校验文件和后续 checkpoint 留出余量时，建议准备 200 GB 以上空间。

移动硬盘推荐使用 ext4。FAT32 存在单文件 4 GB 限制，不能用于镜像归档；exFAT 可以存放大文件，但不会保留完整的 Unix 权限语义。

## 一、在源机器制作离线包

### 1. 设置路径并检查状态

从 MIRA Agent 仓库内执行：

```bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
MODEL_ROOT="${MODEL_ROOT:-/data/dhsun/mira-agent/models}"
TRANSFER_ROOT=/path/to/mounted-disk/mira-agent-offline

mkdir -p \
  "${TRANSFER_ROOT}/images" \
  "${TRANSFER_ROOT}/git" \
  "${TRANSFER_ROOT}/artifacts/models" \
  "${TRANSFER_ROOT}/artifacts/data/raw" \
  "${TRANSFER_ROOT}/manifest"

git -C "${REPO_ROOT}" status --short
git -C "${REPO_ROOT}" submodule status
df -h "${TRANSFER_ROOT}"
```

打包前，主仓库和 Slime submodule 的工作树都应为空。Git bundle 只包含已经提交的内容，不包含未提交文件。

### 2. 导出两套 Docker 镜像

推荐使用 Zstandard 压缩。源机器和目标机器都需要 `zstd`：

```bash
set -euo pipefail

docker image inspect mira-agent/slime-runtime:20260903-cu129 >/dev/null
docker image inspect python:3.11-slim >/dev/null

docker save \
  mira-agent/slime-runtime:20260903-cu129 \
  python:3.11-slim \
  | zstd -T0 -3 -o "${TRANSFER_ROOT}/images/docker-images.tar.zst"
```

如果目标机器没有 `zstd`，可以改为不压缩的 tar，但移动硬盘需要更多空间：

```bash
docker save \
  mira-agent/slime-runtime:20260903-cu129 \
  python:3.11-slim \
  -o "${TRANSFER_ROOT}/images/docker-images.tar"
```

只需保留 `docker-images.tar.zst` 或 `docker-images.tar` 中的一种。

### 3. 打包主仓库和 Slime submodule

```bash
set -euo pipefail

# Git bundle 必须包含完整的提交历史。若仓库是浅克隆，先在源机器联网补全对象。
for repository in \
  "${REPO_ROOT}" \
  "${REPO_ROOT}/third_party/slime"
do
  if test "$(git -C "${repository}" rev-parse --is-shallow-repository)" = true; then
    git -C "${repository}" fetch --unshallow --tags origin
  fi
done

test -z "$(git -C "${REPO_ROOT}" status --porcelain)"
test -z "$(git -C "${REPO_ROOT}/third_party/slime" status --porcelain)"

git -C "${REPO_ROOT}" bundle create \
  "${TRANSFER_ROOT}/git/mira-agent.bundle" --all

git -C "${REPO_ROOT}/third_party/slime" bundle create \
  "${TRANSFER_ROOT}/git/slime.bundle" --all

git -C "${REPO_ROOT}" rev-parse HEAD \
  > "${TRANSFER_ROOT}/manifest/mira-agent.commit"
git -C "${REPO_ROOT}/third_party/slime" rev-parse HEAD \
  > "${TRANSFER_ROOT}/manifest/slime.commit"

git bundle verify "${TRANSFER_ROOT}/git/mira-agent.bundle"
git bundle verify "${TRANSFER_ROOT}/git/slime.bundle"
```

即使正式机器可以访问 GitHub，也建议保留这两个 bundle，避免迁移时受网络或 submodule 下载影响。

### 4. 复制模型和数据

```bash
set -euo pipefail

rsync -aH --info=progress2 \
  "${MODEL_ROOT}/" \
  "${TRANSFER_ROOT}/artifacts/models/"

rsync -aH --info=progress2 \
  "${REPO_ROOT}/data/raw/" \
  "${TRANSFER_ROOT}/artifacts/data/raw/"
```

检查是否已经包含 Megatron checkpoint：

```bash
for checkpoint in \
  Qwen3-8B-Base_torch_dist \
  Qwen3-8B_torch_dist
do
  if test -d "${TRANSFER_ROOT}/artifacts/models/${checkpoint}"; then
    printf 'OK %s\n' "${checkpoint}"
  else
    printf 'MISSING %s; convert on the target machine\n' "${checkpoint}"
  fi
done
```

如果这两个目录缺失，离线包仍可使用，但必须在正式机器上用一张 GPU 完成转换后才能启动 SFT/RL。

### 5. 生成完整性校验文件

该步骤会顺序读取全部镜像和模型文件，耗时取决于移动硬盘速度：

```bash
set -euo pipefail

(
  cd "${TRANSFER_ROOT}"
  find . -type f \
    ! -path './manifest/SHA256SUMS' \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum
) > "${TRANSFER_ROOT}/manifest/SHA256SUMS"

sync
```

写入结束后再安全卸载移动硬盘，避免操作系统尚未刷盘。

离线包应具有以下结构：

```text
mira-agent-offline/
├── images/
│   └── docker-images.tar.zst
├── git/
│   ├── mira-agent.bundle
│   └── slime.bundle
├── artifacts/
│   ├── models/
│   └── data/raw/
└── manifest/
    ├── SHA256SUMS
    ├── mira-agent.commit
    └── slime.commit
```

## 二、在正式机器恢复

### 1. 检查宿主机基础能力

```bash
nvidia-smi
docker version
docker info
df -h /var/lib/docker /data 2>/dev/null || true
```

正式机器的 NVIDIA 驱动必须支持镜像所用的 CUDA 12.9。Docker 必须已经配置 NVIDIA Container Toolkit。Docker 镜像会被加载到目标 daemon 的数据目录，通常是 `/var/lib/docker`；它们不会直接从移动硬盘运行。

### 2. 校验移动硬盘内容

```bash
set -euo pipefail

TRANSFER_ROOT=/path/to/mounted-disk/mira-agent-offline

(
  cd "${TRANSFER_ROOT}"
  sha256sum -c manifest/SHA256SUMS
)
```

任何一项校验失败都应停止恢复，重新复制对应文件。

### 3. 加载 Docker 镜像

Zstandard 归档：

```bash
set -euo pipefail

zstd -dc "${TRANSFER_ROOT}/images/docker-images.tar.zst" \
  | docker load
```

未压缩归档：

```bash
docker load -i "${TRANSFER_ROOT}/images/docker-images.tar"
```

确认镜像 ID：

```bash
test "$(docker image inspect \
  mira-agent/slime-runtime:20260903-cu129 \
  --format '{{.Id}}')" = \
  'sha256:39be6cbb00f9b6770e664ace0c7b9f5ecff2977a1a205e7926a720f906fbc62c'

test "$(docker image inspect \
  python:3.11-slim \
  --format '{{.Id}}')" = \
  'sha256:be1575ed968de893bd54f4c56315ff7c4736ce522c1bca08fd521731aafc0d76'
```

### 4. 恢复 Git 仓库和 submodule

```bash
set -euo pipefail

WORK_ROOT=/path/to/workspace
REPO_ROOT="${WORK_ROOT}/mira-agent"

mkdir -p "${WORK_ROOT}"
git clone "${TRANSFER_ROOT}/git/mira-agent.bundle" "${REPO_ROOT}"

# 临时让 submodule 从移动硬盘上的 bundle 初始化，完成后恢复项目记录的远端 URL。
git -C "${REPO_ROOT}" submodule init
git -C "${REPO_ROOT}" config submodule.third_party/slime.url \
  "${TRANSFER_ROOT}/git/slime.bundle"
git -C "${REPO_ROOT}" -c protocol.file.allow=always \
  submodule update --init --checkout
git -C "${REPO_ROOT}" submodule sync

test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = \
  "$(cat "${TRANSFER_ROOT}/manifest/mira-agent.commit")"
test "$(git -C "${REPO_ROOT}/third_party/slime" rev-parse HEAD)" = \
  "$(cat "${TRANSFER_ROOT}/manifest/slime.commit")"

git -C "${REPO_ROOT}" submodule status
```

如果正式机器可以连接 GitHub，可以把主仓库 origin 改回团队远端；这不是离线运行所必需的：

```bash
git -C "${REPO_ROOT}" remote set-url origin \
  git@github.com:YoungAstronaut/mira-agent.git
```

### 5. 恢复模型和数据

选择正式机器上的共享存储目录：

```bash
set -euo pipefail

TARGET_SHARED_ROOT=/path/to/shared-storage/mira-agent
MODEL_ROOT="${TARGET_SHARED_ROOT}/models"

mkdir -p "${MODEL_ROOT}" "${REPO_ROOT}/data/raw"

rsync -aH --info=progress2 \
  "${TRANSFER_ROOT}/artifacts/models/" \
  "${MODEL_ROOT}/"

rsync -aH --info=progress2 \
  "${TRANSFER_ROOT}/artifacts/data/raw/" \
  "${REPO_ROOT}/data/raw/"

export MIRA_SHARED_ROOT="${TARGET_SHARED_ROOT}"
export MODEL_ROOT
```

如果目标存储路径与源机器不同，只需在启动容器和训练脚本时继续传入 `MIRA_SHARED_ROOT` 与 `MODEL_ROOT`。

### 6. 验证 GPU 容器环境

```bash
docker run --rm \
  --gpus all \
  --entrypoint bash \
  mira-agent/slime-runtime:20260903-cu129 \
  -lc 'nvidia-smi && python - <<\"PY\"
import torch
import ray
import sglang
import transformers
import megatron.core

print(\"torch\", torch.__version__)
print(\"cuda\", torch.cuda.is_available(), torch.cuda.device_count())
print(\"ray\", ray.__version__)
print(\"sglang\", sglang.__version__)
print(\"transformers\", transformers.__version__)
print(\"megatron\", megatron.core.__version__)
PY'
```

如果 `torch.cuda.is_available()` 不是 `True`，不要继续转换或训练，应先修复 NVIDIA Driver、Docker 或 NVIDIA Container Toolkit。

### 7. 验证代码和 Python 工具沙箱

```bash
docker run --rm \
  -v "${REPO_ROOT}:/workspace/mira-agent" \
  -v /usr/bin/docker:/usr/local/bin/docker:ro \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -w /workspace/mira-agent \
  -e MATH_AGENT_RUN_DOCKER_TESTS=1 \
  --entrypoint bash \
  mira-agent/slime-runtime:20260903-cu129 \
  -lc 'PYTHONPATH=/workspace/mira-agent:/workspace/mira-agent/third_party/slime:/root/Megatron-LM \
       pytest -q'
```

该检查应通过普通单元测试和真实的无网络 Python 沙箱测试。挂载 Docker socket 等价于向容器授予较高的宿主机权限，只应在可信的训练容器中使用。

### 8. 准备 Megatron checkpoint

确认以下目录存在：

```bash
test -d "${MODEL_ROOT}/Qwen3-8B-Base"
test -d "${MODEL_ROOT}/Qwen3-8B"
test -d "${MODEL_ROOT}/Qwen3-8B-Base_torch_dist"
test -d "${MODEL_ROOT}/Qwen3-8B_torch_dist"
```

如果后两个目录缺失，使用[中文 README 的模型准备命令](../README_zh.md#模型准备)在正式机器上依次转换。转换脚本至少需要一张可用 GPU；不要在其他训练占用 GPU 时强制执行。

## 三、训练前最终验收

运行训练前应满足：

- `git status --short` 没有非预期修改；
- `git submodule status` 指向固定的 Slime commit；
- 两套 Docker 镜像 ID 与本文一致；
- 两套 HF 模型和两套 `torch_dist` checkpoint 均存在；
- ReTool 与 DAPO-Math 原始数据存在；
- 8 张 GPU 可由本次任务独占；
- 当前容器内没有需要复用或停止的 Ray 集群；
- CPU/容器测试与 Python 沙箱测试通过。

先运行一次 SFT smoke update，确认 loss、loss mask 和 checkpoint 保存正常；再运行一次 RL smoke update，确认 rollout、Python 工具调用、reward 和 learner update 正常。具体命令见[中文 README 的冒烟测试章节](../README_zh.md#冒烟测试)。

当前脚本会在 GPU 已被占用或 checkpoint 缺失时拒绝启动。不要通过 `ALLOW_BUSY_GPUS=1` 绕过共享服务器上的保护检查。

## 常见故障

| 现象 | 优先检查 |
|---|---|
| `could not select device driver` 或 `--gpus` 不可用 | NVIDIA Container Toolkit 是否安装并重启 Docker |
| 转换报 `No usable accelerator was detected` | `docker run` 是否包含 `--gpus device=0` |
| 容器内找不到模型 | `MIRA_SHARED_ROOT` 是否挂载，`MODEL_ROOT` 是否指向容器内可见路径 |
| RL 找不到 `docker` 或无法连接 daemon | 是否挂载 Docker CLI 和 `/var/run/docker.sock` |
| `git submodule status` 前出现 `-` | Slime bundle 是否已经克隆到 `third_party/slime` |
| `docker load` 空间不足 | 检查 Docker data-root 所在文件系统，而不只是移动硬盘空间 |
| GPU 可见但 CUDA 初始化 OOM | GPU 是否已有计算进程；应申请独占资源后重试 |

完成上述验收后，移动硬盘只用于归档和灾备即可；正式训练不应依赖移动硬盘的持续挂载。
