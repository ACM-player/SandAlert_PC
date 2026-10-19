# 第一阶段修复与新设备迁移

更新：2026-10-19。分支：`codex/stage1-bugfixes`。

## 修复内容

| 问题 | 当前处理 |
|---|---|
| Modbus 客户端接口不兼容 | 固定 pymodbus 3.13.0，使用 device_id=126，真实本机 TCP 回归测试 |
| Modbus CSV 时间基准错误 | 使用本次录制期间的仪表累计寄存器增量 |
| 模拟累计量与砂率无关 | 用前一砂率和单调时钟实际间隔积分，内部保留小数 |
| DSP 固定 1000 Hz | 服务器跟随 fs_hz，本机旧波形模式提供显式采样率设置 |
| 窗口值冒充每秒速率 | 以 N/fs 归一化 IPS 与质量流率，累计量按新增样点时长积分 |
| 重叠窗口标定重复累积 | 按新增时长与窗口时长之比累加 E，和 CSV 质量积分一致 |
| 双模拟源轮流覆盖 UI | 增加一致的指标来源选择；自动模式优先波形 |
| TDOA 每事件显示两次 | UI 根据会话和序号去重，保留现有两种结果消息 |
| Edge/Modbus 标定按钮无效 | 禁用不适用的本机标定和本地计算参数，提示实际参数来源 |
| 退出时 SQLite 已关闭错误 | 取消并等待收发任务及磁盘工作结束后关闭 spool |

速率与标定的计算语义已改变，旧 K 必须重新核定。g/s 依赖实测 K；默认参数和合成波形不能证明现场精度。
详细公式见 [技术说明](SandAlert_Technical_Document.md)。

## 新设备拉取

首次获取代码（默认主分支未合并此阶段修改）：

```bash
git clone --branch codex/stage1-bugfixes https://github.com/ACM-player/SandAlert_PC.git
cd SandAlert_PC
```

已有仓库：先保留该设备自己的改动，再执行：

```bash
git fetch origin
git switch codex/stage1-bugfixes
git pull --ff-only origin codex/stage1-bugfixes
```

新 Mac 开发环境建议 Python 3.10，环境名可沿用 playground：

```bash
conda create -n playground python=3.10
conda activate playground
cd SandAlert_PC_Replica
python -m pip install -r requirements/dev.txt
QT_QPA_PLATFORM=offscreen python -m pytest -q
cd ..
python scripts/smoke_pipeline.py
```

冒烟测试使用临时数据、端口和缓存，完成后退出。现有实际配置、密钥、数据库、HDF5、原始现场数据、历史待发缓存均不在 Git 中，不会随 pull 自动迁移。
新设备若连接实际服务器，应从配置样例建立自己的配置，并通过已有安全方式获取真实地址/令牌。

## 手工查看 UI

- 本机指标演示按 [Modbus 手册](LOCAL_MODBUS_RUNBOOK.md)，UI 选择 Modbus RTU Mode 并启用 TCP。
- 本机波形回放或虚拟串口：在 Parameters → System 确认采样率。历史 1 kHz 数据填 `1000`，2 MHz 阵列模拟波形填 `2000000`。
- Edge：先运行已配置的计算服务器与 Edge，再连接本机网关，按需选择“指标来源”。标定页会说明服务器参数配置的生效范围。
- 原有 Edge spool 可能含历史待发送记录，演示请使用新目录；不要为测试而删除历史缓存。

## 服务器测试方式

服务器只从 GitHub 拉取明确分支/提交，使用 `/home/lks/miniconda3/envs/core/bin/python`。
本阶段在独立工作树运行测试及临时计算服务，实际运行目录、配置和数据不自动部署覆盖。
Linux 仅装 server 依赖时，Qt/本机 Modbus 专用测试会明确跳过；Mac 必须通过全部测试。

本机全链路检查：

```bash
python scripts/smoke_pipeline.py
```

SSH 隧道测试一个独立远程测试实例时，由该实例的启动者提供临时令牌，且仅将远程回环端口映射到本机：

```bash
# 终端一：端口以实际独立测试实例为准
ssh -N -L 18765:127.0.0.1:18765 lab

# 终端二：使用独立测试实例的临时令牌
export SANDALERT_SMOKE_TOKEN='临时测试令牌'
python scripts/smoke_pipeline.py --server-url ws://127.0.0.1:18765/v1/ingest
unset SANDALERT_SMOKE_TOKEN
```

检查应得到 12 个 analysis.result、Modbus 结果、接近 60° 的可靠方向、12 行 CSV、零待发消息，以及砂率积分与累计质量一致。
最终测试结果、服务器验证目录和提交信息记录在仓库根目录 [taskline.md](../taskline.md)。

## 本阶段已验证结果

2026-10-19 代码提交 `534683f`：Mac playground 与 lab core 环境均为 **40 passed**，各有一条既有 Starlette/httpx 弃用警告。
本机全链路和 Mac 经 SSH 隧道访问 lab 的全链路均得到 12 条分析、2 条 Modbus、61.5° 可靠方向、12 行 CSV、零待发消息，质量积分一致。
临时测试服务已关闭，原服务器运行目录未自动切换；具体位置及后续操作见 `taskline.md`。
