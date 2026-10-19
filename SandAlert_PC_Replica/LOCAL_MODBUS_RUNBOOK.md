# 本地 Modbus 演示与验证

更新：2026-10-19。本流程使用 UI 直接连接本机模拟仪表，可以独立于计算服务器运行。
本流程仅需要两个本地进程：Mock Modbus TCP 和 PyQt UI。

## 启动

两个终端均使用同一个工作目录与环境：

```bash
cd /Users/lk/XJTU_Research/Code_Playground/SandAlert/SandAlert_PC_Replica
conda activate playground
```

终端一启动模拟仪表：

```bash
python -u mock_modbus_server.py
```

看到 `SandAlert Mock Modbus TCP: 127.0.0.1:5020 | Device ID: 126` 表示监听成功。
终端会约每 0.5 秒输出 IPS、砂率、累计量和报警位。
需要其他本地端口时可指定 `--port 5021`，并在 UI 填相同端口。

终端二启动界面：

```bash
python main.py
```

1. 打开 `Setup -> Comm Settings...`。
2. 选择 `Modbus RTU Mode (Real Hardware)`。
3. 勾选 `Use TCP (Local Test)`。
4. 填写 Host `127.0.0.1`、Port `5020`，点击 `Save & Connect`。

IPS、Mass Rate 和报警随寄存器更新，标签显示 `[Modbus]`。
本接口只提供仪表指标；波形和 TDOA 需要波形数据源。
关闭 UI 后，在模拟器终端按 `Ctrl+C` 结束模拟仪表。

## CSV 累计量规则

点击 `Start CSV Log` 后，`CumulativeMass(g)` 表示本次录制期间的仪表计数增量：

- 开始录制时使用最近一次仪表累计读数作为基线；此前没有读数则用首条录制读数建立基线，该行累计为 0。
- 重复读数不重复计量；后续增量直接来自累计寄存器，不依赖假定的轮询时间。
- 仪表计数下降或重新选择连接时重新建立基线，保留本次已记录的累计值。
- 复位、回绕或连接更换期间无法判定的砂量不补算。
- 仪表累计寄存器以整克存储，因此 CSV 的 Modbus 累计量也具有 1 g 分辨率。

模拟器按前一个砂率与实际经过时间积分累计量，内部保留小数，再编码为两个 16 位寄存器。
砂率、累计量、电流和报警属于模拟数据，用于检查通信及显示。

## 依赖与验证

本地采集依赖固定为 `pymodbus==3.13.0`，客户端使用 `device_id=126`。
当前 `playground` 已安装该版本，可用以下命令核对：

```bash
python -c 'import pymodbus; print(pymodbus.__version__)'
```

执行测试：

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

`test_modbus_local.py` 会临时监听 `127.0.0.1` 的系统分配端口，检查实际 Qt 读取线程与 TCP 模拟器；
`test_modbus_csv_ui.py` 检查 CSV 增量、重复读数、计数复位和重新连接。

## 阶段修复与迁移

DSP、Edge UI、TDOA 显示及标定范围的修复见 [阶段说明](STAGE1_BUGFIXES.md)。
当前运行进度和测试证据记录在仓库根目录 `taskline.md`。
