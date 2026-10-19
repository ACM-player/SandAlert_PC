#!/usr/bin/env python3
"""本机 Modbus TCP 仪表模拟器，适配 pymodbus 3.13.0。

从 SandAlert_PC_Replica 运行 python mock_modbus_server.py。
UI 选择 Modbus RTU Mode，勾选 Use TCP (Local Test)，连接 127.0.0.1:5020。
"""

import argparse
import asyncio
from dataclasses import dataclass, field
import random

from pymodbus.server import ModbusTcpServer
from pymodbus.simulator import DataType, SimData, SimDevice


DEVICE_ID = 126
UPDATE_INTERVAL = 0.5


@dataclass
class MockInstrument:
    """砂率在两次更新间保持不变，累计量按实际经过的秒数积分。"""

    rng: random.Random = field(default_factory=random.Random)
    total_mass: float = 0.0
    mass_rate_raw: int = 0

    def advance(self, elapsed_seconds: float) -> list[int]:
        if elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        self.total_mass += self.mass_rate_raw / 100.0 * elapsed_seconds
        ips = self.rng.randint(20, 200)
        self.mass_rate_raw = self.rng.randint(100, 5000)
        mass_rate = self.mass_rate_raw / 100.0
        relay = 0
        if ips > 100 or mass_rate > 20.0:
            relay |= 0x01
        if ips > 180 or mass_rate > 40.0:
            relay |= 0x02
        current_raw = round((4.0 + mass_rate / 50.0 * 16.0) * 100)
        # 仪表累计寄存器以整克存储；保留内部小数，避免每次更新截断。
        counter = int(self.total_mass) & 0xFFFFFFFF
        return [
            ips, 0, self.mass_rate_raw, 0,
            counter >> 16, counter & 0xFFFF,
            0, 0, relay, current_raw, ips + 20,
        ]


def create_server(port: int = 5020) -> ModbusTcpServer:
    device = SimDevice(
        id=DEVICE_ID,
        simdata=(
            [SimData(address=0, values=False, datatype=DataType.BITS)],
            [SimData(address=0, values=False, datatype=DataType.BITS)],
            [SimData(address=0, values=0, datatype=DataType.UINT16)],
            [SimData(address=0, count=20, values=0, datatype=DataType.UINT16)],
        ),
    )
    return ModbusTcpServer(device, address=("127.0.0.1", port))


async def _background_updater(server: ModbusTcpServer) -> None:
    instrument = MockInstrument()
    loop = asyncio.get_running_loop()
    previous_update = loop.time()
    while True:
        now = loop.time()
        registers = instrument.advance(now - previous_update)
        previous_update = now
        await server.async_setValues(DEVICE_ID, 4, 0, registers)
        print(
            f"[Mock] IPS={registers[0]:3d} | "
            f"MassRate={registers[2] / 100.0:5.2f} g/s | "
            f"TotalMass={int(instrument.total_mass)} g | Relay=0x{registers[8]:02X}"
        )
        await asyncio.sleep(UPDATE_INTERVAL)


async def main(port: int = 5020) -> None:
    server = create_server(port)
    try:
        await server.serve_forever(background=True)
        print(f"SandAlert Mock Modbus TCP: 127.0.0.1:{port} | Device ID: {DEVICE_ID}")
        await _background_updater(server)
    finally:
        await server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5020)
    args = parser.parse_args()
    try:
        asyncio.run(main(args.port))
    except KeyboardInterrupt:
        print("\n[停止] 模拟仪表已关闭")
