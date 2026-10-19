import asyncio
import random

import pytest

pytest.importorskip("PySide6", reason="Local Qt/Modbus test requires the edge environment")
pytest.importorskip("pymodbus", reason="Local Modbus test requires edge dependencies")
from PySide6.QtCore import Qt

from core.comm import ModbusReaderThread
from mock_modbus_server import DEVICE_ID, MockInstrument, create_server


def test_simulated_mass_integrates_previous_rate_and_preserves_fraction():
    instrument = MockInstrument(rng=random.Random(42), total_mass=65535.75)
    first = instrument.advance(0.0)
    second = instrument.advance(0.5)
    expected_mass = 65535.75 + first[2] / 100.0 * 0.5
    assert instrument.total_mass == pytest.approx(expected_mass)
    assert (second[4] << 16) + second[5] == int(expected_mass)
    third = instrument.advance(0.125)
    expected_mass += second[2] / 100.0 * 0.125
    assert instrument.total_mass == pytest.approx(expected_mass)
    assert (third[4] << 16) + third[5] == int(expected_mass)


def test_modbus_reader_decodes_real_tcp_registers():
    """Exercise the actual Qt reader against the simulator's TCP server."""
    async def exercise():
        server = create_server(port=0)
        reader = None
        readings = []
        messages = []
        try:
            await server.serve_forever(background=True)
            port = server.transport.sockets[0].getsockname()[1]
            registers = [123, 0, 2345, 0, 1, 17, 0, 0, 3, 1234, 143]
            await server.async_setValues(DEVICE_ID, 4, 0, registers)
            reader = ModbusReaderThread("127.0.0.1", use_tcp=True, tcp_port=port)
            reader.POLL_INTERVAL = 0.01

            def receive(data):
                readings.append(data)
                if len(readings) == 2:
                    reader._is_running = False

            reader.data_received.connect(receive, Qt.DirectConnection)
            reader.status_msg.connect(messages.append, Qt.DirectConnection)
            reader.start()

            async def wait_for_reader():
                while reader.isRunning():
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(wait_for_reader(), timeout=4.0)
            assert len(readings) == 2, messages
            for reading in readings:
                assert reading["raw_registers"] == registers
                assert reading["ips"] == 123
                assert reading["mass_rate"] == 23.45
                assert reading["total_mass"] == 65553.0
                assert reading["relay_status"] == 3
                assert reading["current_ma"] == pytest.approx(12.34)
        finally:
            if reader is not None:
                reader._is_running = False
                assert await asyncio.to_thread(reader.wait, 4000)
            await server.shutdown()

    asyncio.run(exercise())
