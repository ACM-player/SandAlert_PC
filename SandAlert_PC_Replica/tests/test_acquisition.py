import asyncio
from uuid import uuid4

from edge.acquisition import FileReplaySource


def test_four_channel_file_replay(tmp_path):
    replay_file = tmp_path / "four_channel.txt"
    replay_file.write_text("1,2,3,4\n5,6,7,8\n", encoding="utf-8")

    async def collect():
        source = FileReplaySource(
            tmp_path, uuid4(), fs_hz=2000, chunk_samples=2, speed=1000
        )
        return [item async for item in source.frames()]

    frames = asyncio.run(collect())
    assert len(frames) == 1
    metadata, data = frames[0]
    assert metadata.channels == 4
    assert data.shape == (4, 2)
