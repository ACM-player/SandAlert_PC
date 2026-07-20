# SandAlert Phase 1 Runbook

The Phase 1 topology is:

```text
PyQt UI <- ws://127.0.0.1:8766 -> Mac Edge
                                      |
                                      +-> WebSocket binary waveform / JSON Modbus
                                          -> FastAPI Compute Server
                                          -> SQLite metadata + HDF5 waveform archive
```

## Mac setup

Run from `SandAlert_PC_Replica`:

```bash
conda run -n playground python -m pip install -r requirements/dev.txt
cp config/edge.example.toml config/edge.toml
python -m edge.generate_sample runtime/sample_data/tdoa_60deg.txt --angle 60
python -m edge.main --config config/edge.toml
python main.py
```

Select `Setup -> Comm Settings -> Edge Service Mode` in the UI.

## Lab server setup

All Python commands must use the dedicated interpreter:

```bash
cd /home/lks/project/SandAlert_PC/SandAlert_PC_Replica
/home/lks/miniconda3/envs/core/bin/python -m pip install -r requirements/server.txt
cp config/server.example.toml config/server.toml
```

Set the server bind address, shared token, SQLite path and HDF5 directory in the
untracked `config/server.toml`. Start it inside tmux:

```bash
tmux new -s sandalert-server
/home/lks/miniconda3/envs/core/bin/python -m server.main --config config/server.toml
```

The server checkout is deployment-only. Make source changes on the Mac, push a
task branch to GitHub, then fetch and check out the exact branch/commit on the
server.
