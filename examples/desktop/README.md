# Desktop Application

Python backend entry point for the AgentScope desktop application.

## Usage

### Standalone

```bash
python main.py --port 8000
```

### With Electron

```bash
cd ../web_ui
PYTHON_CMD="conda run -n QwenPaw python" pnpm electron:dev
```

See [docs/electron_desktop_guide.md](../../docs/electron_desktop_guide.md) for details.

## Prerequisites

- Redis running on `localhost:6379`
- `pip install "agentscope[service,storage]"`
