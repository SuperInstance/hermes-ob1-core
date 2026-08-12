# Agent Shells

Agent shells are operational modules for Hermes OB1's perception capabilities. Each shell wraps a specific domain of expertise.

## Structure

```
shells/
├── agents/
│   ├── shell-ts-architect/     — TypeScript architecture (status: initializing)
│   │   └── state.json          — Pending: fix_syntax, resolve_exports, fix_bridge_types
│   ├── shell-math-specialist/  — Predictive sonar engine
│   │   ├── engine.py           — PredictiveSonarEngine (intensity derivatives, trajectory)
│   │   ├── simulation.py       — Simulation runner for the engine
│   │   └── tests/
│   │       └── test_engine.py  — 15 tests covering engine behavior
│   ├── shell-signal-specialist/ — Temporal signature analysis
│   │   ├── temporal_analyzer.py — TemporalSignatureAnalyzer (pre-strike detection)
│   │   └── tests/
│   │       └── test_temporal_analyzer.py — 13 tests covering escalation detection
│   └── shell-bard/             — Creative output
│       └── riff.md             — "Echoes of the Iron Sea" (sea opera fragment)
```

## Status

| Shell | Status | Tests |
|-------|--------|-------|
| ts-architect | ⚠️ initializing (3 pending tasks) | 0 |
| math-specialist | ✅ functional | 15 |
| signal-specialist | ✅ functional | 13 |
| bard | ✅ creative output exists | 0 (creative) |

## Running Tests

```bash
# Math specialist
cd shells/agents/shell-math-specialist
pip install numpy pytest
python -m pytest tests/ -v

# Signal specialist
cd shells/agents/shell-signal-specialist
pip install numpy pytest
python -m pytest tests/ -v
```
