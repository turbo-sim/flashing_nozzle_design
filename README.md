# Flashing Nozzle Design

Sizing and analysis tools for flashing (two-phase) nozzles and turbine expanders in partial-evaporation Rankine cycle (ORC) power systems.

The package covers the calculations needed to size a preliminary ORC system around a flashing nozzle or turbine: thermodynamic cycle evaluation, choked (sonic) flow conditions, droplet size and slip velocity estimation at the nozzle throat, and rig sizing across candidate working fluids.

## Project layout

```
flashing_nozzle_design/
├── src/
│   └── flashing_nozzle_design/   # installable package (import as `import flashing_nozzle_design`)
│       ├── __init__.py
│       ├── functions.py          # cycle, choking, droplet-size and rig sizing calculations
│       └── graphics.py           # matplotlib styling and plotting helpers
├── examples/                     # example scripts and input configuration files
└── pyproject.toml                # Poetry project and dependency configuration
```

## Installation

### 1. Install Poetry

This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging. If it is not already installed, install it with the official installer:

**Linux / macOS:**

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Windows (PowerShell):**

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

After installation, restart your terminal and verify it worked:

```bash
poetry --version
```

See the [official installation guide](https://python-poetry.org/docs/#installation) if you encounter problems.

### 2. Clone the repository

```bash
git clone https://github.com/turbo-sim/flashing_nozzle_design.git
cd flashing_nozzle_design
```

### 3. Install the project and its dependencies

```bash
poetry install
```

This creates an isolated virtual environment, installs all runtime and development dependencies declared in `pyproject.toml`, and installs the project itself in editable mode so that `import flashing_nozzle_design` works everywhere.

## Usage

Run any script through Poetry so it uses the project's virtual environment:

```bash
poetry run python examples/run_flashing_nozzle_calculation.py
```

[Add something about the expected output]