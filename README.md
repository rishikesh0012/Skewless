# ML Skew

Training-serving feature parity and model monitoring.

## Development setup

Create and activate the virtual environment, then install the project:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -e ".[dev]"

## Quality checks

    pytest
    ruff check .
    mypy src

## Supported skew modes

- none
- distance_unit
- timezone
- missing_value
- rush_hour_rule

The fault injector is used only for testing and demonstration.
