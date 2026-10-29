"""load_config checks the system file against the awesIO system_schema."""

import warnings
from pathlib import Path

import pytest
import yaml

from awes_ekf.setup import settings
from awes_ekf.setup.settings import load_config, validate_system_config

KITE_DIR = Path(__file__).resolve().parents[2] / "data" / "LEI-V3-KITE"


def test_a_schema_violation_warns_and_does_not_stop():
    with pytest.warns(UserWarning, match="Validation failed"):
        validate_system_config({"metadata": {"schema": "system_schema.yml"}})


@pytest.mark.parametrize("flight", ["2017", "2019", "2025"])
def test_flown_system_files_conform(flight):
    system = yaml.safe_load(
        (KITE_DIR / f"system_flown_{flight}.yaml").read_text(encoding="utf-8")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_system_config(system)


@pytest.mark.skipif(not (KITE_DIR / "ekf_config.yaml").exists(), reason="LEI-V3 data missing")
def test_load_config_validates_the_chosen_system_file(monkeypatch):
    seen = []
    monkeypatch.setattr(settings, "validate_system_config", seen.append)
    load_config(KITE_DIR, system_file=KITE_DIR / "system_flown_2025.yaml")
    assert len(seen) == 1 and "components" in seen[0]
