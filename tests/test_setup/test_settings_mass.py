"""The EKF point mass is the wing plus the bridle (load_config)."""

from pathlib import Path

import pytest
import yaml

from awes_ekf.setup.settings import get_kite, load_config

KITE_DIR = Path(__file__).resolve().parents[2] / "data" / "LEI-V3-KITE"


@pytest.mark.skipif(not (KITE_DIR / "ekf_config.yaml").exists(), reason="LEI-V3 data missing")
def test_kite_mass_is_wing_plus_bridle():
    system_file = KITE_DIR / "system_flown_2019.yaml"
    config = load_config(KITE_DIR, system_file=system_file)
    kite = get_kite(yaml.safe_load(system_file.read_text(encoding="utf-8")))
    assert config["kite"]["mass"] == pytest.approx(
        kite["wing"]["structure"]["mass"] + kite["bridle"]["structure"]["mass"]
    )
