"""KCU (+ onboard turbine) drag in the tether model.

The KCU is a finite cylinder along the bridle segment (KCU -> kite). With the
apparent wind at the KCU split about that axis ``e`` into an axial part
``v_ax = (va . e) e`` and a crossflow part ``v_n = va - v_ax``::

    D = 0.5 rho ( cdt At |v_ax| v_ax + (cdp Ap + C_T A_rotor) |v_n| v_n )

``cdt`` is the handbook axial-flow coefficient on the frontal area
``At = pi d^2 / 4`` and ``cdp`` the crossflow one on the broadside area
``Ap = d L``. The onboard turbine sits on top of the KCU facing the apparent
wind, so its momentum thrust joins the crossflow term only.

Until 2026-10 the tether model paired them the other way round (``cdp Ap`` on
the axial flow, ``cdt At`` on the crossflow), which under-predicted the
crosswind KCU drag by ~1.5x; these tests evaluate the model's own
``cd_kcu`` CasADi function against the closed form.
"""

from __future__ import annotations

import numpy as np
import pytest

import awes_ekf.setup.tether as tether_mod
from awes_ekf.setup.kcu import KCU
from awes_ekf.setup.settings import ObservationData, _kcu_mass, _turbine_config
from awes_ekf.setup.tether import Tether

LEI_V3_KCU = dict(
    length=1.0,
    diameter=0.48,
    mass=22.0,
    distance_kcu_kite=10.0,
    total_length_bridle_lines=0.0,  # no bridle drag: isolate the KCU term
    diameter_bridle_lines=0.0,
)


class _KiteStub:
    """Minimal kite the tether reads (``mass`` and ``area`` only)."""

    mass = 14.2
    area = 19.75


def _cd_kcu(kcu, elevation_0, *, wind=(10.0, 0.0, 0.0)):
    """``Tether.cd_kcu`` and the closed form at the same configuration.

    Zero kite velocity makes every node's apparent wind the log-law wind at
    its height, so the expected force follows from the integrated KCU node
    position alone.
    """
    tether = Tether(
        _KiteStub(),
        kcu,
        ObservationData(),  # optional measurements off -> 7 args
        elastic=False,
        material_name="Dyneema-SK78",
        diameter=0.01,
        n_elements=10,
    )
    length = 200.0
    tension = 1.0e6  # taut: the tether stays near-straight
    direction = np.array([np.cos(elevation_0), 0.0, np.sin(elevation_0)])
    r_kite = direction * (length + kcu.distance_kcu_kite)
    vw = np.asarray(wind, dtype=float)
    args = (elevation_0, 0.0, length, tension, r_kite, np.zeros(3), vw)

    cd_model = float(tether.cd_kcu(*args))

    positions = np.asarray(tether.positions(*args))
    r_kcu = positions[-2]
    axis = (r_kite - r_kcu) / np.linalg.norm(r_kite - r_kcu)

    def log_wind(z):
        z0 = tether_mod.z0
        return np.log(z / z0) / np.log(r_kite[2] / z0) * vw

    va = log_wind(r_kcu[2])
    v_ax = np.dot(va, axis) * axis
    v_n = va - v_ax
    drag = (
        kcu.cd_area_axial * np.linalg.norm(v_ax) * v_ax
        + kcu.cd_area_crossflow * np.linalg.norm(v_n) * v_n
    )  # / (0.5 rho), cancels in the coefficient
    # Like every EKF coefficient, cd_kcu is referenced to the apparent wind
    # at the KITE node (the last tether node), not at the KCU.
    va_kite = log_wind(positions[-1][2])
    cd_expected = np.linalg.norm(drag) / (np.dot(va_kite, va_kite) * _KiteStub.area)
    return cd_model, cd_expected


def test_drag_areas_pair_coefficient_with_its_area():
    kcu = KCU(**LEI_V3_KCU)
    assert kcu.cd_area_axial == pytest.approx(kcu.cdt * np.pi * 0.24**2)
    assert kcu.cd_area_crossflow == pytest.approx(kcu.cdp * 0.48 * 1.0)
    # LEI-V3 1.0 x 0.48 m: crossflow area ~2.2x the axial one.
    assert kcu.cd_area_axial == pytest.approx(0.1505, rel=5e-3)
    assert kcu.cd_area_crossflow == pytest.approx(0.3294, rel=5e-3)


def test_turbine_thrust_joins_crossflow_only():
    bare = KCU(**LEI_V3_KCU)
    with_turbine = KCU(
        **LEI_V3_KCU, turbine_diameter=0.5, turbine_thrust_coefficient=0.85
    )
    thrust_area = 0.85 * np.pi * 0.25**2
    assert with_turbine.cd_area_axial == pytest.approx(bare.cd_area_axial)
    assert with_turbine.cd_area_crossflow == pytest.approx(
        bare.cd_area_crossflow + thrust_area
    )


def test_no_turbine_block_means_no_turbine():
    assert _turbine_config({}) == {
        "turbine_diameter": 0.0,
        "turbine_thrust_coefficient": 0.0,
    }
    cfg = _turbine_config(
        {"turbine": {"structure": {"rotor_diameter": 0.5, "mass": 2.8}}}
    )
    assert cfg["turbine_thrust_coefficient"] == pytest.approx(0.85)
    assert KCU(**LEI_V3_KCU, **cfg).area_turbine == pytest.approx(np.pi * 0.25**2)


def test_kcu_mass_adds_the_turbine_mass_once():
    """awesIO states the turbine mass apart from the KCU's; the EKF lumps both."""
    control_system = {
        "structure": {"mass": 19.2},
        "turbine": {"structure": {"rotor_diameter": 0.5, "mass": 2.8}},
    }
    assert _kcu_mass(control_system) == pytest.approx(22.0)
    assert _kcu_mass({"structure": {"mass": 19.2}}) == pytest.approx(19.2)


@pytest.mark.parametrize("turbine_diameter", [0.0, 0.5])
@pytest.mark.parametrize("elevation_deg", [90.0, 45.0, 25.0])
def test_tether_cd_kcu_matches_closed_form(elevation_deg, turbine_diameter):
    kcu = KCU(
        **LEI_V3_KCU,
        turbine_diameter=turbine_diameter,
        turbine_thrust_coefficient=0.85,
    )
    cd_model, cd_expected = _cd_kcu(kcu, np.radians(elevation_deg))
    assert cd_model == pytest.approx(cd_expected, rel=1e-9)


def test_crossflow_uses_broadside_area():
    """Pure crossflow (vertical KCU, horizontal wind) sees the broadside area.

    The old crossed pairing gave the frontal ``cdt At`` here, ~2.2x lower.
    """
    kcu = KCU(**LEI_V3_KCU)
    cd_model, _ = _cd_kcu(kcu, np.pi / 2)
    # 2 % covers the log-law wind deficit between the KCU and the kite.
    assert cd_model == pytest.approx(
        kcu.cd_area_crossflow / _KiteStub.area, rel=2e-2
    )
    assert cd_model > 1.5 * kcu.cd_area_axial / _KiteStub.area
