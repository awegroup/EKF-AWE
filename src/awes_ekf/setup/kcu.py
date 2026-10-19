import numpy as np
from scipy.interpolate import splrep, splev


class KCU:
    # Exracted from Applied fluid dynamics handbook
    ldt_data = np.array([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])  # L/D values
    cdt_data = np.array(
        [1.15, 1.1, 0.93, 0.85, 0.83, 0.85, 0.85, 0.85]
    )  # Cd values for tangential flow
    ldp_data = np.array([1, 1.98, 2.96, 5, 10, 20, 40, 1e6])  # L/D values
    cdp_data = np.array(
        [0.64, 0.68, 0.74, 0.74, 0.82, 0.91, 0.98, 1.2]
    )  # Cd values perpendicular flow

    cdt_cone_data = np.array([0.43, 0.35, 0.22, 0.19, 0.20, 0.21, 0.22, 0.24])
    ldt_cone = np.array([0.01, 0.64, 1.98, 4.57, 8.91, 12.02, 13.69, 15.29])
    # Create spline interpolations
    spline_t = splrep(ldt_data, cdt_data, s=0)
    spline_p = splrep(ldp_data, cdp_data, s=0)

    def __init__(self, **kwargs):

        self.length: float = float(kwargs.get("length") or 0.0)
        self.diameter: float = float(kwargs.get("diameter") or 0.0)
        self.mass: float = float(kwargs.get("mass") or 0.0)
        self.distance_kcu_kite: float = float(kwargs.get("distance_kcu_kite") or 0.0)
        self.total_length_bridle_lines: float = float(kwargs.get("total_length_bridle_lines") or 0.0)
        self.diameter_bridle_lines: float = float(kwargs.get("diameter_bridle_lines") or 0.0)

        # Optional onboard wind turbine (rotor diameter 0 = none). Its
        # momentum-theory thrust C_T * pi (D/2)^2 is its drag.
        self.turbine_diameter: float = float(kwargs.get("turbine_diameter") or 0.0)
        self.turbine_thrust_coefficient: float = float(
            kwargs.get("turbine_thrust_coefficient") or 0.0
        )

        # cdt: flow ALONG the KCU axis, on the frontal area At (cdt(L/D -> 0)
        # = 1.15 is a disc facing the flow). cdp: flow ACROSS the axis, on the
        # broadside area Ap (cdp(L/D -> inf) = 1.2 is an infinite cylinder).
        self.cdt = splev(self.length / self.diameter, KCU.spline_t)
        self.cdp = splev(self.length / self.diameter, KCU.spline_p)

        self.At = np.pi * (self.diameter / 2) ** 2  # Frontal area of the KCU
        self.Ap = self.diameter * self.length  # Broadside area of the KCU

        # Drag areas [m^2] the tether model applies to the axial and crossflow
        # apparent-wind components at the KCU. The turbine sits on top of the
        # KCU facing the apparent wind, shadowed from the along-tether flow,
        # so its thrust joins the crossflow area only.
        self.area_turbine = np.pi * (self.turbine_diameter / 2) ** 2
        self.cd_area_axial = float(self.cdt * self.At)
        self.cd_area_crossflow = float(
            self.cdp * self.Ap + self.turbine_thrust_coefficient * self.area_turbine
        )

    def calculate_cd(self, ld):
        ##
        self.cd_kcu = 0
