import numpy as np
import openmdao.api as om

from aviary import constants
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Dynamic


class FlightConditions(om.ExplicitComponent):
    """
    Given a speed type (TAS, MACH, or EAS) and air density,
    compute dynamic pressure and other two speed types.
    """

    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare(
            "input_speed_type",
            default=SpeedType.TAS,
            types=SpeedType,
            desc="defines input airspeed as equivalent airspeed, true airspeed, or mach number",
        )

    def setup(self):
        nn = self.options["num_nodes"]
        in_type = self.options["input_speed_type"]
        arange = np.arange(self.options["num_nodes"])

        self.add_input(
            Dynamic.Mission.DENSITY,
            val=np.zeros(nn),
            units="slug/ft**3",
            desc="density of air",
        )
        self.add_input(
            Dynamic.Mission.SPEED_OF_SOUND,
            val=np.zeros(nn),
            units="ft/s",
            desc="speed of sound",
        )

        self.add_output(
            Dynamic.Mission.DYNAMIC_PRESSURE,
            val=np.zeros(nn),
            units="lbf/ft**2",
            desc="dynamic pressure",
        )

        if in_type is SpeedType.TAS:
            self.add_input(
                Dynamic.Mission.VELOCITY,
                val=np.zeros(nn),
                units="ft/s",
                desc="true air speed",
            )
            self.add_output(
                "EAS",
                val=np.zeros(nn),
                units="ft/s",
                desc="equivalent air speed",
            )
            self.add_output(
                Dynamic.Mission.MACH,
                val=np.zeros(nn),
                units="unitless",
                desc="mach number",
            )

            self.declare_partials(
                Dynamic.Mission.DYNAMIC_PRESSURE,
                [Dynamic.Mission.DENSITY, Dynamic.Mission.VELOCITY],
                rows=arange,
                cols=arange,
            )
            self.declare_partials(
                Dynamic.Mission.MACH,
                [Dynamic.Mission.SPEED_OF_SOUND, Dynamic.Mission.VELOCITY],
                rows=arange,
                cols=arange,
            )
            self.declare_partials(
                "EAS",
                [Dynamic.Mission.VELOCITY, Dynamic.Mission.DENSITY],
                rows=arange,
                cols=arange,
            )
        elif in_type is SpeedType.EAS:
            self.add_input(
                "EAS",
                val=np.zeros(nn),
                units="ft/s",
                desc="equivalent air speed at",
            )
            self.add_output(
                Dynamic.Mission.VELOCITY,
                val=np.zeros(nn),
                units="ft/s",
                desc="true air speed",
            )
            self.add_output(
                Dynamic.Mission.MACH,
                val=np.zeros(nn),
                units="unitless",
                desc="mach number",
            )

            self.declare_partials(
                Dynamic.Mission.DYNAMIC_PRESSURE,
                [Dynamic.Mission.DENSITY, "EAS"],
                rows=arange,
                cols=arange,
            )
            self.declare_partials(
                Dynamic.Mission.MACH,
                [
                    Dynamic.Mission.SPEED_OF_SOUND,
                    "EAS",
                    Dynamic.Mission.DENSITY,
                ],
                rows=arange,
                cols=arange,
            )
            self.declare_partials(
                Dynamic.Mission.VELOCITY,
                [Dynamic.Mission.DENSITY, "EAS"],
                rows=arange,
                cols=arange,
            )
        elif in_type is SpeedType.MACH:
            self.add_input(
                Dynamic.Mission.MACH,
                val=np.zeros(nn),
                units="unitless",
                desc="mach number",
            )
            self.add_output(
                "EAS",
                val=np.zeros(nn),
                units="ft/s",
                desc="equivalent air speed",
            )
            self.add_output(
                Dynamic.Mission.VELOCITY,
                val=np.zeros(nn),
                units="ft/s",
                desc="true air speed",
            )

            self.declare_partials(
                Dynamic.Mission.DYNAMIC_PRESSURE,
                [
                    Dynamic.Mission.SPEED_OF_SOUND,
                    Dynamic.Mission.MACH,
                    Dynamic.Mission.DENSITY,
                ],
                rows=arange,
                cols=arange,
            )
            self.declare_partials(
                Dynamic.Mission.VELOCITY,
                [Dynamic.Mission.SPEED_OF_SOUND, Dynamic.Mission.MACH],
                rows=arange,
                cols=arange,
            )
            self.declare_partials(
                "EAS",
                [
                    Dynamic.Mission.SPEED_OF_SOUND,
                    Dynamic.Mission.MACH,
                    Dynamic.Mission.DENSITY,
                ],
                rows=arange,
                cols=arange,
            )

    def compute(self, inputs, outputs):

        in_type = self.options["input_speed_type"]

        rho = inputs[Dynamic.Mission.DENSITY]
        sos = inputs[Dynamic.Mission.SPEED_OF_SOUND]

        if in_type is SpeedType.TAS:
            TAS = inputs[Dynamic.Mission.VELOCITY]
            outputs[Dynamic.Mission.MACH] = mach = TAS / sos
            outputs["EAS"] = TAS * (rho / constants.RHO_SEA_LEVEL_ENGLISH) ** 0.5
            outputs[Dynamic.Mission.DYNAMIC_PRESSURE] = 0.5 * rho * TAS**2

        elif in_type is SpeedType.EAS:
            EAS = inputs["EAS"]
            outputs[Dynamic.Mission.VELOCITY] = TAS = (
                EAS / (rho / constants.RHO_SEA_LEVEL_ENGLISH) ** 0.5
            )
            outputs[Dynamic.Mission.MACH] = mach = TAS / sos
            outputs[Dynamic.Mission.DYNAMIC_PRESSURE] = (
                0.5 * EAS**2 * constants.RHO_SEA_LEVEL_ENGLISH
            )

        elif in_type is SpeedType.MACH:
            mach = inputs[Dynamic.Mission.MACH]
            outputs[Dynamic.Mission.VELOCITY] = TAS = sos * mach
            outputs["EAS"] = TAS * (rho / constants.RHO_SEA_LEVEL_ENGLISH) ** 0.5
            outputs[Dynamic.Mission.DYNAMIC_PRESSURE] = 0.5 * rho * sos**2 * mach**2

    def compute_partials(self, inputs, J):
        in_type = self.options["input_speed_type"]

        rho = inputs[Dynamic.Mission.DENSITY]
        sos = inputs[Dynamic.Mission.SPEED_OF_SOUND]

        if in_type is SpeedType.TAS:
            TAS = inputs[Dynamic.Mission.VELOCITY]

            J[Dynamic.Mission.DYNAMIC_PRESSURE, Dynamic.Mission.VELOCITY] = rho * TAS
            J[Dynamic.Mission.DYNAMIC_PRESSURE, Dynamic.Mission.DENSITY] = 0.5 * TAS**2

            J[Dynamic.Mission.MACH, Dynamic.Mission.VELOCITY] = 1 / sos
            J[Dynamic.Mission.MACH, Dynamic.Mission.SPEED_OF_SOUND] = -TAS / sos**2

            J["EAS", Dynamic.Mission.VELOCITY] = (
                rho / constants.RHO_SEA_LEVEL_ENGLISH
            ) ** 0.5
            J["EAS", Dynamic.Mission.DENSITY] = (
                TAS * 0.5 * (rho ** (-0.5) / constants.RHO_SEA_LEVEL_ENGLISH**0.5)
            )

        elif in_type is SpeedType.EAS:
            EAS = inputs["EAS"]
            TAS = EAS / (rho / constants.RHO_SEA_LEVEL_ENGLISH) ** 0.5

            dTAS_dRho = -0.5 * EAS * constants.RHO_SEA_LEVEL_ENGLISH**0.5 / rho**1.5
            dTAS_dEAS = 1 / (rho / constants.RHO_SEA_LEVEL_ENGLISH) ** 0.5

            J[Dynamic.Mission.DYNAMIC_PRESSURE, "EAS"] = (
                EAS * constants.RHO_SEA_LEVEL_ENGLISH
            )
            J[Dynamic.Mission.MACH, "EAS"] = dTAS_dEAS / sos
            J[Dynamic.Mission.MACH, Dynamic.Mission.DENSITY] = dTAS_dRho / sos
            J[Dynamic.Mission.MACH, Dynamic.Mission.SPEED_OF_SOUND] = -TAS / sos**2
            J[Dynamic.Mission.VELOCITY, Dynamic.Mission.DENSITY] = dTAS_dRho
            J[Dynamic.Mission.VELOCITY, "EAS"] = dTAS_dEAS

        elif in_type is SpeedType.MACH:
            mach = inputs[Dynamic.Mission.MACH]
            TAS = sos * mach

            J[Dynamic.Mission.DYNAMIC_PRESSURE, Dynamic.Mission.SPEED_OF_SOUND] = (
                rho * sos * mach**2
            )
            J[Dynamic.Mission.DYNAMIC_PRESSURE, Dynamic.Mission.MACH] = (
                rho * sos**2 * mach
            )
            J[Dynamic.Mission.DYNAMIC_PRESSURE, Dynamic.Mission.DENSITY] = (
                0.5 * sos**2 * mach**2
            )
            J[Dynamic.Mission.VELOCITY, Dynamic.Mission.SPEED_OF_SOUND] = mach
            J[Dynamic.Mission.VELOCITY, Dynamic.Mission.MACH] = sos
            J["EAS", Dynamic.Mission.SPEED_OF_SOUND] = (
                mach * (rho / constants.RHO_SEA_LEVEL_ENGLISH) ** 0.5
            )
            J["EAS", Dynamic.Mission.MACH] = (
                sos * (rho / constants.RHO_SEA_LEVEL_ENGLISH) ** 0.5
            )
            J["EAS", Dynamic.Mission.DENSITY] = (
                TAS * (1 / constants.RHO_SEA_LEVEL_ENGLISH) ** 0.5 * 0.5 * rho ** (-0.5)
            )
