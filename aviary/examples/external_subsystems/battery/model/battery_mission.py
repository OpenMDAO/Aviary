"""
Battery Performance Modeling on Maxwell X-57
Jeffrey C. Chin, Sydney L. Schnulo, Thomas B. Miller,
Kevin Prokopius, and Justin Gray
http://openmdao.org/pubs/chin_battery_performance_x57_2019.pdf

Thevenin voltage equation based on paper
"Evaluation of Lithium Ion Battery Equivalent Circuit Models
for State of Charge Estimation by an Experimental Approach"
Hongwen H, Rui Xiong, Jinxin Fan

Equivalent Circuit component values derived from
"High Fidelity Electrical Model with Thermal Dependence for
 Characterization and Simulation of High Power Lithium Battery Cells"
Tarun Huria, Massimo Ceraolo, Javier Gazzarri, Robyn Jackey
"""

from openmdao.api import Group
from .cell_comp import CellComp
from .reg_thevenin_interp_group import RegTheveninInterpGroup

from aviary.utils.aviary_values import AviaryValues
from aviary.examples.external_subsystems.battery.battery_variables import Aircraft, Mission


class BatteryMission(Group):
    """Assembly to connect subcomponents of the Thevenin Battery Equivalent
    Circuit Model From Interpolated Performance Maps
    """

    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare(
            'aviary_inputs', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
            default=None,
        )

    def setup(self):
        n = self.options["num_nodes"]

        self.add_subsystem(
            name="interp_group",
            subsys=RegTheveninInterpGroup(num_nodes=n),
        )

        self.add_subsystem(name="cell", subsys=CellComp(num_nodes=n))

        # Connect internal names
        self.connect("interp_group.U_oc", "cell.U_oc")
        self.connect("interp_group.C_Th", "cell.C_Th")
        self.connect("interp_group.R_Th", "cell.R_Th")
        self.connect("interp_group.R_0", "cell.R_0")

        # Promote interp group inputs to match Aviary variable names
        self.promotes(
            "interp_group",
            inputs=[
                ("T_batt", "mission:battery:temperature"),
                ("SOC", "mission:battery:state_of_charge"),
            ],
        )

        self.promotes(
            "cell",
            inputs=[
                Mission.Battery.VOLTAGE_THEVENIN,
                Mission.Battery.CURRENT,
                Aircraft.Battery.N_SERIES,
                Aircraft.Battery.N_PARALLEL,
                Aircraft.Battery.Cell.ENERGY_CAPACITY_MAX,
            ],
        )

        self.promotes(
            "cell",
            outputs=[
                Mission.Battery.VOLTAGE,
                Mission.Battery.VOLTAGE_THEVENIN_RATE,
                Mission.Battery.STATE_OF_CHARGE_RATE,
                Mission.Battery.HEAT_OUT,
                Aircraft.Battery.EFFICIENCY,
            ],
        )
