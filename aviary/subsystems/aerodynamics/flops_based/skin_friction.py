import numpy as np
import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft, Dynamic


class SkinFriction(om.ImplicitComponent):
    """
    Computes skin friction coefficient using the Sommer and Short T Prime method as used
    in FLOPS AERSCL.

    The fixed-point iteration scheme has been replaced with Newton's method, which can
    converge the equations for multiple mach numbers and characteristic lengths
    simultaneously.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.CONLOG = 2.302585
        self.TAW = 1.0
        self.sea_level_pressure = 14.6959 * 144  # psi -> psf

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        self.linear_solver = om.DirectSolver()
        self.nonlinear_solver.options['iprint'] = -1
        self.linear_solver.options['iprint'] = -1

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare(
            'num_nodes', types=int, default=1,
            desc='The number of points at which the cross product is computed.')
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        nn = self.options['num_nodes']
        aviary_options: AviaryValues = self.options['aviary_options']
        zero_count = (0, None)
        num_tails, _ = aviary_options.get_item(
            Aircraft.VerticalTail.NUM_TAILS, zero_count)
        num_fuselages, _ = aviary_options.get_item(
            Aircraft.Fuselage.NUM_FUSELAGES, zero_count)
        # TODO does not used vectorized multi-engine. Temp using single engine
        num_engines, _ = aviary_options.get_item(
            Aircraft.Engine.NUM_ENGINES, zero_count)
        self.nc = nc = 2 + num_tails + num_fuselages + int(sum(num_engines))

        # Simulation inputs
        self.add_input(Dynamic.Mission.TEMPERATURE, np.ones(nn), units='degR')
        self.add_input(Dynamic.Mission.STATIC_PRESSURE, np.ones(nn), units='lbf/ft**2')
        self.add_input(Dynamic.Mission.MACH, np.ones(nn), units='unitless')

        # Aero subsystem inputs
        self.add_input('characteristic_lengths', np.ones(nc), units='ft')

        self.add_output('cf_iter', np.ones((nn, nc)), units='unitless')
        self.add_output('skin_friction_coeff', np.ones((nn, nc)), units='unitless')
        self.add_output('Re', np.ones((nn, nc)), units='unitless')
        self.add_output('wall_temp', np.ones((nn, nc)), units='degR')

    def setup_partials(self):
        nn = self.options["num_nodes"]
        nc = self.nc
        n = nn * nc

        row_col = np.arange(n)
        self.declare_partials('Re', 'Re', rows=row_col, cols=row_col, val=1.0)
        self.declare_partials(
            'skin_friction_coeff', 'skin_friction_coeff',
            rows=row_col, cols=row_col, val=1.0)

        self.declare_partials(
            'cf_iter', ['wall_temp', 'cf_iter'], rows=row_col, cols=row_col)
        self.declare_partials(
            'wall_temp', ['wall_temp', 'cf_iter'], rows=row_col, cols=row_col)
        self.declare_partials(
            'skin_friction_coeff', ['wall_temp', 'cf_iter'], rows=row_col, cols=row_col)

        col = np.arange(nn)
        cols = np.repeat(col, nc)
        self.declare_partials(
            'cf_iter', [Dynamic.Mission.TEMPERATURE, Dynamic.Mission.STATIC_PRESSURE, Dynamic.Mission.MACH], rows=row_col, cols=cols)
        self.declare_partials(
            'wall_temp', [Dynamic.Mission.TEMPERATURE, Dynamic.Mission.STATIC_PRESSURE, Dynamic.Mission.MACH], rows=row_col, cols=cols)
        self.declare_partials(
            'Re', [Dynamic.Mission.TEMPERATURE, Dynamic.Mission.STATIC_PRESSURE, Dynamic.Mission.MACH], rows=row_col, cols=cols)
        self.declare_partials(
            'skin_friction_coeff', [Dynamic.Mission.TEMPERATURE,
                                    Dynamic.Mission.STATIC_PRESSURE, Dynamic.Mission.MACH],
            rows=row_col, cols=cols)

        col = np.arange(nc)
        cols = np.tile(col, nn)
        self.declare_partials('Re', 'characteristic_lengths', rows=row_col, cols=cols)
        self.declare_partials(
            'cf_iter', 'characteristic_lengths', rows=row_col, cols=cols)

    def guess_nonlinear(self, inputs, outputs, resids):
        nn = self.options["num_nodes"]
        nc = self.nc

        rankine, pressure, mach, length = inputs.values()

        Pratio = pressure / self.sea_level_pressure
        kelvin = rankine / 1.8
        RE = 1.479301E9 * Pratio * (kelvin + 110.4) / kelvin ** 2

        # REYNOLDS NUMBER
        reynolds_num = np.einsum('i,i,j->ij', RE, mach, length)
        outputs['Re'] = reynolds_num

        # Initial guess for WALL TEMPERATURE
        wall_temp = (1.0 + 0.176 * mach * mach) * rankine
        wall_temp = np.tile(wall_temp, nc).reshape((nc, nn)).T
        outputs['wall_temp'] = wall_temp
        self.TAW = wall_temp

        # INITIAL GUESS AT SKIN FRICTION COEFFICIENT
        outputs['cf_iter'] = (0.242 / (np.log(reynolds_num * 0.0015) / self.CONLOG)) ** 2

    def apply_nonlinear(self, inputs, outputs, residuals):
        T, pressure, mach, length = inputs.values()
        cf = outputs['cf_iter']
        wall_temp = outputs['wall_temp']
        nc = self.nc

        Pratio = pressure / self.sea_level_pressure
        kelvin = T / 1.8
        RE = 1.479301E9 * Pratio * (kelvin + 110.4) / kelvin ** 2

        # SUTHERLAND'S CONSTANT IS 198.72 DEG R FROM 1962 ON
        suth_const = T + 198.72
        E = 0.80

        # COMBINED CONSTANT INCLUDING 1/RHO
        combined_const = 4.593153E-6 * E * suth_const / (RE * mach * T ** 1.5)

        # REYNOLDS NUMBER
        reynolds_num = np.einsum('i,i,j->ij', RE, mach, length)
        residuals['Re'] = outputs['Re'] - reynolds_num

        # WALL TEMPERATURE RATIO
        wall_temp_ratio = 1.0 + 0.45 * (np.einsum('ij,i->ij', wall_temp,  1.0 / T) - 1.0)
        fact = 0.035 * mach * mach
        for i in range(nc):
            wall_temp_ratio[:, i] += fact

        CFL = cf / (1.0 + 3.59 * np.sqrt(cf) * wall_temp_ratio)

        prod = np.einsum('j,jk->jk', combined_const, wall_temp ** 3)
        residuals['wall_temp'] = (
            self.TAW / (1.0 + prod / CFL) + wall_temp) * 0.5 - wall_temp

        RP = (
            reynolds_num * (np.einsum('ij,i->ij', wall_temp_ratio, T) + 198.72)
            / np.einsum('i,ij->ij', suth_const, wall_temp_ratio ** 2.5))

        residuals['cf_iter'] = (0.242 * self.CONLOG / np.log(RP * cf)) ** 2 - cf

        residuals['skin_friction_coeff'] = \
            outputs['skin_friction_coeff'] - outputs['cf_iter'] / wall_temp_ratio

    def linearize(self, inputs, outputs, partials):
        nn = self.options["num_nodes"]
        nc = self.nc

        T, pressure, mach, length = inputs.values()
        cf = outputs['cf_iter']
        wall_temp = outputs['wall_temp']

        Pratio = pressure / self.sea_level_pressure
        kelvin = T / 1.8
        RE = 1.479301E9 * Pratio * (kelvin + 110.4) / kelvin ** 2

        dRE_dp = 1.479301E9 * (kelvin + 110.4) / (self.sea_level_pressure * kelvin ** 2)
        dRE_dT = -1.479301E9 / 1.8 * (
            Pratio * (1.0 / kelvin ** 2 + 2.0 * 110.4 / kelvin ** 3))

        reynolds_num = np.einsum('i,i,j->ij', RE, mach, length)
        dreyn_dp = np.einsum('i,i,j->ij', dRE_dp, mach, length)
        dreyn_dT = np.einsum('i,i,j->ij', dRE_dT, mach, length)
        dreyn_dmach = np.einsum('i,j->ij', RE, length)
        dreyn_dlen = np.tile(RE * mach, nc).reshape((nc, nn)).T

        partials['Re', Dynamic.Mission.STATIC_PRESSURE] = -dreyn_dp.ravel()
        partials['Re', Dynamic.Mission.TEMPERATURE] = -dreyn_dT.ravel()
        partials['Re', Dynamic.Mission.MACH] = -dreyn_dmach.ravel()
        partials['Re', 'characteristic_lengths'] = -dreyn_dlen.ravel()

        suth_const = T + 198.72
        E = 0.80
        combined_const = 4.593153E-6 * E * suth_const / (RE * mach * T ** 1.5)
        dcomb_dRE = -4.593153E-6 * E * suth_const / (RE * RE * mach * T ** 1.5)
        dcomb_dmach = -4.593153E-6 * E * suth_const / (RE * mach * mach * T ** 1.5)
        dcomb_dT = 4.593153E-6 * E * (1.0 / T ** 1.5 - 1.5 * suth_const / T ** 2.5) / (
            RE * mach) + dcomb_dRE * dRE_dT
        dcomb_dp = dcomb_dRE * dRE_dp

        wall_temp_ratio = 1.0 + 0.45 * (np.einsum('ij,i->ij', wall_temp,  1.0 / T) - 1.0)
        fact = 0.035 * mach * mach
        for i in range(nc):
            wall_temp_ratio[:, i] += fact
        dwtr_dmach = 0.07 * mach
        dwtr_dT = -0.45 * np.einsum('ij,i->ij', wall_temp, 1.0 / T ** 2)
        dwtr_dwt = 0.45 / T

        den = 1.0 + 3.59 * np.sqrt(cf) * wall_temp_ratio
        CFL = cf / den
        dCFL_dcf = \
            1.0 / den - cf * 3.59 * wall_temp_ratio * 0.5 / (np.sqrt(cf) * den ** 2)
        dCFL_dwtr = - cf * 3.59 * np.sqrt(cf) / den ** 2
        dCFL_dmach = np.einsum('ij,i->ij', dCFL_dwtr, dwtr_dmach)
        dCFL_dT = dCFL_dwtr * dwtr_dT
        dCFL_dwt = np.einsum('ij,i->ij', dCFL_dwtr, dwtr_dwt)

        term = np.einsum('i,ij->ij', combined_const, wall_temp ** 3)
        den = 1.0 + term / CFL
        dreswt_dcomb = -0.5 * self.TAW * wall_temp ** 3 / (CFL * den ** 2)
        dreswt_dCFL = 0.5 * self.TAW * term / (CFL * den) ** 2
        dreswt_dwt = (
            -0.5 - 1.5 * self.TAW * np.einsum('i,ij->ij', combined_const, wall_temp ** 2)
            / (CFL * den ** 2))

        partials['wall_temp', Dynamic.Mission.STATIC_PRESSURE] = (
            np.einsum('ij,i->ij', dreswt_dcomb, dcomb_dp)).ravel()
        partials['wall_temp', Dynamic.Mission.TEMPERATURE] = (
            np.einsum('ij,i->ij', dreswt_dcomb, dcomb_dT)
            + dreswt_dCFL * dCFL_dT).ravel()
        partials['wall_temp', Dynamic.Mission.MACH] = (
            np.einsum('ij,i->ij', dreswt_dcomb, dcomb_dmach)
            + dreswt_dCFL * dCFL_dmach).ravel()
        partials['wall_temp', 'wall_temp'] = (
            dreswt_dCFL * dCFL_dwt + dreswt_dwt).ravel()
        partials['wall_temp', 'cf_iter'] = (dreswt_dCFL * dCFL_dcf).ravel()

        den = 1.0 / np.einsum('i,ij->ij', suth_const, wall_temp_ratio ** 2.5)
        num = np.einsum('ij,i->ij', wall_temp_ratio, T) + 198.72
        RP = reynolds_num * num * den
        dRP_dreyn = num * den
        term = np.einsum('i,ij->ij', T, 1.0 * den)
        dRP_dwtr = reynolds_num * (
            term - num * 2.5
            * np.einsum('i,ij->ij', suth_const, wall_temp_ratio ** 1.5) * den ** 2)
        dRP_dT = reynolds_num * (
            wall_temp_ratio * den - num * wall_temp_ratio ** 2.5 * den ** 2)
        dRP_dp = dRP_dreyn * dreyn_dp
        dRP_dT = dRP_dreyn * dreyn_dT + dRP_dwtr * dwtr_dT + dRP_dT
        dRP_dmach = dRP_dreyn * dreyn_dmach + np.einsum('ij,i->ij', dRP_dwtr, dwtr_dmach)
        dRP_dlen = dRP_dreyn * dreyn_dlen
        dRP_dwt = np.einsum('ij,i->ij', dRP_dwtr, dwtr_dwt)

        fact = (0.242 * self.CONLOG) ** 2
        drescf_dRP = -2.0 * fact / (RP * np.log(RP * cf) ** 3)
        drescf_dcf = -2.0 * fact / (cf * np.log(RP * cf) ** 3) - 1.0

        partials['cf_iter', Dynamic.Mission.STATIC_PRESSURE] = (
            drescf_dRP * dRP_dp).ravel()
        partials['cf_iter', Dynamic.Mission.TEMPERATURE] = (drescf_dRP * dRP_dT).ravel()
        partials['cf_iter', Dynamic.Mission.MACH] = (drescf_dRP * dRP_dmach).ravel()
        partials['cf_iter', 'characteristic_lengths'] = (drescf_dRP * dRP_dlen).ravel()
        partials['cf_iter', 'wall_temp'] = (drescf_dRP * dRP_dwt).ravel()
        partials['cf_iter', 'cf_iter'] = drescf_dcf.ravel()

        dskf_dwtr = outputs['cf_iter'] / wall_temp_ratio ** 2

        partials['skin_friction_coeff', Dynamic.Mission.TEMPERATURE] = (
            dskf_dwtr * dwtr_dT).ravel()
        partials['skin_friction_coeff', Dynamic.Mission.MACH] = np.einsum(
            'ij,i->ij', dskf_dwtr, dwtr_dmach).ravel()
        partials['skin_friction_coeff', 'wall_temp'] = np.einsum(
            'ij,i->ij', dskf_dwtr, dwtr_dwt).ravel()
        partials['skin_friction_coeff', 'cf_iter'] = (- 1.0 / wall_temp_ratio).ravel()
