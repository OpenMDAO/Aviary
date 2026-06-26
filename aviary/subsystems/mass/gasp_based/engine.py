import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class TotalEngineMass(om.ExplicitComponent):
    """
    Computation of total engine mass, nacelle mass, pylon mass.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        add_aviary_input(
            self, Aircraft.Engine.MASS_SPECIFIC, shape=num_engine_type, units='lbm/lbf'
        )
        add_aviary_input(
            self, Aircraft.Engine.SCALED_SLS_THRUST, shape=num_engine_type, units='lbf'
        )
        add_aviary_input(
            self,
            Aircraft.Nacelle.MASS_SPECIFIC,
            shape=num_engine_type,
            units='lbm/ft**2',
        )
        add_aviary_input(self, Aircraft.Nacelle.SURFACE_AREA, shape=num_engine_type, units='ft**2')
        add_aviary_input(
            self, Aircraft.Nacelle.MASS_SCALER, shape=num_engine_type, units='unitless'
        )
        add_aviary_input(
            self, Aircraft.Engine.PYLON_FACTOR, shape=num_engine_type, units='unitless'
        )

        # for multiengine implementation needs this to always be available
        add_aviary_input(
            self,
            Aircraft.Engine.Propeller.MASS,
            # val=np.full(num_engine_type, 0.000000001),
            val=np.zeros(num_engine_type),
            units='lbm',
            desc='WPROP1: mass of one propeller',
        )

        # add_aviary_output(self, Aircraft.Engine.MASS, units='lbm')
        add_aviary_output(self, Aircraft.Propulsion.TOTAL_ENGINE_MASS, units='lbm')
        add_aviary_output(self, Aircraft.Nacelle.MASS, shape=num_engine_type)
        self.add_output(
            'pylon_mass',
            units='lbm',
            desc='WPYLON: mass of each pylon',
            val=np.zeros(num_engine_type),
        )

    def setup_partials(self):
        # derivatives w.r.t vectorized engine inputs have known sparsity pattern
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])
        shape = np.arange(num_engine_type)

        self.declare_partials(
            Aircraft.Propulsion.TOTAL_ENGINE_MASS,
            [Aircraft.Engine.MASS_SPECIFIC, Aircraft.Engine.SCALED_SLS_THRUST],
        )

        self.declare_partials(
            Aircraft.Nacelle.MASS,
            [
                Aircraft.Nacelle.MASS_SPECIFIC,
                Aircraft.Nacelle.SURFACE_AREA,
                Aircraft.Nacelle.MASS_SCALER,
            ],
            rows=shape,
            cols=shape,
        )

        self.declare_partials(
            'pylon_mass',
            [
                Aircraft.Nacelle.MASS_SPECIFIC,
                Aircraft.Nacelle.SURFACE_AREA,
                Aircraft.Engine.PYLON_FACTOR,
                Aircraft.Engine.MASS_SPECIFIC,
                Aircraft.Engine.SCALED_SLS_THRUST,
                Aircraft.Nacelle.MASS_SCALER,
            ],
            rows=shape,
            cols=shape,
        )

    def compute(self, inputs, outputs):
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]

        eng_spec_wt = inputs[Aircraft.Engine.MASS_SPECIFIC] * GRAV_ENGLISH_LBM
        Fn_SLS = inputs[Aircraft.Engine.SCALED_SLS_THRUST]

        dry_wt_eng = eng_spec_wt * Fn_SLS
        dry_wt_eng_all = dry_wt_eng * num_engines
        outputs[Aircraft.Propulsion.TOTAL_ENGINE_MASS] = sum(dry_wt_eng_all) / GRAV_ENGLISH_LBM

        #######

        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]

        pylon_fac = inputs[Aircraft.Engine.PYLON_FACTOR]
        scaler = inputs[Aircraft.Nacelle.MASS_SCALER]
        spec_nacelle_wt = inputs[Aircraft.Nacelle.MASS_SPECIFIC] * GRAV_ENGLISH_LBM
        nacelle_area = inputs[Aircraft.Nacelle.SURFACE_AREA]

        nacelle_wt = scaler * spec_nacelle_wt * nacelle_area
        pylon_wt = pylon_fac * (dry_wt_eng + nacelle_wt) ** 0.736
        # sec_wt_all = sum((nacelle_wt + pylon_wt) * num_engines)
        # In GASP, WPEI = SKPEI * (WEP + ENP*WTGB), even though WTGB = 0.
        outputs[Aircraft.Nacelle.MASS] = nacelle_wt / GRAV_ENGLISH_LBM
        outputs['pylon_mass'] = pylon_wt / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]

        eng_spec_wt = inputs[Aircraft.Engine.MASS_SPECIFIC] * GRAV_ENGLISH_LBM
        Fn_SLS = inputs[Aircraft.Engine.SCALED_SLS_THRUST]
        spec_nacelle_wt = inputs[Aircraft.Nacelle.MASS_SPECIFIC] * GRAV_ENGLISH_LBM
        nacelle_area = inputs[Aircraft.Nacelle.SURFACE_AREA]
        pylon_fac = inputs[Aircraft.Engine.PYLON_FACTOR]
        scaler = inputs[Aircraft.Nacelle.MASS_SCALER]

        dDWEA_dESW = num_engines * Fn_SLS
        dDWEA_dFNSLS = num_engines * eng_spec_wt
        dNW_dNWS = scaler * nacelle_area
        dPW_dNWS = (
            0.736
            * pylon_fac
            * (eng_spec_wt * Fn_SLS + scaler * spec_nacelle_wt * nacelle_area) ** (-0.264)
            * nacelle_area
        )
        dNW_dNSA = scaler * spec_nacelle_wt
        dPW_dNSA = (
            0.736
            * pylon_fac
            * (eng_spec_wt * Fn_SLS + scaler * spec_nacelle_wt * nacelle_area) ** (-0.264)
            * spec_nacelle_wt
        )
        dPW_dPF = (eng_spec_wt * Fn_SLS + scaler * spec_nacelle_wt * nacelle_area) ** 0.736
        dPW_dEWS = (
            0.736
            * pylon_fac
            * (eng_spec_wt * Fn_SLS + scaler * spec_nacelle_wt * nacelle_area) ** (-0.264)
            * Fn_SLS
        )
        dPW_dSLST = (
            pylon_fac
            * 0.736
            * (eng_spec_wt * Fn_SLS + scaler * spec_nacelle_wt * nacelle_area) ** (-0.264)
            * eng_spec_wt
        )

        J[Aircraft.Propulsion.TOTAL_ENGINE_MASS, Aircraft.Engine.MASS_SPECIFIC] = dDWEA_dESW
        J[Aircraft.Propulsion.TOTAL_ENGINE_MASS, Aircraft.Engine.SCALED_SLS_THRUST] = (
            dDWEA_dFNSLS / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Nacelle.MASS, Aircraft.Nacelle.MASS_SPECIFIC] = dNW_dNWS
        J['pylon_mass', Aircraft.Nacelle.MASS_SPECIFIC] = dPW_dNWS
        J[Aircraft.Nacelle.MASS, Aircraft.Nacelle.SURFACE_AREA] = dNW_dNSA / GRAV_ENGLISH_LBM
        J['pylon_mass', Aircraft.Nacelle.SURFACE_AREA] = dPW_dNSA / GRAV_ENGLISH_LBM
        J['pylon_mass', Aircraft.Engine.PYLON_FACTOR] = dPW_dPF / GRAV_ENGLISH_LBM
        J['pylon_mass', Aircraft.Engine.MASS_SPECIFIC] = dPW_dEWS
        J['pylon_mass', Aircraft.Engine.SCALED_SLS_THRUST] = dPW_dSLST / GRAV_ENGLISH_LBM

        dry_wt_eng = eng_spec_wt * Fn_SLS
        J[Aircraft.Nacelle.MASS, Aircraft.Nacelle.MASS_SCALER] = (
            spec_nacelle_wt * nacelle_area
        ) / GRAV_ENGLISH_LBM
        J['pylon_mass', Aircraft.Nacelle.MASS_SCALER] = (
            pylon_fac
            * 0.736
            * (dry_wt_eng + scaler * spec_nacelle_wt * nacelle_area) ** -0.264
            * spec_nacelle_wt
            * nacelle_area
        ) / GRAV_ENGLISH_LBM

        J[Aircraft.Nacelle.MASS, Aircraft.Nacelle.MASS_SCALER] = (
            spec_nacelle_wt * nacelle_area
        ) / GRAV_ENGLISH_LBM


class EnginePODMass(om.ExplicitComponent):
    """
    Computation of engine pod mass and total engine pod mass.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        add_aviary_input(self, Aircraft.Engine.POD_MASS_SCALER)
        add_aviary_input(self, Aircraft.Nacelle.MASS, shape=num_engine_type, units='lbm')
        self.add_input(
            'pylon_mass',
            units='lbm',
            desc='WPYLON: mass of each pylon',
            val=np.zeros(num_engine_type),
        )

        add_aviary_output(self, Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS, units='lbm', desc='WPES')
        add_aviary_output(self, Aircraft.Engine.POD_MASS, shape=num_engine_type, units='lbm')

    def setup_partials(self):
        # derivatives w.r.t vectorized engine inputs have known sparsity pattern
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])
        shape = np.arange(num_engine_type)

        self.declare_partials(
            Aircraft.Engine.POD_MASS,
            [
                Aircraft.Nacelle.MASS,
                'pylon_mass',
            ],
            rows=shape,
            cols=shape,
        )

        self.declare_partials(
            Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS,
            [
                Aircraft.Nacelle.MASS,
                Aircraft.Engine.POD_MASS_SCALER,
                'pylon_mass',
            ],
        )

    def compute(self, inputs, outputs):
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]
        CK14 = inputs[Aircraft.Engine.POD_MASS_SCALER]
        nacelle_wt = inputs[Aircraft.Nacelle.MASS] * GRAV_ENGLISH_LBM
        pylon_wt = inputs['pylon_mass'] * GRAV_ENGLISH_LBM
        pod_wt = nacelle_wt + pylon_wt
        outputs[Aircraft.Engine.POD_MASS] = pod_wt / GRAV_ENGLISH_LBM
        # NOTE TOTAL_ENGINE_POD_MASS by definition includes everything *in* the pod too! This component
        #      should probably use a new/different variable name (same for pod mass scaler)
        pod_wt_sum = sum(pod_wt * num_engines)
        outputs[Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS] = CK14 * pod_wt_sum / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        CK14 = inputs[Aircraft.Engine.POD_MASS_SCALER]
        nacelle_wt = inputs[Aircraft.Nacelle.MASS] * GRAV_ENGLISH_LBM
        pylon_wt = inputs['pylon_mass'] * GRAV_ENGLISH_LBM
        pod_wt = nacelle_wt + pylon_wt

        J[Aircraft.Engine.POD_MASS, Aircraft.Nacelle.MASS] = np.ones(num_engine_type)
        J[Aircraft.Engine.POD_MASS, 'pylon_mass'] = np.ones(num_engine_type)

        J[Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS, Aircraft.Nacelle.MASS] = CK14 * num_engines

        J[Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS, 'pylon_mass'] = CK14 * num_engines

        J[Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS, Aircraft.Engine.POD_MASS_SCALER] = (
            sum(pod_wt * num_engines) / GRAV_ENGLISH_LBM
        )


class AdditionalEngineMass(om.ExplicitComponent):
    """
    Computation of additional engine mass.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)
        add_aviary_option(self, Aircraft.Engine.ADDITIONAL_MASS_FRACTION)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        add_aviary_input(
            self, Aircraft.Engine.MASS_SPECIFIC, shape=num_engine_type, units='lbm/lbf'
        )
        add_aviary_input(
            self, Aircraft.Engine.SCALED_SLS_THRUST, shape=num_engine_type, units='lbf'
        )
        add_aviary_input(self, Aircraft.Propulsion.MISC_MASS_SCALER, units='unitless')

        add_aviary_output(self, Aircraft.Engine.ADDITIONAL_MASS, shape=num_engine_type, units='lbm')

    def setup_partials(self):
        # derivatives w.r.t vectorized engine inputs have known sparsity pattern
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])
        shape = np.arange(num_engine_type)

        self.declare_partials(
            Aircraft.Engine.ADDITIONAL_MASS,
            [
                Aircraft.Engine.MASS_SPECIFIC,
                Aircraft.Engine.SCALED_SLS_THRUST,
            ],
            rows=shape,
            cols=shape,
            val=1.0,
        )

        self.declare_partials(
            Aircraft.Engine.ADDITIONAL_MASS,
            Aircraft.Propulsion.MISC_MASS_SCALER,
            val=1.0,
        )

    def compute(self, inputs, outputs):
        CK7 = inputs[Aircraft.Propulsion.MISC_MASS_SCALER]
        c_instl = self.options[Aircraft.Engine.ADDITIONAL_MASS_FRACTION]
        eng_spec_wt = inputs[Aircraft.Engine.MASS_SPECIFIC] * GRAV_ENGLISH_LBM
        Fn_SLS = inputs[Aircraft.Engine.SCALED_SLS_THRUST]

        dry_wt_eng = eng_spec_wt * Fn_SLS
        # In GASP, WPEI = SKPEI * (WEP + ENP*WTGB), even though WTGB = 0.
        eng_instl_wt = c_instl * dry_wt_eng

        outputs[Aircraft.Engine.ADDITIONAL_MASS] = CK7 * eng_instl_wt / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        c_instl = self.options[Aircraft.Engine.ADDITIONAL_MASS_FRACTION]

        eng_spec_wt = inputs[Aircraft.Engine.MASS_SPECIFIC] * GRAV_ENGLISH_LBM
        Fn_SLS = inputs[Aircraft.Engine.SCALED_SLS_THRUST]
        CK7 = inputs[Aircraft.Propulsion.MISC_MASS_SCALER]

        J[Aircraft.Engine.ADDITIONAL_MASS, Aircraft.Engine.MASS_SPECIFIC] = CK7 * c_instl * Fn_SLS
        J[Aircraft.Engine.ADDITIONAL_MASS, Aircraft.Engine.SCALED_SLS_THRUST] = (
            CK7 * c_instl * eng_spec_wt / GRAV_ENGLISH_LBM
        )
        J[Aircraft.Engine.ADDITIONAL_MASS, Aircraft.Propulsion.MISC_MASS_SCALER] = (
            c_instl * eng_spec_wt * Fn_SLS / GRAV_ENGLISH_LBM
        )


class WingMountEngineMass(om.ExplicitComponent):
    """
    Computation of total engine mass, nacelle mass, pylon mass, total engine pod mass,
    additional engine mass.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Electrical.HAS_HYBRID_SYSTEM)
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)
        add_aviary_option(self, Aircraft.Engine.ADDITIONAL_MASS_FRACTION)
        add_aviary_option(self, Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])
        total_num_wing_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES]

        add_aviary_input(
            self, Aircraft.Engine.MASS_SPECIFIC, shape=num_engine_type, units='lbm/lbf'
        )
        add_aviary_input(
            self, Aircraft.Engine.SCALED_SLS_THRUST, shape=num_engine_type, units='lbf'
        )
        add_aviary_input(self, Aircraft.Engine.MASS_SCALER, shape=num_engine_type, units='unitless')
        add_aviary_input(self, Aircraft.Propulsion.MISC_MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.LandingGear.MAIN_GEAR_MASS, units='lbm')
        add_aviary_input(self, Aircraft.LandingGear.MAIN_GEAR_LOCATION, units='unitless')

        if total_num_wing_engines > 1:
            add_aviary_input(
                self,
                Aircraft.Engine.WING_LOCATIONS,
                shape=int(total_num_wing_engines / 2),
                units='unitless',
            )
        else:
            add_aviary_input(self, Aircraft.Engine.WING_LOCATIONS, units='unitless')

        has_hybrid_system = self.options[Aircraft.Electrical.HAS_HYBRID_SYSTEM]
        if has_hybrid_system:
            self.add_input(
                'aug_mass',
                units='lbm',
                desc='WEAUG: mass of electrical augmentation system',
            )

        # for multiengine implementation needs this to always be available
        add_aviary_input(
            self,
            Aircraft.Engine.Propeller.MASS,
            # val=np.full(num_engine_type, 0.000000001),
            val=np.zeros(num_engine_type),
            units='lbm',
            desc='WPROP1: mass of one propeller',
        )
        add_aviary_input(self, Aircraft.Engine.POD_MASS, shape=num_engine_type, units='lbm')
        add_aviary_input(self, Aircraft.Engine.ADDITIONAL_MASS, shape=num_engine_type, units='lbm')

        self.add_output(
            'eng_comb_mass',
            units='lbm',
            desc='WPSTAR: combined mass of dry engine and engine installation,'
            ' includes mass of electrical augmentation system',
        )
        self.add_output(
            'wing_mounted_mass',
            units='lbm',
            desc='WM: mass of gear and engine, basically everything mounted on the wing',
        )

        self.add_output('prop_mass_sum', units='lbm', desc='WPROP: mass of all propellers')

    def setup_partials(self):
        has_hybrid_system = self.options[Aircraft.Electrical.HAS_HYBRID_SYSTEM]

        self.declare_partials('prop_mass_sum', [Aircraft.Engine.Propeller.MASS])

        # derivatives w.r.t vectorized engine inputs have known sparsity pattern
        # num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])
        # shape = np.arange(num_engine_type)

        self.declare_partials(
            'wing_mounted_mass',
            [
                Aircraft.Engine.WING_LOCATIONS,
                Aircraft.Engine.MASS_SPECIFIC,
                Aircraft.Engine.SCALED_SLS_THRUST,
                Aircraft.LandingGear.MAIN_GEAR_MASS,
                Aircraft.LandingGear.MAIN_GEAR_LOCATION,
                Aircraft.Propulsion.MISC_MASS_SCALER,
                Aircraft.Engine.POD_MASS,
                Aircraft.Engine.ADDITIONAL_MASS,
                Aircraft.Engine.Propeller.MASS,
            ],
        )
        if not has_hybrid_system:
            self.declare_partials(
                'eng_comb_mass',
                [
                    Aircraft.Engine.MASS_SCALER,
                    Aircraft.Engine.MASS_SPECIFIC,
                    Aircraft.Engine.SCALED_SLS_THRUST,
                    Aircraft.Engine.ADDITIONAL_MASS,
                ],
            )
        else:
            self.declare_partials(
                'eng_comb_mass',
                [
                    Aircraft.Engine.MASS_SCALER,
                    Aircraft.Engine.MASS_SPECIFIC,
                    Aircraft.Engine.SCALED_SLS_THRUST,
                    Aircraft.Engine.ADDITIONAL_MASS,
                    'aug_mass',
                ],
            )

    def compute(self, inputs, outputs):
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]
        num_engine_type = len(num_engines)
        CK7 = inputs[Aircraft.Propulsion.MISC_MASS_SCALER]
        eng_spec_wt = inputs[Aircraft.Engine.MASS_SPECIFIC] * GRAV_ENGLISH_LBM
        Fn_SLS = inputs[Aircraft.Engine.SCALED_SLS_THRUST]
        dry_wt_eng = eng_spec_wt * Fn_SLS
        dry_wt_eng_all = dry_wt_eng * num_engines

        eng_instl_wt = inputs[Aircraft.Engine.ADDITIONAL_MASS] / CK7 * GRAV_ENGLISH_LBM
        eng_instl_wt_all = eng_instl_wt * num_engines

        CK5 = inputs[Aircraft.Engine.MASS_SCALER]
        eng_span_frac = inputs[Aircraft.Engine.WING_LOCATIONS]
        eng_additional_mass_sum = sum(inputs[Aircraft.Engine.ADDITIONAL_MASS] * num_engines)
        pod_wt = inputs[Aircraft.Engine.POD_MASS] * GRAV_ENGLISH_LBM
        pod_wt_all = pod_wt * num_engines

        # In GASP, WPSTAR=CK5*WEP+CK7*WPEI+WPROP+WTGB*ENP, even though the last two terms are 0.
        if self.options[Aircraft.Electrical.HAS_HYBRID_SYSTEM]:
            aug_mass = inputs['aug_mass']
            outputs['eng_comb_mass'] = (
                sum(CK5 * dry_wt_eng_all) / GRAV_ENGLISH_LBM + eng_additional_mass_sum + aug_mass
            )
        else:
            outputs['eng_comb_mass'] = (
                sum(CK5 * dry_wt_eng_all) / GRAV_ENGLISH_LBM + eng_additional_mass_sum
            )

        prop_wt = inputs[Aircraft.Engine.Propeller.MASS] * GRAV_ENGLISH_LBM
        prop_wt_all = prop_wt * num_engines
        outputs['prop_mass_sum'] = sum(prop_wt_all) / GRAV_ENGLISH_LBM

        span_frac_factor = eng_span_frac / (eng_span_frac + 0.001)
        # sum span_frac_factor for each engine type
        span_frac_factor_sum = np.zeros(num_engine_type, dtype=Fn_SLS.dtype)
        idx = 0
        for i in range(num_engine_type):
            # fmt: off
            span_frac_factor_sum[i] = sum(span_frac_factor[idx : idx + num_engines[i]])
            # fmt: on
            idx = idx + num_engines[i]

        main_gear_wt = inputs[Aircraft.LandingGear.MAIN_GEAR_MASS] * GRAV_ENGLISH_LBM
        loc_main_gear = inputs[Aircraft.LandingGear.MAIN_GEAR_LOCATION]
        # In GASP,
        # WM = YP/(YP+.001)*(WEP+WPEI+WPES+WPROP+ENP*WTGB)
        #      + WMG*YMG/(YMG+.001)
        #      + WCMIN*YC/(YC+.001)
        outputs['wing_mounted_mass'] = (
            sum(
                span_frac_factor_sum
                * (dry_wt_eng_all + eng_instl_wt_all + pod_wt_all + prop_wt_all)
            )
            + main_gear_wt * loc_main_gear / (loc_main_gear + 0.001)
        ) / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]
        num_engine_type = len(num_engines)
        c_instl = self.options[Aircraft.Engine.ADDITIONAL_MASS_FRACTION]

        eng_spec_wt = inputs[Aircraft.Engine.MASS_SPECIFIC] * GRAV_ENGLISH_LBM
        Fn_SLS = inputs[Aircraft.Engine.SCALED_SLS_THRUST]
        CK5 = inputs[Aircraft.Engine.MASS_SCALER]
        CK7 = inputs[Aircraft.Propulsion.MISC_MASS_SCALER]
        eng_span_frac = inputs[Aircraft.Engine.WING_LOCATIONS]
        main_gear_wt = inputs[Aircraft.LandingGear.MAIN_GEAR_MASS] * GRAV_ENGLISH_LBM
        loc_main_gear = inputs[Aircraft.LandingGear.MAIN_GEAR_LOCATION]

        J['eng_comb_mass', Aircraft.Engine.MASS_SCALER] = (
            eng_spec_wt * Fn_SLS * num_engines / GRAV_ENGLISH_LBM
        )
        J['eng_comb_mass', Aircraft.Engine.MASS_SPECIFIC] = CK5 * num_engines * Fn_SLS
        J['eng_comb_mass', Aircraft.Engine.SCALED_SLS_THRUST] = (
            CK5 * num_engines * eng_spec_wt / GRAV_ENGLISH_LBM
        )
        J['eng_comb_mass', Aircraft.Engine.ADDITIONAL_MASS] = num_engines

        J['prop_mass_sum', Aircraft.Engine.Propeller.MASS] = num_engines

        dry_wt_eng = eng_spec_wt * Fn_SLS
        pod_wt = inputs[Aircraft.Engine.POD_MASS] * GRAV_ENGLISH_LBM
        eng_instl_wt = c_instl * dry_wt_eng
        prop_wt = inputs[Aircraft.Engine.Propeller.MASS] * GRAV_ENGLISH_LBM
        # prop_wt_all = sum(num_engines * prop_wt) / GRAV_ENGLISH_LBM
        span_frac_factor = eng_span_frac / (eng_span_frac + 0.001)
        # sum span_frac_factor for each engine type
        span_frac_factor_sum = np.zeros(num_engine_type, dtype=Fn_SLS.dtype)
        wing_mass_deriv = np.zeros(len(span_frac_factor), dtype=Fn_SLS.dtype)
        idx = 0
        # wing_mass_vec = (eng_spec_wt * Fn_SLS * (1 + c_instl) +
        #                 sec_wt + prop_wt) * num_engines
        wing_mass_vec = (dry_wt_eng + eng_instl_wt + pod_wt + prop_wt) * num_engines
        for i in range(num_engine_type):
            span_frac_factor_sum[i] = sum(span_frac_factor[idx : idx + num_engines[i]])
            wing_mass_deriv[idx : idx + num_engines[i]] = wing_mass_vec[i]
            idx = idx + num_engines[i]

        J['wing_mounted_mass', Aircraft.Engine.WING_LOCATIONS] = (
            0.001 / (eng_span_frac + 0.001) ** 2 * wing_mass_deriv / GRAV_ENGLISH_LBM
        )
        J['wing_mounted_mass', Aircraft.Engine.MASS_SPECIFIC] = (
            span_frac_factor_sum * (Fn_SLS) * num_engines
        )
        J['wing_mounted_mass', Aircraft.Engine.SCALED_SLS_THRUST] = (
            span_frac_factor_sum * (num_engines * eng_spec_wt) / GRAV_ENGLISH_LBM
        )
        J['wing_mounted_mass', Aircraft.LandingGear.MAIN_GEAR_MASS] = loc_main_gear / (
            loc_main_gear + 0.001
        )
        J['wing_mounted_mass', Aircraft.LandingGear.MAIN_GEAR_LOCATION] = (
            main_gear_wt
            / GRAV_ENGLISH_LBM
            * ((loc_main_gear + 0.001) - loc_main_gear)
            / (loc_main_gear + 0.001) ** 2
        )
        J['wing_mounted_mass', Aircraft.Engine.Propeller.MASS] = span_frac_factor_sum * num_engines
        J['wing_mounted_mass', Aircraft.Engine.POD_MASS] = span_frac_factor_sum * num_engines
        J['wing_mounted_mass', Aircraft.Engine.ADDITIONAL_MASS] = (
            span_frac_factor_sum / CK7 * num_engines
        )
        J['wing_mounted_mass', Aircraft.Propulsion.MISC_MASS_SCALER] = sum(
            -span_frac_factor_sum * inputs[Aircraft.Engine.ADDITIONAL_MASS] / CK7**2 * num_engines
        )
        if self.options[Aircraft.Electrical.HAS_HYBRID_SYSTEM]:
            J['eng_comb_mass', 'aug_mass'] = 1


class EngineMassGroup(om.Group):
    """Group of all engine components for GASP-based mass."""

    def setup(self):
        self.add_subsystem(
            'total_engine',
            TotalEngineMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'engine_pod',
            EnginePODMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'additional_engine',
            AdditionalEngineMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'wing_engine',
            WingMountEngineMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
