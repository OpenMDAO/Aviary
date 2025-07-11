import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class DetailedWingBendingFact(om.ExplicitComponent):
    """
    Computation of wing bending factor and engine inertia relief factor
    used for FLOPS-based detailed wing mass estimation.

    If one or zero wing-mounted engines are present, it is assumed there is no engine
    inertial relief factor (i.e. the single engine is mounted at the wing root)
    """

    # Basically, Engine.WING_LOCATIONS is ignored if there are one or fewer wing engines

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)
        add_aviary_option(self, Aircraft.Engine.NUM_WING_ENGINES)
        add_aviary_option(self, Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES)
        add_aviary_option(self, Aircraft.Wing.INPUT_STATION_DIST)
        add_aviary_option(self, Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL)
        add_aviary_option(self, Aircraft.Wing.NUM_INTEGRATION_STATIONS)

    def setup(self):
        input_station_dist = self.options[Aircraft.Wing.INPUT_STATION_DIST]
        num_input_stations = len(input_station_dist)
        total_num_wing_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES]
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        add_aviary_input(
            self, Aircraft.Wing.LOAD_PATH_SWEEP_DIST, shape=num_input_stations - 1, units='deg'
        )
        add_aviary_input(
            self, Aircraft.Wing.THICKNESS_TO_CHORD_DIST, shape=num_input_stations, units='unitless'
        )
        add_aviary_input(
            self, Aircraft.Wing.CHORD_PER_SEMISPAN_DIST, shape=num_input_stations, units='unitless'
        )
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Engine.POD_MASS, shape=num_engine_type, units='lbm')
        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO_REF, units='unitless')
        add_aviary_input(self, Aircraft.Wing.STRUT_BRACING_FACTOR, units='unitless')
        add_aviary_input(self, Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR, units='unitless')

        if total_num_wing_engines > 1:
            add_aviary_input(
                self,
                Aircraft.Engine.WING_LOCATIONS,
                shape=int(total_num_wing_engines / 2),
                units='unitless',
            )
        else:
            add_aviary_input(self, Aircraft.Engine.WING_LOCATIONS, units='unitless')

        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD, units='unitless')
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_REF, units='unitless')

        add_aviary_output(self, Aircraft.Wing.BENDING_MATERIAL_FACTOR, units='unitless')
        add_aviary_output(self, Aircraft.Wing.ENG_POD_INERTIA_FACTOR, units='unitless')

    def setup_partials(self):
        # TODO: Analytic derivs will be challenging, but possible.
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        input_station_dist = self.options[Aircraft.Wing.INPUT_STATION_DIST]
        inp_stations = np.array(input_station_dist)
        num_integration_stations = self.options[Aircraft.Wing.NUM_INTEGRATION_STATIONS]
        num_wing_engines = self.options[Aircraft.Engine.NUM_WING_ENGINES]
        num_engine_type = len(num_wing_engines)

        # TODO: Support all options for this parameter.
        # 0.0 : input distribution
        # 1.0 : triangular distribution
        # 2.0 : elliptical distribution (default)
        # 3.0 : rectangular distribution
        # 1.0-2.0 : blend of triangular and elliptical
        # 2.0-3.0 : blend of elliptical and rectangular
        load_distribution_factor = self.options[Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL]

        load_path_sweep = inputs[Aircraft.Wing.LOAD_PATH_SWEEP_DIST]
        thickness_to_chord = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_DIST]
        chord = inputs[Aircraft.Wing.CHORD_PER_SEMISPAN_DIST]
        engine_locations = inputs[Aircraft.Engine.WING_LOCATIONS]
        gross_mass = inputs[Mission.Design.GROSS_MASS]
        # NOTE pod mass assumed the same for wing/non-wing mounted engines, only using
        #      wing mounted pods here
        pod_mass = inputs[Aircraft.Engine.POD_MASS]
        fstrt = inputs[Aircraft.Wing.STRUT_BRACING_FACTOR]
        faert = inputs[Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR]

        ar = inputs[Aircraft.Wing.ASPECT_RATIO]
        arref = inputs[Aircraft.Wing.ASPECT_RATIO_REF]
        tc = inputs[Aircraft.Wing.THICKNESS_TO_CHORD]
        tcref = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_REF]

        # NOTE changes to FLOPS routines based on LEAPS1 improved multiengine effort
        # odd numbers of wing mounted engines assume the "odd" engine out is not on the
        # wing and is ignored
        # TODO There are also no checks that number of engine locations is consistent with
        # half of number of wing mounted engines, which should get added to preprocessor

        target_dy = (inp_stations[-1] - inp_stations[0]) / num_integration_stations
        stations_per_section = np.floor(np.abs(np.diff(inp_stations) / target_dy + 0.5))
        stations_per_section[-1] += 1  # add one more point to the last section
        integration_stations = np.empty(0, dtype=chord.dtype)
        sweep_int_stations = np.empty(0, dtype=chord.dtype)

        for i, val in enumerate(inp_stations[1:]):
            endpoint = i == len(inp_stations) - 2
            per_section = int(stations_per_section[i])
            integration_stations = np.append(
                integration_stations,
                np.linspace(inp_stations[i], val, per_section, endpoint=endpoint),
            )
            sweep_int_stations = np.append(
                sweep_int_stations, load_path_sweep[i] * np.ones(per_section)
            )

        dy = np.diff(integration_stations)
        avg_sweep = np.sum(
            (dy[1:] + 2.0 * integration_stations[1:-1]) * dy[1:] * sweep_int_stations[1:-1]
        )

        # TODO: add all load_distribution_factor options
        if load_distribution_factor == 1:
            load_intensity = 1.0 - integration_stations
        elif load_distribution_factor == 2:
            load_intensity = np.sqrt(1.0 - integration_stations**2)
        elif load_distribution_factor == 3:
            load_intensity = np.ones(num_integration_stations + 1)
        else:
            raise om.AnalysisError(
                f'{load_distribution_factor} is not a valid value for '
                f'{Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL}, it must be "1", "2", or "3".'
            )

        chord_interp = InterpND(
            method='slinear', points=(inp_stations), x_interp=integration_stations
        )
        chord_int_stations = chord_interp.evaluate_spline(chord, compute_derivative=False)
        if arref > 0.0:
            # Scale
            chord_int_stations *= arref / ar

        del_load = (
            dy
            * (
                chord_int_stations[:-1] * (2 * load_intensity[:-1] + load_intensity[1:])
                + chord_int_stations[1:] * (2 * load_intensity[1:] + load_intensity[:-1])
            )
            / 6
        )

        el = np.sum(del_load)

        del_moment = (
            dy**2
            * (
                chord_int_stations[:-1] * (load_intensity[:-1] + load_intensity[1:])
                + chord_int_stations[1:] * (3 * load_intensity[1:] + load_intensity[:-1])
            )
            / 12
        )

        load_path_length = np.flip(
            np.append(np.zeros(1, chord.dtype), np.cumsum(np.flip(del_load)[:-1]))
        )
        csw = 1.0 / np.cos(sweep_int_stations[:-1] * np.pi / 180.0)
        emi = (del_moment + dy * load_path_length) * csw
        # em = np.sum(emi)

        tc_interp = InterpND(method='slinear', points=(inp_stations), x_interp=integration_stations)
        tc_int_stations = tc_interp.evaluate_spline(thickness_to_chord, compute_derivative=False)
        if tcref > 0.0:
            tc_int_stations *= tc / tcref

        total_moment = np.cumsum(emi[::-1])[::-1]

        bma = total_moment * csw / (chord_int_stations[:-1] * tc_int_stations[:-1])

        pm = np.sum((bma[:-1] + bma[1:]) * dy[:-1] * 0.5)

        # s = np.sum((chord_int_stations[:-1] + chord_int_stations[1:]) * dy * 0.5)

        btb = 4 * pm / el

        sa = np.sin(avg_sweep * np.pi / 180.0)
        if ar <= 5.0:
            caya = 0.0
        else:
            caya = ar - 5.0

        bt = btb / (
            ar ** (0.25 * fstrt)
            * (1.0 + (0.5 * faert - 0.16 * fstrt) * sa**2 + 0.03 * caya * (1.0 - 0.5 * faert) * sa)
        )

        outputs[Aircraft.Wing.BENDING_MATERIAL_FACTOR] = bt

        inertia_factor = np.zeros(num_engine_type, dtype=chord.dtype)
        eel = np.zeros(len(dy) + 1, dtype=chord.dtype)

        # idx is the index where this engine type begins in location list
        idx = 0
        # i is the counter for which engine model we are checking
        for i in range(num_engine_type):
            # idx2 is the last index for the range of engines of this type
            idx2 = idx + int(num_wing_engines[i] / 2)
            if num_wing_engines[i] > 1:
                # engine locations must be in order from wing root to tip
                eng_loc = np.sort(engine_locations[idx:idx2])

            else:
                continue

            if eng_loc[0] <= integration_stations[0]:
                inertia_factor[i] = 1.0

            elif eng_loc[0] >= integration_stations[-1]:
                inertia_factor[i] = 0.84

            else:
                eel[:] = 0.0
                # Find all points on integration station before first engine
                loc = np.where(integration_stations < eng_loc[0])[0]
                eel[loc] = 1.0

                delme = dy * eel[1:]

                delme[loc[-1]] = eng_loc[0] - integration_stations[loc[-1]]

                eem = delme * csw
                eem = np.cumsum(eem[::-1])[::-1]

                ea = eem * csw / (chord_int_stations[:-1] * tc_int_stations[:-1])

                bte = 8 * np.sum((ea[:-1] + ea[1:]) * dy[:-1] * 0.5)

                inertia_factor_i = 1 - bte / bt[0] * pod_mass[i] / gross_mass[0]
                # avoid passing an array into specific index of inertia_factor
                inertia_factor[i] = inertia_factor_i

            # increment idx to next engine set
            idx = idx2

        # LEAPS updated multiengine routine applies each engine pod's factor
        # multiplicatively, and enforces a minimum bound of 0.84
        inertia_factor_prod = np.prod(inertia_factor)
        if inertia_factor_prod < 0.84:
            inertia_factor_prod = 0.84

        outputs[Aircraft.Wing.ENG_POD_INERTIA_FACTOR] = inertia_factor_prod
