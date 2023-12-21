import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class DetailedWingBendingFact(om.ExplicitComponent):

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        aviary_options: AviaryValues = self.options['aviary_options']
        input_station_dist = aviary_options.get_val(Aircraft.Wing.INPUT_STATION_DIST)
        num_input_stations = len(input_station_dist)
        num_wing_engines = aviary_options.get_val(Aircraft.Engine.NUM_WING_ENGINES)
        count = len(aviary_options.get_val('engine_models'))

        add_aviary_input(self, Aircraft.Wing.LOAD_PATH_SWEEP_DIST,
                         val=np.zeros(num_input_stations - 1))

        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_DIST,
                         val=np.zeros(num_input_stations))

        add_aviary_input(self, Aircraft.Wing.CHORD_PER_SEMISPAN_DIST,
                         val=np.zeros(num_input_stations))

        add_aviary_input(self, Mission.Design.GROSS_MASS, val=0.0)

        add_aviary_input(self, Aircraft.Engine.POD_MASS, val=0.0)

        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, val=0.0)

        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO_REF, val=0.0)

        add_aviary_input(self, Aircraft.Wing.STRUT_BRACING_FACTOR, val=0.0)

        add_aviary_input(self, Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR, val=0.0)

        add_aviary_input(self, Aircraft.Engine.WING_LOCATIONS,
                         val=np.zeros([count, int(num_wing_engines/2)]))

        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD, val=0.0)

        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_REF, val=0.0)

        add_aviary_output(self, Aircraft.Wing.BENDING_FACTOR, val=0.0)

        add_aviary_output(self, Aircraft.Wing.ENG_POD_INERTIA_FACTOR, val=0.0)

    def setup_partials(self):
        # TODO: Analytic derivs will be challenging, but possible.
        self.declare_partials("*", "*", method='cs')

    def compute(self, inputs, outputs):
        aviary_options: AviaryValues = self.options['aviary_options']
        input_station_dist = aviary_options.get_val(Aircraft.Wing.INPUT_STATION_DIST)
        inp_stations = np.array(input_station_dist)
        num_integration_stations = \
            aviary_options.get_val(Aircraft.Wing.NUM_INTEGRATION_STATIONS)
        num_wing_engines = aviary_options.get_val(Aircraft.Engine.NUM_WING_ENGINES)

        # TODO: Support all options for this parameter.
        # 0.0 : input distribution
        # 1.0 : triangular distribution
        # 2.0 : elliptical distribution (default)
        # 3.0 : rectangular distribution
        # 1.0-2.0 : blend of triangular and elliptical
        # 2.0-3.0 : blend of elliptical and rectangular
        load_distribution_factor = \
            aviary_options.get_val(Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL)

        load_path_sweep = inputs[Aircraft.Wing.LOAD_PATH_SWEEP_DIST]
        thickness_to_chord = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_DIST]
        chord = inputs[Aircraft.Wing.CHORD_PER_SEMISPAN_DIST]
        engine_locations = inputs[Aircraft.Engine.WING_LOCATIONS]
        gross_mass = inputs[Mission.Design.GROSS_MASS]
        # TODO currently using pod mass of engines even if not mounted on wing
        pod_mass = inputs[Aircraft.Engine.POD_MASS]
        fstrt = inputs[Aircraft.Wing.STRUT_BRACING_FACTOR]
        faert = inputs[Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR]

        ar = inputs[Aircraft.Wing.ASPECT_RATIO]
        arref = inputs[Aircraft.Wing.ASPECT_RATIO_REF]
        tc = inputs[Aircraft.Wing.THICKNESS_TO_CHORD]
        tcref = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_REF]

        # NOTE look at leaps1 code for examples on multi-engine support
        #    - will require list of pod masses for each engine
        # Currently implementation only partially addresses issues - odd numbers of wing
        # mounted engines pretend the "odd" engine out is not on the wing and is ignored
        # There are also no checks that number of engine locations is consistent with
        # half of number of wing mounted engines, which should get added to preprocessor

        # flatten engine variables into single lists, sorty by wing location
        engine_locations = engine_locations.flatten()
        # there should be a pod mass for every engine location
        pod_mass = np.repeat(pod_mass.flatten(), (1+num_wing_engines) % 2)
        engine_data = np.vstack((pod_mass, engine_locations))
        engine_data = engine_data.transpose()[np.lexsort(engine_data)]
        pod_mass = engine_data[:, 0]
        engine_locations = engine_data[:, 1]

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
                np.linspace(inp_stations[i], val, per_section, endpoint=endpoint))
            sweep_int_stations = np.append(sweep_int_stations,
                                           load_path_sweep[i] * np.ones(per_section))

        dy = np.diff(integration_stations)
        avg_sweep = np.sum((dy[1:] + 2.0 * integration_stations[1:-1]) * dy[1:] *
                           sweep_int_stations[1:-1])

        # TODO: add all load_distribution_factor options
        if load_distribution_factor == 1:
            load_intensity = 1.0 - integration_stations
        elif load_distribution_factor == 2:
            load_intensity = np.sqrt(1.0 - integration_stations ** 2)
        elif load_distribution_factor == 3:
            load_intensity = np.ones(num_integration_stations + 1)
        else:
            raise om.AnalysisError(
                f'{load_distribution_factor} is not a valid value for {Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL}, it must be "1", "2", or "3".')

        chord_interp = InterpND(method='slinear', points=(inp_stations),
                                x_interp=integration_stations)
        chord_int_stations = chord_interp.evaluate_spline(
            chord, compute_derivative=False)
        if arref > 0.0:
            # Scale
            chord_int_stations *= arref / ar

        del_load = dy * (
            chord_int_stations[:-1] * (2*load_intensity[:-1] + load_intensity[1:]) +
            chord_int_stations[1:] * (2*load_intensity[1:] + load_intensity[:-1])) / 6

        el = np.sum(del_load)

        del_moment = dy**2 * (
            chord_int_stations[:-1] * (load_intensity[:-1]+load_intensity[1:]) +
            chord_int_stations[1:] * (3*load_intensity[1:]+load_intensity[:-1])) / 12

        load_path_length = np.flip(
            np.append(np.zeros(1), np.cumsum(np.flip(del_load)[:-1])))
        csw = 1. / np.cos(sweep_int_stations[:-1] * np.pi/180.)
        emi = (del_moment + dy * load_path_length) * csw
        # em = np.sum(emi)

        tc_interp = InterpND(method='slinear', points=(inp_stations),
                             x_interp=integration_stations)
        tc_int_stations = tc_interp.evaluate_spline(
            thickness_to_chord, compute_derivative=False)
        if tcref > 0.0:
            tc_int_stations *= tc / tcref

        total_moment = np.cumsum(emi[::-1])[::-1]

        bma = total_moment * csw / (chord_int_stations[:-1] * tc_int_stations[:-1])

        pm = np.sum((bma[:-1] + bma[1:]) * dy[:-1] * 0.5)

        # s = np.sum((chord_int_stations[:-1] + chord_int_stations[1:]) * dy * 0.5)

        btb = 4 * pm / el

        sa = np.sin(avg_sweep * np.pi / 180.)
        if ar <= 5.0:
            caya = 0.0
        else:
            caya = ar - 5.0

        bt = btb / (ar**(0.25*fstrt) * (1.0 + (0.5*faert - 0.16*fstrt)
                    * sa**2 + 0.03*caya * (1.0-0.5*faert)*sa))
        outputs[Aircraft.Wing.BENDING_FACTOR] = bt

        eel = np.zeros(len(dy) + 1)
        loc = np.where(integration_stations < engine_locations[0])[0]
        eel[loc] = 1.0

        delme = dy * eel[1:]
        delme[loc[-1]] = engine_locations[0] - integration_stations[loc[-1]]

        eem = delme * csw
        eem = np.cumsum(eem[::-1])[::-1]

        ea = eem * csw / (chord_int_stations[:-1] * tc_int_stations[:-1])

        bte = 8 * np.sum((ea[:-1] + ea[1:]) * dy[:-1] * 0.5)

        outputs[Aircraft.Wing.ENG_POD_INERTIA_FACTOR] = 1.0 - \
            bte / bt * pod_mass / gross_mass
