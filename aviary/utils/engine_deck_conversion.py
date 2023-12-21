#!/usr/bin/python

import argparse
import getpass
import itertools
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import openmdao.api as om
from dymos.models.atmosphere import USatm1976Comp
from openmdao.components.interp_util.interp import InterpND

from aviary.subsystems.propulsion.engine_deck import normalize
from aviary.subsystems.propulsion.utils import EngineModelVariables, default_units
from aviary.variable_info.variables import Dynamic
from aviary.utils.csv_data_file import write_data_file
from aviary.utils.functions import get_path
from aviary.utils.named_values import NamedValues


class EngineDeckType(Enum):
    FLOPS = 'FLOPS'
    GASP = 'GASP'


MACH = EngineModelVariables.MACH
ALTITUDE = EngineModelVariables.ALTITUDE
THROTTLE = EngineModelVariables.THROTTLE
HYBRID_THROTTLE = EngineModelVariables.HYBRID_THROTTLE
THRUST = EngineModelVariables.THRUST
GROSS_THRUST = EngineModelVariables.GROSS_THRUST
RAM_DRAG = EngineModelVariables.RAM_DRAG
FUEL_FLOW = EngineModelVariables.FUEL_FLOW
ELECTRIC_POWER = EngineModelVariables.ELECTRIC_POWER
NOX_RATE = EngineModelVariables.NOX_RATE
TEMPERATURE = EngineModelVariables.TEMPERATURE_ENGINE_T4
# EXIT_AREA = EngineModelVariables.EXIT_AREA

flops_keys = [
    MACH,
    ALTITUDE,
    THROTTLE,
    GROSS_THRUST,
    RAM_DRAG,
    FUEL_FLOW,
    NOX_RATE]  # , EXIT_AREA]

# later code assumes T4 is last item in keys
gasp_keys = [MACH, ALTITUDE, THROTTLE, THRUST, FUEL_FLOW, TEMPERATURE]

header_names = {
    MACH: 'Mach_Number',
    ALTITUDE: 'Altitude',
    THROTTLE: 'Throttle',
    THRUST: 'Thrust',
    GROSS_THRUST: 'Gross_Thrust',
    RAM_DRAG: 'Ram_Drag',
    FUEL_FLOW: 'Fuel_Flow',
    NOX_RATE: 'NOx_Rate',
    TEMPERATURE: 'T4',
    # EXIT_AREA: 'Exit Area',
}


def EngineDeckConverter(input_file=None, output_file=None, data_format=None):
    '''
    Converts FLOPS- or GASP-formatted engine decks into Aviary csv format.
    FLOPS decks are changed from column-delimited to csv format with added headers.
    GASP decks are reorganized into csv. T4 is recovered using assumptions used in GASPy.
    Data points whose T4 exceeds T4max are removed.

    Parameters
    ----------
    input_file : (str, Path)
        path to engine deck file to be converted
    output_file : (str, Path)
        path to file where new converted data will be written
    data_format : (EngineDeckType)
        data format used by input_file (FLOPS or GASP)
    readable : (bool)
        output_file will be organized with consistent column widths for easier reading
    '''
    # TODO rounding for calculated values?

    timestamp = datetime.now().strftime('%m/%d/%y at %H:%M')
    user = getpass.getuser()
    comments = []
    header = {}
    data = {}

    data_file = get_path(input_file)

    comments.append(f'# created {timestamp} by {user}')
    comments.append(
        f'# {data_format.value}-derived engine deck converted from {data_file.name}')
    if data_format == EngineDeckType.FLOPS:
        header = {key: default_units[key] for key in flops_keys}
        data = {key: np.array([]) for key in flops_keys}

        with open(data_file, newline='', encoding='utf-8-sig') as file:
            reader = _read_flops_engine(file)
            data_starts = False

            for line in reader:
                if not data_starts:
                    # pull out comments before data starts
                    if line[0][0] == '#':
                        comments.append(line)
                        continue
                    data_starts = True

                data[MACH] = np.append(data[MACH], line[0])
                data[ALTITUDE] = np.append(data[ALTITUDE], line[1])
                data[THROTTLE] = np.append(data[THROTTLE], line[2])
                data[GROSS_THRUST] = np.append(data[GROSS_THRUST], line[3])
                data[RAM_DRAG] = np.append(data[RAM_DRAG], line[4])
                data[FUEL_FLOW] = np.append(data[FUEL_FLOW], line[5])
                data[NOX_RATE] = np.append(data[NOX_RATE], line[6])
                # data[EXIT_AREA].append(line[7])

    elif data_format == EngineDeckType.GASP:
        data = {key: [] for key in gasp_keys}

        scalars, tables = _read_gasp_engine(data_file)
        # save scalars as comments
        comments.extend(['# ' + key + ': ' + str(scalars[key])
                        for key in scalars.keys()])

        # recommended to always generate structured grid
        structure_data = True
        if structure_data:
            structured_data = _make_structured_grid(tables, method='lagrange3')

            data[MACH] = structured_data['thrust']['machs']
            data[ALTITUDE] = structured_data['thrust']['alts']
            data[THRUST] = structured_data['thrust']['vals']
            data[FUEL_FLOW] = structured_data['fuelflow']['vals']
            T4T2 = structured_data['thrust']['t4t2s']

        else:
            data[MACH] = tables['thrust'][:, 2]
            data[ALTITUDE] = tables['thrust'][:, 0]
            data[THRUST] = tables['thrust'][:, 3]
            data[FUEL_FLOW] = tables['fuelflow'][:, 3]
            T4T2 = tables['thrust'][:, 1]

        generate_flight_idle = True
        if generate_flight_idle:
            data, T4T2 = _generate_flight_idle(data, T4T2,
                                               ref_sls_airflow=scalars['sls_airflow'],
                                               ref_sfn_idle=scalars['sfn_idle'])

        t4max = scalars['t4max']

        # if t4max 100 or less, it is actually throttle. Remove temperature as variable
        if t4max <= 100:
            compute_T4 = False
            data.pop(TEMPERATURE)
            # temperature is assumed last in keys
            gasp_keys.pop(-1)
        else:
            compute_T4 = True

        # define header now that we know what is in the engine deck
        header = {key: default_units[key] for key in gasp_keys}

        if compute_T4:
            # compute T4 using atmospheric model
            prob = om.Problem()

            prob.model.add_subsystem('T4T2', om.IndepVarComp('T4:T2',
                                                             T4T2,
                                                             units='unitless'),
                                     promotes=['*'])

            prob.model.add_subsystem(
                Dynamic.Mission.MACH,
                om.IndepVarComp(
                    Dynamic.Mission.MACH,
                    data[MACH],
                    units='unitless'),
                promotes=['*'])

            prob.model.add_subsystem(
                Dynamic.Mission.ALTITUDE,
                om.IndepVarComp(
                    Dynamic.Mission.ALTITUDE,
                    data[ALTITUDE],
                    units='ft'),
                promotes=['*'])

            prob.model.add_subsystem(
                name='atmosphere',
                subsys=USatm1976Comp(num_nodes=len(data[MACH])),
                promotes_inputs=[('h', Dynamic.Mission.ALTITUDE)],
                promotes_outputs=[('temp', Dynamic.Mission.TEMPERATURE)])

            prob.model.add_subsystem(
                name='conversion',
                subsys=AtmosCalc(num_nodes=len(data[MACH])),
                promotes_inputs=[Dynamic.Mission.MACH,
                                 Dynamic.Mission.TEMPERATURE],
                promotes_outputs=['t2']
            )

            prob.setup()
            prob.run_model()

            T2 = prob.get_val('t2')
            T4 = T2 * T4T2
            data[TEMPERATURE] = T4
            # Throttle is T4 normalized from 0 to 1 (T4max)
            # By always keeping minimum T4 zero for normalization, throttle stays
            #   consistent with fraction of T4max
            # TODO flight condition dependent throttle range?
            # NOTE this often leaves max throttles less than 1 in the deck - this caues
            #     problems when finding reference SLS thrust, as there is often no max
            #     power data at that point in the engine deck. It is reccomended GASP
            #     engine decks override Aircraft.Engine.REFERENCE_THRUST in EngineDecks
            data[THROTTLE] = normalize(data[TEMPERATURE], minimum=0.0, maximum=t4max)

            # remove all points above T4max
            # TODO save these points as commented out?
            valid_idx = np.where(data[THROTTLE] <= 1.0)
            data[MACH] = data[MACH][valid_idx]
            data[ALTITUDE] = data[ALTITUDE][valid_idx]
            data[THROTTLE] = data[THROTTLE][valid_idx]
            data[THRUST] = data[THRUST][valid_idx]
            data[FUEL_FLOW] = data[FUEL_FLOW][valid_idx]
            data[TEMPERATURE] = data[TEMPERATURE][valid_idx]

        else:
            data[THROTTLE] = T4T2

        # data needs to be string so column length can be easily found later
        for var in data:
            data[var] = np.array([str(item) for item in data[var]])

    else:
        quit("Invalid engine deck format provided")

    # sort data
    # create parallel dict to data that stores floats
    formatted_data = {}
    for key in data:
        formatted_data[key] = data[key].astype(float)

    # convert engine_data from dict to list so it can be sorted
    sorted_values = np.array([formatted_data[key] for key in formatted_data]).transpose()

    # Sort by mach, then altitude, then throttle, then hybrid throttle
    sorted_values = sorted_values[np.lexsort(
        [formatted_data[THROTTLE],
         formatted_data[ALTITUDE],
         formatted_data[MACH]])]
    for idx, key in enumerate(formatted_data):
        formatted_data[key] = sorted_values[:, idx]

    # rework formatted_data into 2d array
    formatted_data = np.array([formatted_data[key]
                              for key in header.keys()]).transpose().astype(float)

    # store formatted data into NamedValues object
    write_data = NamedValues()
    for idx, key in enumerate(data):
        write_data.set_val(header_names[key], data[key], default_units[key])

    write_data_file(output_file, write_data, comments, include_timestamp=False)


def _read_flops_engine(input_file):
    '''
    Read engine data file using FLOPS standard, which is column delimited data
    always assumed to be in the order defined in the FLOPS manual
    '''
    for line in input_file:
        sz = len(line)

        if sz < 80:
            line = line + (80 - sz) * ' '

        if line[0] == '#':
            data = line.strip()
        else:
            data = [
                _flops_field_convert(line[0:5]),
                _flops_field_convert(line[5:15]),
                _flops_field_convert(line[15:20]),
                _flops_field_convert(line[20:30]),
                _flops_field_convert(line[30:40]),
                _flops_field_convert(line[40:50]),
                # intenional gap from 50:60 - column is left blank in FLOPS standard
                _flops_field_convert(line[60:70]),
                _flops_field_convert(line[70:80]),
            ]

        yield data


def _flops_field_convert(arg: str):
    rvalue = arg.strip()

    if not rvalue:
        rvalue = _flops_empty_field

    return rvalue


# in FLOPS, empty fields are converted to zero
_flops_empty_field = '0'


def _read_gasp_engine(fp):
    """Read a GASP engine deck file and parse its scalars and tabular data.
    Scalars (T4 max, SLS airflow, etc.) are read from the first line of the engine deck
    (IREAD=1 is assumed) and returned in a dictionary.
    Data tables are also returned as a dictionary, with separate tables for thrust,
    fuelflow, and airflow, since they may have different grids in general.
    Each table consists of both the independent variables and the dependent variable for
    the corresponding field. The table is a "tidy format" 2D array where the first three
    columns are the independent varaiables (altitude, T4/T2, and Mach number) and the
    final column is the dependent variable (one of thrust, fuelflow, or airflow).
    """
    with open(fp, "r") as f:
        scalars = _read_header(f)
        tables = {k: _read_table(f) for k in ["thrust", "fuelflow", "airflow"]}

    return scalars, tables


def _read_header(f):
    """Read GASP engine deck header, returning the engine scalars in a dict"""
    # file header: FORMAT(2I5,10X,5F10.4)
    iread, iprint, wamap, t4max, t4mcl, t4mc, sfnidl = _parse(
        f, [*_rep(2, (int, 5)), (None, 10), *_rep(5, (float, 10))]
    )

    if iread != 1:
        raise RuntimeError(f"IREAD=1 expected, got {iread}")

    return {
        "t4max": t4max,
        "t4cruise": t4mc,
        "t4climb": t4mcl,
        "sls_airflow": wamap,
        "sfn_idle": sfnidl,
    }


def _read_table(f):
    """Read an entire table from a GASP engine deck file.
    The table data is returned as a "tidy format" array with three columns for the
    independent variables (altitude, T4/T2, and Mach number) and the final column for
    the table field (one of thrust, fuelflow, or airflow).
    """
    tab_data = None

    # table title
    title = f.readline().strip()
    # number of maps in the table
    (nmaps,) = _parse(f, [(int, 5)])
    # blank line
    f.readline()

    for i in range(nmaps):
        map_data = _read_map(f)

        # blank line following all but the last map in the table
        if i < nmaps - 1:
            f.readline()

        if tab_data is None:
            tab_data = map_data
        else:
            tab_data = np.r_[tab_data, map_data]

    return tab_data


def _rep(n, t):
    """Shorthand for ``itertools.repeat`` with the multiplier first."""
    return itertools.repeat(t, n)


def _parse(f, fmt):
    """Read a line from file ``f`` and parse it according to the given ``fmt``"""
    return _strparse(f.readline(), fmt)


def _strparse(s, fmt):
    """Parse a string into fixed-width numeric fields.
    ``fmt`` should be a list of tuples specifying (type, length) for each field in
    string ``s``. Use None for the type to skip (i.e. not yield) that field.
    """
    p = 0
    for typ, length in fmt:
        sub = s[p: p + length]
        if typ is not None:
            yield typ(sub)
        p += length


def _read_map(f):
    """Read a single map of a table from the engine deck file.
    The map data is returned in the same format as in ``read_table``, except there is a
    single altitude value per map.
    """
    # map dimensions: FORMAT(/2I5,F10.1,10X))
    npts, nline, amap = _parse(f, [*_rep(2, (int, 5)), (float, 10)])

    map_data = np.empty((npts * nline, 4))
    map_data[:, 0] = amap

    # number of points on a single line - wrapped if more than 6
    nptloc = min(6, npts)

    # point vals: FORMAT(10X,6F10.4,10X)
    x = list(_parse(f, [(None, 10), *_rep(nptloc, (float, 10))]))
    if npts > nptloc:
        # remaining vals on wrapped line
        x.extend(list(_parse(f, [(None, 10), *_rep(npts - nptloc, (float, 10))])))

    map_data[:, 2] = np.tile(x, nline)

    for j in range(nline):
        # line (y) val then z vals: FORMAT(F10.4,6F10.1,10X,/(6F10.1,10X))
        vals = list(_parse(f, [(float, 10), *_rep(nptloc, (float, 10))]))
        y = vals[0]
        z = vals[1:]
        if npts > nptloc:
            # ad remaining vals on warapped line
            z.extend(list(_parse(f, _rep(npts - nptloc, (float, 10)))))

        sl = slice(j * npts, (j + 1) * npts)
        map_data[sl, 1] = y
        map_data[sl, 3] = z

    return map_data


def _make_structured_grid(data, method="lagrange3"):
    """Generate a structured grid of unique mach/T4:T2/alt values in the deck"""
    # step size in t4/t2 ratio used in generating the structured grid
    # t2t2_step = 0.5 # original value
    t4t2_step = 0.5
    # step size in mach number used in generating the structured grid
    # mach_step = 0.02 # original value
    mach_step = 0.05

    structured_data = {}

    tt4 = data['thrust'][:, 1]
    tma = data['thrust'][:, 2]
    t4t2s = np.arange(min(tt4), max(tt4) + t4t2_step, t4t2_step)
    machs = np.arange(min(tma), max(tma) + mach_step, mach_step)

    # need t4t2 in first column, mach varies on each row
    pts = np.dstack(np.meshgrid(t4t2s, machs, indexing="ij")).reshape(-1, 2)
    npts = pts.shape[0]

    for field in ["thrust", "fuelflow", "airflow"]:
        map_data = data[field]
        all_alts = map_data[:, 0]
        alts = np.unique(all_alts)

        sizes = (alts.size, t4t2s.size, machs.size)
        vals = np.zeros(np.prod(sizes), dtype=float)
        alt_vec = np.zeros(np.prod(sizes), dtype=float)
        mach_vec = np.zeros(np.prod(sizes), dtype=float)
        t4t2_vec = np.zeros(np.prod(sizes), dtype=float)

        for i, alt in enumerate(alts):
            d = map_data[all_alts == alt]
            t4t2 = np.unique(d[:, 1])
            mach = np.unique(d[:, 2])
            f = d[:, 3].reshape(t4t2.size, mach.size)

            # would explicitly use lagrange3 here to mimic GASP, but some engine
            # decks may not have enough points per dimension
            interp = InterpND(
                method="2D-" + method, points=(t4t2, mach), values=f, extrapolate=True
            )
            sl = slice(i * npts, (i + 1) * npts)
            vals[sl] = interp.interpolate(pts)
            alt_vec[sl] = [alt] * len(pts)
            t4t2_vec[sl] = pts[:, 0]
            mach_vec[sl] = pts[:, 1]

        structured_data[field] = {
            "vals": vals,
            "alts": alt_vec,
            "t4t2s": t4t2_vec,
            "machs": mach_vec,
        }

    return structured_data


def _generate_flight_idle(data, T4T2, ref_sls_airflow, ref_sfn_idle):
    machs = np.unique(data[MACH])
    alts = np.unique(data[ALTITUDE])

    mach_list, alt_list = np.meshgrid(machs, alts)
    mach_list = mach_list.flatten()
    alt_list = alt_list.flatten()

    nn = len(mach_list)

    prob = om.Problem()

    prob.model.add_subsystem(
        Dynamic.Mission.MACH,
        om.IndepVarComp(
            Dynamic.Mission.MACH,
            mach_list,
            units='unitless'),
        promotes=['*'])

    prob.model.add_subsystem(
        Dynamic.Mission.ALTITUDE,
        om.IndepVarComp(
            Dynamic.Mission.ALTITUDE,
            alt_list,
            units='ft'),
        promotes=['*'])

    prob.model.add_subsystem(
        name='atmosphere', subsys=USatm1976Comp(
            num_nodes=nn), promotes_inputs=[
            ('h', Dynamic.Mission.ALTITUDE)], promotes_outputs=[
                ('temp', Dynamic.Mission.TEMPERATURE), ('pres', Dynamic.Mission.STATIC_PRESSURE)])

    prob.model.add_subsystem(
        name='conversion',
        subsys=AtmosCalc(
            num_nodes=nn),
        promotes_inputs=[
            Dynamic.Mission.MACH,
            Dynamic.Mission.TEMPERATURE,
            Dynamic.Mission.STATIC_PRESSURE],
        promotes_outputs=[
            't2',
            'p2'])

    prob.model.add_subsystem(
        name='flight_idle',
        subsys=CalculateIdle(
            num_nodes=nn,
            ref_sfn_idle=ref_sfn_idle,
            ref_sls_airflow=ref_sls_airflow),
        promotes_inputs=[
            't2',
            'p2',
            'pct_corr_airflow_idle',
            'sfc_idle'],
        promotes_outputs=[
            'idle_thrust',
            'idle_fuelflow'])

    prob.setup()

    prob.run_model()

    idle_thrust = prob.get_val('idle_thrust')
    idle_fuelflow = prob.get_val('idle_fuelflow')

    data[MACH] = np.append(data[MACH], mach_list)
    data[ALTITUDE] = np.append(data[ALTITUDE], alt_list)
    data[THRUST] = np.append(data[THRUST], idle_thrust)
    data[FUEL_FLOW] = np.append(data[FUEL_FLOW], idle_fuelflow)
    T4T2 = np.append(T4T2, np.zeros(nn))

    return data, T4T2


_PSLS_PSF = 2116.22  # SLS pressure in psf
_TSLS_DEGR = 518.67  # SLS temperature in deg R


class CalculateIdle(om.ExplicitComponent):
    '''
    Calculates idle conditions of a GASP engine at a specified flight condition
    Vectorized to calculate values for entire flight regime
    '''

    def initialize(self):
        self.options.declare('num_nodes', types=int)

        self.options.declare(
            'ref_sfn_idle',
            1.0,
            desc='Idle thrust-specific fuel consumption, from engine deck')
        self.options.declare(
            'ref_sls_airflow',
            1.0,
            desc='Sea-level static airflow of the reference engine')

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input(
            "t2", _TSLS_DEGR, units="degR", shape=nn, desc="Engine inlet temperature"
        )
        self.add_input(
            "p2", _PSLS_PSF, units="psf", shape=nn, desc="Engine inlet pressure"
        )
        self.add_input(
            "pct_corr_airflow_idle", 0.5, desc="Percent corrected airflow at idle"
        )
        self.add_input(
            "sfc_idle",
            1.0,
            units="lbm/h/lbf",
            desc="Thrust-specific fuel consumption at idle",
        )

        self.add_output("idle_thrust", units="lbf", shape=nn, desc="Idle thrust")
        # self.add_output(
        #     "idle_airflow", units="lbf/s", shape=nn, desc="Idle corrected airflow"
        # )
        self.add_output("idle_fuelflow", units="lbm/h", shape=nn, desc="Idle fuel flow")

    def compute(self, inputs, outputs):
        (
            t2,
            p2,
            pct_corr_airflow_idle,
            sfc_idle,
        ) = inputs.values()

        ref_sls_airflow = self.options['ref_sls_airflow']
        ref_sfn_idle = self.options['ref_sfn_idle']

        rthet2 = np.sqrt(t2 / _TSLS_DEGR)
        delta2 = p2 / _PSLS_PSF

        airflow_ref = pct_corr_airflow_idle * ref_sls_airflow  # don't un-correct
        thrust_ref = airflow_ref * delta2 / rthet2 * ref_sfn_idle
        fuelflow_ref = thrust_ref * sfc_idle

        outputs["idle_thrust"] = thrust_ref
        # outputs["idle_airflow"] = airflow_ref
        outputs["idle_fuelflow"] = fuelflow_ref


class AtmosCalc(om.ExplicitComponent):
    '''
    Calculates T2 and P2 given static temperature and pressure
    '''

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input(Dynamic.Mission.MACH, val=np.zeros(nn),
                       desc='current Mach number', units='unitless')
        self.add_input(Dynamic.Mission.TEMPERATURE, val=np.zeros(nn),
                       desc='current atmospheric temperature', units='degR')
        self.add_input(
            Dynamic.Mission.STATIC_PRESSURE,
            _PSLS_PSF,
            units="psf",
            shape=nn,
            desc="Ambient static pressure")

        self.add_output(
            "t2",
            units="degR",
            shape=nn,
            desc="Engine inlet total temperature")
        self.add_output("p2", units="psf", shape=nn, desc="Engine inlet total pressure")

    def compute(self, inputs, outputs):
        mach, T, P = inputs.values()

        gamma = 1.4
        t2 = T * (1 + 0.5 * (gamma - 1) * mach**2)
        p2 = P * (t2 / T) ** (gamma / (gamma - 1))

        outputs["t2"] = t2
        outputs["p2"] = p2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts FLOPS- or GASP-formatted '
                                     'engine decks into Aviary csv format.\nFLOPS decks '
                                     'are changed from column-delimited to csv format '
                                     'with added headers.\nGASP decks are reorganized '
                                     'into column based csv. T4 is recovered through '
                                     'calculation. Data points whose T4 exceeds T4max '
                                     'are removed.')
    parser.add_argument('input_file', type=str,
                        help='path to engine deck file to be converted')
    parser.add_argument('output_file', type=str,
                        help='path to file where new converted data will be written')
    parser.add_argument('data_format', type=EngineDeckType, choices=list(EngineDeckType),
                        help='data format used by input_file')

    args = parser.parse_args()

    EngineDeckConverter(**vars(args))
