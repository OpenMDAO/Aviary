import json
import os
from unittest import skipIf

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

XDSM_PATH = os.path.abspath(os.path.join(__file__, "../../..", "xdsm"))


def skipIfMissingXDSM(filename):
    return skipIf(not os.path.isfile(os.path.join(XDSM_PATH, filename)), "`"+filename+"` does not exist")


def _load_spec(spec_path, path_prefix=XDSM_PATH):
    with open(os.path.join(path_prefix, spec_path), "rb") as f:
        spec_data = json.load(f)

    spec_data["inputs"] = set(spec_data["inputs"])
    spec_data["outputs"] = set(spec_data["outputs"])
    return spec_data


def assert_match_spec(system, spec_path, path_prefix=XDSM_PATH):

    spec_data = _load_spec(spec_path, path_prefix)

    prob = om.Problem()
    prob.model = system

    prob.setup()
    prob.final_setup()

    sys_inputs = prob.model.list_inputs(out_stream=None, prom_name=True)
    sys_outputs = prob.model.list_outputs(out_stream=None, prom_name=True)

    input_set = set([inp[1]["prom_name"] for inp in sys_inputs])
    output_set = set([out[1]["prom_name"] for out in sys_outputs])

    # this checks if there are any vars in the spec, that aren't in the model
    missing_inputs = spec_data["inputs"] - input_set
    if len(missing_inputs) > 0:
        raise ValueError(
            f"inputs {missing_inputs} are in {spec_path}, but are missing in the provided system."
        )

    # filter the inputs into connected and unconnected sets
    connect_dict = system._conn_global_abs_in2out
    unconnected_inputs = set()
    connected_inputs = set()
    for abs_name, in_data in sys_inputs:
        if abs_name in connect_dict and (not "auto_ivc" in connect_dict[abs_name]):
            connected_inputs.add(in_data["prom_name"])
        else:
            unconnected_inputs.add(in_data["prom_name"])

    # now we need to check if there are any unconnected inputs
    # in the model that aren't in the spec
    extra_inputs = unconnected_inputs - spec_data["inputs"]
    if len(extra_inputs) > 0:
        raise ValueError(
            f"unconnected inputs {extra_inputs} are in your model, but are missing as inputs in the {spec_path} spec."
        )

    # last we need to check if any of the spec inputs have
    # internal connections (e.g. from stray IVCs)
    blocked_inputs = connected_inputs.intersection(spec_data["inputs"])
    if len(blocked_inputs) > 0:
        raise ValueError(
            f"inputs {blocked_inputs} are connected inside your model, but are listed as required"
            f" inputs in the {spec_path} spec so they should be unconnected."
        )

    # check if any missing required outputs are missing from the model, based on the spec
    missing_outputs = spec_data["outputs"] - output_set
    if len(missing_outputs) > 0:
        raise ValueError(
            f"outputs {missing_outputs} are in {spec_path}, but are missing in the provided system."
        )


def assert_match_vals(
    test_case, system, spec_path, path_prefix=XDSM_PATH, tolerance=1e-5, test_names=None
):

    spec_data = _load_spec(spec_path, path_prefix)

    if test_names:
        # User passed a collection of test names to check.
        verify_data = {name: system.verify_data[name] for name in test_names}
    else:
        # if there is just one case, then modify the dict to make auto-name the single case
        try:
            # Setup the verification case model
            v_data_inputs = system.verify_data["inputs"]
            verify_data = {"v_test1": system.verify_data}
        except KeyError:  # there is already more than one, use the fist one to figure things out
            verify_data = system.verify_data

    for v_case_name, v_case in verify_data.items():
        v_data_inputs = v_case["inputs"]
        v_data_outputs = v_case["outputs"]

        for v_name in v_data_inputs.keys():
            if not v_name in spec_data["inputs"]:
                raise ValueError(
                    f'input value for "{v_name}" was given in verification data, '
                    f'but this variable is not given in "{spec_path}"'
                )

        prob = om.Problem()
        ivc = prob.model.add_subsystem("spec_inputs", om.IndepVarComp(), promotes=["*"])
        for v_name, v_val in v_data_inputs.items():
            if isinstance(v_val, (list, tuple)):
                v_unit = v_val[1]
                v_val = v_val[0]
            else:
                v_unit = None

            ivc.add_output(v_name, val=v_val, units=v_unit)
            # note, can't set units, but don't want to hack it in.
            # Unit data should be available from the system itself via the default-auto-ivc stuff

        prob.model.add_subsystem("sys", system, promotes=["*"])

        prob.setup()
        prob.final_setup()

        prob.run_model()

        for v_name, v_val in v_data_outputs.items():
            if isinstance(v_val, (list, tuple)):
                v_unit = v_val[1]
                v_val = v_val[0]
            else:
                v_unit = None

            try:
                computed_val = prob.get_val(v_name, units=v_unit)
            except KeyError:  # hacky way to work around needing inputs/vs outputs
                computed_val = prob.get_val(f"sys.{v_name}", units=v_unit)

            try:
                assert_near_equal(computed_val, v_val, tolerance=tolerance)
            except AssertionError as err:
                raise ValueError(f'in case "{v_case_name}", for {v_name}: ' + str(err))


def check_prob_outputs(prob, vals, rtol=1e-6):
    """Check multiple problem outputs and print all failures.

    testflo doesn't handle unittest subTests, so this is a way to get all the
    failures at once without needing to iteratively fix and re-run. It also prints the
    variable path so you can more easily tell which assertions failed.

    Parameters
    ----------
    prob : Problem
        Problem instance to check. Should have already been run.
    vals : dict
        Dictionary mapping variable paths (i.e. ``prob[path]`` should work) to the
        value(s) to check against.
    rtol : float or list, optional
        Maximum relative tolerance to pass the test. If given as a list, it should match
        the number of entries in `vals`.
    """
    if isinstance(rtol, float):
        rtol = len(vals) * [rtol]

    errors = []
    for (path, val), t in zip(vals.items(), rtol):
        try:
            assert_near_equal(prob[path], val, tolerance=t)
        except ValueError as e:
            errors.append(f"\n  {path}: {e}")

    if errors:
        raise ValueError("".join(errors))
