import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft, Dynamic


class AeroForces(om.ExplicitComponent):
    """Compute lift and drag from coefficients"""

    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input("CL", 1.0, units="unitless", shape=nn, desc="Lift coefficient")
        self.add_input("CD", 1.0, units="unitless", shape=nn, desc="Drag coefficient")
        self.add_input(Dynamic.Mission.DYNAMIC_PRESSURE, 1.0,
                       units="psf", shape=nn, desc="Dynamic pressure")

        add_aviary_input(self, Aircraft.Wing.AREA, val=1370.3)

        self.add_output(Dynamic.Mission.LIFT, units="lbf", shape=nn, desc="Lift force")
        self.add_output(Dynamic.Mission.DRAG, units="lbf", shape=nn, desc="Drag force")

    def setup_partials(self):
        nn = self.options["num_nodes"]
        arange = np.arange(nn)
        self.declare_partials(
            Dynamic.Mission.LIFT, [
                "CL", Dynamic.Mission.DYNAMIC_PRESSURE], rows=arange, cols=arange)
        self.declare_partials(Dynamic.Mission.LIFT, [Aircraft.Wing.AREA])
        self.declare_partials(
            Dynamic.Mission.DRAG, [
                "CD", Dynamic.Mission.DYNAMIC_PRESSURE], rows=arange, cols=arange)
        self.declare_partials(Dynamic.Mission.DRAG, [Aircraft.Wing.AREA])

    def compute(self, inputs, outputs):
        CL, CD, q, wing_area = inputs.values()
        outputs[Dynamic.Mission.LIFT] = q * CL * wing_area
        outputs[Dynamic.Mission.DRAG] = q * CD * wing_area

    def compute_partials(self, inputs, J):
        CL, CD, q, wing_area = inputs.values()

        J[Dynamic.Mission.LIFT, "CL"] = q * wing_area
        J[Dynamic.Mission.LIFT, Dynamic.Mission.DYNAMIC_PRESSURE] = CL * wing_area
        J[Dynamic.Mission.LIFT, Aircraft.Wing.AREA] = q * CL

        J[Dynamic.Mission.DRAG, "CD"] = q * wing_area
        J[Dynamic.Mission.DRAG, Dynamic.Mission.DYNAMIC_PRESSURE] = CD * wing_area
        J[Dynamic.Mission.DRAG, Aircraft.Wing.AREA] = q * CD


class CLFromLift(om.ExplicitComponent):
    """Compute a CL from lift, used commonly for getting CL required"""

    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int)

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("lift_req", 1, units="lbf", shape=nn, desc="Lift force")
        self.add_input(Dynamic.Mission.DYNAMIC_PRESSURE, 1.0,
                       units="psf", shape=nn, desc="Dynamic pressure")

        add_aviary_input(self, Aircraft.Wing.AREA, val=1370.3)

        self.add_output("CL", shape=nn, units='unitless', desc="Lift coefficient")

    def setup_partials(self):
        ar = np.arange(self.options["num_nodes"])
        self.declare_partials(
            "CL", ["lift_req", Dynamic.Mission.DYNAMIC_PRESSURE], rows=ar, cols=ar)
        self.declare_partials("CL", [Aircraft.Wing.AREA])

    def compute(self, inputs, outputs):
        lift_req, q, wing_area = inputs.values()
        outputs["CL"] = lift_req / (q * wing_area)

    def compute_partials(self, inputs, J):
        lift_req, q, wing_area = inputs.values()
        J["CL", "lift_req"] = 1 / (q * wing_area)
        J["CL", Dynamic.Mission.DYNAMIC_PRESSURE] = -lift_req / (q**2 * wing_area)
        J["CL", Aircraft.Wing.AREA] = -lift_req / (q * wing_area**2)


class TimeRamp(om.ExplicitComponent):
    """Ramp up or down an output over time."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_index = None

    def initialize(self):
        self.options.declare("num_inputs", default=1, types=int)
        self.options.declare("num_nodes", default=1, types=int)
        self.options.declare(
            "ramp_up",
            default=True,
            types=bool,
            desc="True to ramp up from 0, False to ramp down to 0",
        )
        self.options.declare(
            "units",
            default="unitless",
            types=str,
            desc="Units of the input and output variables",
        )

    def setup(self):
        k = self.options["num_inputs"]
        nn = self.options["num_nodes"]
        units = self.options["units"]

        self.add_input("t_init", 0.0, units="s", desc="Time at which to start the ramp")
        self.add_input("t_curr", 0.0, units="s", shape=nn, desc="Current time")
        self.add_input("duration", 1.0, units="s", desc="Time duration of the ramp")
        self.add_input(
            "x", 1.0, shape=(k, nn), units=units, desc="Input values in the 'on' condition"
        )

        self.add_output("y", shape=(k, nn), units=units,
                        desc="Input values ramped over time")

        # Declare partials, taking advantage of sparsity.

        row_col = np.arange(k * nn)
        self.declare_partials(of="y", wrt="x", rows=row_col, cols=row_col)

        col = np.tile(np.arange(nn), k)
        self.declare_partials(of="y", wrt="t_curr", rows=row_col, cols=col)

        self.declare_partials(of="y", wrt=["t_init", "duration"])

    def compute(self, inputs, outputs):
        x = inputs["x"]
        y = outputs["y"]
        t_init = inputs["t_init"]
        t_curr = inputs["t_curr"]
        duration = inputs["duration"]

        t_end = t_init + duration
        idx = np.searchsorted(t_curr.real, [t_init.real, t_end.real])
        i1 = idx[0, 0]
        i2 = idx[1, 0]
        self._last_index = (i1, i2)

        fact = 1.0 / duration
        scale = (t_curr[i1:i2] - t_init) * fact

        if self.options["ramp_up"]:
            y[:, :i1] = 0
            y[:, i2:] = x[:, i2:]
            y[:, i1:i2] = scale * x[:, i1:i2]
        else:
            y[:, :i1] = x[:, :i1]
            y[:, i2:] = 0
            y[:, i1:i2] = (1 - scale) * x[:, i1:i2]

    def compute_partials(self, inputs, J):
        nn = self.options["num_nodes"]
        k = self.options["num_inputs"]

        x = inputs["x"]
        t_init = inputs["t_init"]
        t_curr = inputs["t_curr"]
        duration = inputs["duration"]

        i1, i2 = self._last_index

        fact = 1.0 / duration
        delta_t = t_curr[i1:i2] - t_init
        scale = delta_t * fact

        d_x = np.zeros(nn)
        d_tcurr = x.copy()

        if self.options["ramp_up"]:
            d_x[i2:] = 1.0
            d_x[i1:i2] = scale

            d_tcurr[:, :i1] = 0.0
            d_tcurr[:, i2:] = 0.0
            d_tcurr[:, i1:i2] *= fact

            d_duration = x * (t_curr - t_init) * (-1.0 / duration**2)
            d_duration[:, :i1] = 0.0
            d_duration[:, i2:] = 0.0

            d_tinit = x * -fact
            d_tinit[:, :i1] = 0.0
            d_tinit[:, i2:] = 0.0

        else:
            d_x[:i1] = 1.0
            d_x[i1:i2] = 1 - scale

            d_tcurr[:, :i1] = 0.0
            d_tcurr[:, i2:] = 0.0
            d_tcurr[:, i1:i2] *= -fact

            d_duration = x * (t_curr - t_init) * (1.0 / duration**2)
            d_duration[:, :i1] = 0.0
            d_duration[:, i2:] = 0.0

            d_tinit = x * fact
            d_tinit[:, :i1] = 0.0
            d_tinit[:, i2:] = 0.0

        J["y", "x"] = np.tile(d_x, k)
        J["y", "t_curr"] = d_tcurr.ravel()
        J["y", "duration"] = d_duration.ravel()
        J["y", "t_init"] = d_tinit.ravel()


class TanhRampComp(om.ExplicitComponent):
    """ Differentiable transition from one steady state condition to another using a hyperbolic tangent.
    The hyperbolic tangent activation function is:
    r_1(t) = np.tanh(t)
    This has a response centered about t=0, and a value of (-0.996, 0.996) at (-np.pi, np.pi) (thus the duration
    is roughly 2*np.pi.
    This implementation allows that response to be shifted and stretch such that the user can specify:
    1. the initial steady state value before the ramp.
    2. the final steady state value after the ramp.
    3. the starting time of the ramp (where we assume the nominal start time is -np.pi).
    4. the duration of the ramp (where we assume the nominal duration of the ramp is 2 * np.pi.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ramps = {}

    def add_ramp(self, output_name, output_units=None, shape=(1,), initial_val=0.0,
                 final_val=1.0, t_init_val=0.0, t_duration_val=1.0):
        """
        Add a tanh ramp function with the given output name to the component.
        Parameters
        ----------
        output_name : str
            The name of the output response variable.
        output_units : str or None
            Units of the response variable.
        shape : tuple
            Shape of the variable subject to the ramp at each node.
        initial_val : float
            The initial value that the ramp asymptotically approaches backward in time.
        final_val : float
            The final value that the ramp asymptotically approaches forward in time.
        t_init_val : float
            The default value for the time at which the ramp is initiated. Note that the value asymptotically
            departs the initial value and so it nearly but not exactly the initial value at this point.
        t_duration_val : float
            The default value for the time after t_init_val at which the ramp is approximately equal to the desired
            final value. Again, the hyperbolic tangent function will never exactly equal the final value but it is
            relatively flat after this duration is expired.
        """
        self._ramps[output_name] = {'shape': shape,
                                    'units': output_units,
                                    'initial_val_name': f"{output_name}:initial_val",
                                    'initial_val': initial_val,
                                    'final_val_name': f"{output_name}:final_val",
                                    'final_val': final_val,
                                    't_init_name': f"{output_name}:t_init",
                                    't_init_val': t_init_val,
                                    't_duration_name': f"{output_name}:t_duration",
                                    't_duration_val': t_duration_val}

    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare("time_units", types=str, default='s', allow_none=True)

    def setup(self):
        nn = self.options["num_nodes"]
        ar = np.arange(nn, dtype=int)

        self.add_input("time", units=self.options["time_units"], shape=(nn,))

        for output_name, options in self._ramps.items():
            size = np.prod(options['shape'], dtype=int)
            cs = np.tile(np.arange(size, dtype=int), nn)

            self.add_output(output_name, shape=(nn,) +
                            options['shape'], units=options['units'])

            self.add_input(options['initial_val_name'], val=options['initial_val']
                           * np.ones(options['shape']), units=options['units'])
            self.add_input(options['final_val_name'], val=options['final_val']
                           * np.ones(options['shape']), units=options['units'])
            self.add_input(
                options['t_init_name'], val=options['t_init_val'], units=self.options['time_units'])
            self.add_input(
                options['t_duration_name'], val=options['t_duration_val'], units=self.options['time_units'])

            self.declare_partials(
                of=output_name, wrt=f"{output_name}:initial_val", rows=ar, cols=cs)
            self.declare_partials(
                of=output_name, wrt=f"{output_name}:final_val", rows=ar, cols=cs)

            self.declare_partials(
                of=output_name, wrt=f"{output_name}:t_init", rows=ar, cols=np.zeros(nn, dtype=int))
            self.declare_partials(of=output_name, wrt=f"{output_name}:t_duration",
                                  rows=ar, cols=np.zeros(nn, dtype=int))
            self.declare_partials(of=output_name, wrt="time", rows=ar, cols=ar)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        t = inputs["time"]

        for name, options in self._ramps.items():
            initial_val = inputs[options["initial_val_name"]]
            final_val = inputs[options["final_val_name"]]
            t_init = inputs[options["t_init_name"]]
            t_duration = inputs[options["t_duration_name"]]

            dval = final_val - initial_val
            tanh_term = np.tanh(2 * np.pi * (t - t_init) / t_duration - np.pi)
            outputs[name] = 0.5 * dval * (1+tanh_term) + initial_val

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        t = inputs["time"]

        for name, options in self._ramps.items():
            initial_val = inputs[options["initial_val_name"]]
            final_val = inputs[options["final_val_name"]]
            t_init = inputs[options["t_init_name"]]
            t_duration = inputs[options["t_duration_name"]]

            dval = final_val - initial_val
            dvald2 = 0.5 * dval

            tanh_term = np.tanh(2 * np.pi * (t - t_init) / t_duration - np.pi)
            omtanh2 = 1 - tanh_term ** 2

            dtanh_term_dt = omtanh2 * 2 * np.pi / t_duration
            dtanh_term_dtinit = -dtanh_term_dt

            dtanh_term_dtduration = omtanh2 * \
                (-2 * np.pi * (t - t_init) / t_duration ** 2)

            partials[name, "time"] = dvald2 * dtanh_term_dt
            partials[name, options["t_init_name"]] = dvald2 * dtanh_term_dtinit
            partials[name, options["t_duration_name"]] = dvald2 * dtanh_term_dtduration
            partials[name, options["initial_val_name"]] = -0.5 * (1+tanh_term) + 1.0
            partials[name, options["final_val_name"]] = 0.5 * (1+tanh_term)
