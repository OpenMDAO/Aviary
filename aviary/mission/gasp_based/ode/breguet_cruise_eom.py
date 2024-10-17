import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.variable_info.variables import Dynamic


class RangeComp(om.ExplicitComponent):
    """
    Compute the cruise range and time for the breguet range component
    """

    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):

        nn = self.options["num_nodes"]

        self.add_input("cruise_time_initial", val=0.0, units="s",
                       desc="time at which cruise begins")
        self.add_input("cruise_distance_initial", val=0.0, units="NM",
                       desc="range reference at which cruise begins")

        self.add_input("TAS_cruise", val=0.8 * np.ones(nn),
                       units="ft/s", desc="true airspeed")
        self.add_input("mass", val=150000 * np.ones(nn), units="lbm",
                       desc="mass at each node, monotonically nonincreasing")

        self.add_input(Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
                       0.74 * np.ones(nn), units="lbm/h")

        self.add_output("cruise_time", shape=(nn,), units="s", desc="time in cruise",
                        tags=['dymos.state_source:cruise_time'])
        self.add_output("cruise_range", shape=(nn,), units="NM", desc="cruise range",
                        tags=["dymos.state_source:distance"])

    def setup_partials(self):
        nn = self.options["num_nodes"]

        # The nonzero partials of the change in range between each two nodes (dr) wrt fuel flow and mass are
        # along two diagonals. The lower diagonal contains the partials wrt the initial values of mass or fuel
        # flow across each two-node section. The upper diagonal contains the partials wrt the final values of mass
        # or fuel flow across each two-node section. For instance, for five nodes we have:
        #
        #           0  0  0  0  0
        #           i  f  0  0  0
        # ddR/dW =  0  i  f  0  0
        #           0  0  i  f  0
        #           0  0  0  i  f
        #
        # The change of range and time at the first node is zero, and it has no dependence on the mass
        # history or fuel flow setting.
        # Since dR is the difference in range between two nodes, we need to accumulate it across all provided masses
        # using np.cumsum.
        # The partial derivative of np.cumsum is just a lower triangular matrix of ones.
        # Thus, the derivative of the accumulated range (and time) will be the matrix product:
        #           1  0  0  0  0     0  0  0  0  0
        #           1  1  0  0  0     i  f  0  0  0
        # ddR/dW =  1  1  1  0  0  @  0  i  f  0  0
        #           1  1  1  1  0     0  0  i  f  0
        #           1  1  1  1  1     0  0  0  i  f
        rs, cs = np.tril_indices(nn)
        self._tril_rs, self._tril_cs = rs, cs

        self.declare_partials(
            "cruise_range", [Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL, "mass", "TAS_cruise"], rows=rs, cols=cs)
        self.declare_partials(
            "cruise_time", [Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL, "mass", "TAS_cruise"], rows=rs, cols=cs)

        self.declare_partials("cruise_range", "cruise_distance_initial", val=1.0)
        self.declare_partials("cruise_time", "cruise_time_initial", val=1.0)

        # Allocated memory so we don't have to repeatedly do it in compute_partials
        # Note: since these are only used in compute_partials we don't have to worry about them supporting
        # complex values under complex step.
        self._d_cumsum_dx = np.tri(nn)
        # Note: We could make this sparse, probably doesn't matter.
        self._scratch_nn_x_nn = np.zeros((nn, nn))

    def compute(self, inputs, outputs):
        v_x = inputs["TAS_cruise"]
        m = inputs["mass"]
        FF = -inputs[Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL]
        r0 = inputs["cruise_distance_initial"]
        t0 = inputs["cruise_time_initial"]
        r0 = r0[0]
        t0 = t0[0]

        FF_1 = FF[:-1]  # Initial fuel flow across each two-node pair
        FF_2 = FF[1:]  # Final fuel flow across each two-node pair

        # Initial weight across each two-node pair
        W1 = m[:-1] * GRAV_ENGLISH_LBM
        # Final weight across each two-node pair
        W2 = m[1:] * GRAV_ENGLISH_LBM

        vx_1 = v_x[:-1]  # Initial airspeed across each two-node pair
        vx_2 = v_x[1:]  # Final airspeed across each two-node pair
        vx_m = (vx_1 + vx_2) / 2  # Average airspeed across each two-node pair.

        breg_1 = vx_1 * W1 * 3600 / (FF_1 * 6076.1)
        breg_2 = vx_2 * W2 * 3600 / (FF_2 * 6076.1)
        bregA = (breg_1 + breg_2) / 2

        drange_cruise = bregA * np.log(1. / (1. - (W1 - W2) / W1))

        outputs["cruise_range"][0] = r0
        outputs["cruise_range"][1:] = r0 + np.cumsum(drange_cruise)
        outputs["cruise_time"][0] = t0
        outputs["cruise_time"][1:] = t0 + np.cumsum(drange_cruise) / vx_m * 6076.1

    def compute_partials(self, inputs, J):
        v_x = inputs["TAS_cruise"]
        vx_1 = v_x[:-1]  # Initial airspeed across each two-node pair
        vx_2 = v_x[1:]  # Final airspeed across each two-node pair
        vx_m = (vx_1 + vx_2) / 2  # Average airspeed across each two-node pair.

        m = inputs["mass"]
        # Initial mass across each two-node pair
        W1 = m[:-1] * GRAV_ENGLISH_LBM
        W2 = m[1:] * GRAV_ENGLISH_LBM  # Final mass across each two-node pair

        FF = -inputs[Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL]
        FF_1 = FF[:-1]  # Initial fuel flow across each two-node pair
        FF_2 = FF[1:]  # Final fuel flow across each two_node pair

        breg_1 = vx_1 * W1 * 3600 / (FF_1 * 6076.1)
        breg_2 = vx_2 * W2 * 3600 / (FF_2 * 6076.1)
        bregA = (breg_1 + breg_2) / 2
        star = np.log(1 / (1 - (W1 - W2) / W1))

        dBreg1_dVx1 = W1 * 3600 / (FF_1 * 6076.1)
        dBreg1_dW1 = vx_1 * 3600 / (FF_1 * 6076.1)
        dBreg1_dFF1 = vx_1 * W1 * 3600 / (FF_1**2 * 6076.1)

        dBreg2_dVx2 = W2 * 3600 / (FF_2 * 6076.1)
        dBreg2_dW2 = vx_2 * 3600 / (FF_2 * 6076.1)
        dBreg2_dFF2 = vx_2 * W2 * 3600 / (FF_2**2 * 6076.1)

        dStar_dW1 = 1.0 / W1
        dStar_dW2 = -1.0 / W2

        dBregA_dVx1 = dBreg1_dVx1 / 2
        dBregA_dVx2 = dBreg2_dVx2 / 2
        dBregA_dW1 = dBreg1_dW1 / 2
        dBregA_dW2 = dBreg2_dW2 / 2
        dBregA_dFF1 = dBreg1_dFF1 / 2
        dBregA_dFF2 = dBreg2_dFF2 / 2

        dRange_dVx1 = dBregA_dVx1 * star
        dRange_dVx2 = dBregA_dVx2 * star
        dRange_dW1 = dBregA_dW1 * star + bregA * dStar_dW1
        dRange_dW2 = dBregA_dW2 * star + bregA * dStar_dW2
        dRange_dFF1 = dBregA_dFF1 * star
        dRange_dFF2 = dBregA_dFF2 * star

        # Partials of cruise_range

        # WRT Fuel Flow
        np.fill_diagonal(self._scratch_nn_x_nn[1:, :-1], dRange_dFF1)
        np.fill_diagonal(self._scratch_nn_x_nn[1:, 1:], dRange_dFF2)

        J["cruise_range", Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL][...] = \
            (self._d_cumsum_dx @ self._scratch_nn_x_nn)[self._tril_rs, self._tril_cs]

        # WRT Mass: dRange_dm = dRange_dW * dW_dm
        np.fill_diagonal(self._scratch_nn_x_nn[1:, :-1],
                         dRange_dW1 * GRAV_ENGLISH_LBM)
        np.fill_diagonal(self._scratch_nn_x_nn[1:, 1:],
                         dRange_dW2 * GRAV_ENGLISH_LBM)

        J["cruise_range", "mass"][...] = \
            (self._d_cumsum_dx @ self._scratch_nn_x_nn)[self._tril_rs, self._tril_cs]

        # WRT TAS_cruise
        np.fill_diagonal(self._scratch_nn_x_nn[1:, :-1], dRange_dVx1)
        np.fill_diagonal(self._scratch_nn_x_nn[1:, 1:], dRange_dVx2)

        J["cruise_range", "TAS_cruise"][...] = \
            (self._d_cumsum_dx @ self._scratch_nn_x_nn)[self._tril_rs, self._tril_cs]

        # Partials of cruise_time

        # Here we need to multiply rows [1:] of the jacobian by (6076.1 / vx_m)
        # But the jacobian is in a flat format in row-major order. The rows associated
        # with the nonzero elements are stored in self._tril_rs.

        J["cruise_time", Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL][1:] = \
            J["cruise_range", Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL][1:] / \
            vx_m[self._tril_rs[1:] - 1] * 6076.1
        J["cruise_time", "mass"][1:] = \
            J["cruise_range", "mass"][1:] / vx_m[self._tril_rs[1:] - 1] * 6076.1

        drange_cruise = bregA * star

        f = np.cumsum(drange_cruise)[:, np.newaxis]
        df_du = self._d_cumsum_dx @ self._scratch_nn_x_nn

        np.fill_diagonal(self._scratch_nn_x_nn[1:, :-1], 0.5)
        np.fill_diagonal(self._scratch_nn_x_nn[1:, 1:], 0.5)

        g = vx_m[:, np.newaxis]
        dg_du = self._scratch_nn_x_nn

        dt_dvx = 6076.1 * ((df_du[1:, ...] * g) - (dg_du[1:, ...] * f)) / g**2

        J["cruise_time", "TAS_cruise"][1:] = \
            dt_dvx[self._tril_rs[1:] - 1, self._tril_cs[1:]]
