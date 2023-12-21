
import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class CompressibilityDrag(om.ExplicitComponent):
    """
    Computes compressibility drag coefficient.
    """

    def initialize(self):

        self.options.declare("num_nodes", default=6, types=int,
                             desc="Number of nodes along mission segment")

    def setup(self):
        nn = self.options["num_nodes"]

        # Simulation inputs
        self.add_input(
            Dynamic.Mission.MACH,
            shape=(nn),
            units='unitless',
            desc="Mach number")

        # Aero design inputs
        add_aviary_input(self, Mission.Design.MACH, 0.0)

        # Aircraft design inputs
        add_aviary_input(self, Aircraft.Design.BASE_AREA, 0.0)
        add_aviary_input(self, Aircraft.Wing.AREA, 0.0)
        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, 0.0)
        add_aviary_input(self, Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN, 0.0)
        add_aviary_input(self, Aircraft.Wing.SWEEP, 0.0)
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, 0.0)
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD, 0.0)
        add_aviary_input(self, Aircraft.Fuselage.CROSS_SECTION, 0.0)
        add_aviary_input(self, Aircraft.Fuselage.DIAMETER_TO_WING_SPAN, 0.0)
        add_aviary_input(self, Aircraft.Fuselage.LENGTH_TO_DIAMETER, 0.0)

        # Outputs
        self.add_output('compress_drag_coeff', shape=(nn, ), units='unitless',
                        desc="Drag coefficient due to compressibility.")

    def setup_partials(self):
        nn = self.options["num_nodes"]

        row_col = np.arange(nn)
        self.declare_partials(of='compress_drag_coeff', wrt=[Dynamic.Mission.MACH],
                              rows=row_col, cols=row_col)

        wrt2 = [Aircraft.Wing.THICKNESS_TO_CHORD, Aircraft.Wing.ASPECT_RATIO,
                Aircraft.Wing.SWEEP, Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN,
                Aircraft.Fuselage.CROSS_SECTION, Aircraft.Wing.AREA,
                Aircraft.Fuselage.LENGTH_TO_DIAMETER, Aircraft.Wing.TAPER_RATIO,
                Aircraft.Fuselage.DIAMETER_TO_WING_SPAN, Aircraft.Design.BASE_AREA,
                Mission.Design.MACH]

        self.declare_partials(of='compress_drag_coeff', wrt=wrt2)

    def compute(self, inputs, outputs):
        """
        Calculate compressibility drag.
        """

        del_mach = inputs[Dynamic.Mission.MACH] - inputs[Mission.Design.MACH]

        idx_super = np.where(del_mach > 0.05)
        idx_sub = np.where(del_mach <= 0.05)
        cdc_super = self._compute_supersonic(inputs, outputs, idx_super)
        cdc_sub = self._compute_subsonic(inputs, outputs, idx_sub)

        outputs['compress_drag_coeff'][idx_super] = cdc_super
        outputs['compress_drag_coeff'][idx_sub] = cdc_sub

    def _compute_supersonic(self, inputs, outputs, idx):
        """
        Calculate compressibility drag for supersonic speeds.
        """

        mach = inputs[Dynamic.Mission.MACH][idx]
        nn = len(mach)
        del_mach = mach - inputs[Mission.Design.MACH]
        AR = inputs[Aircraft.Wing.ASPECT_RATIO]
        TC = inputs[Aircraft.Wing.THICKNESS_TO_CHORD]
        max_camber_70 = inputs[Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN]
        sweep25 = inputs[Aircraft.Wing.SWEEP]
        wing_taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
        fuse_area = inputs[Aircraft.Fuselage.CROSS_SECTION]
        base_area = inputs[Aircraft.Design.BASE_AREA]
        wing_area = inputs[Aircraft.Wing.AREA]
        fuselage_len_to_diam_ratio = inputs[Aircraft.Fuselage.LENGTH_TO_DIAMETER]
        diam_to_wing_span_ratio = inputs[Aircraft.Fuselage.DIAMETER_TO_WING_SPAN]

        ART = AR * np.tan(sweep25 / 57.2958) \
            + (1.0 - wing_taper_ratio) / (1.0 + wing_taper_ratio)
        x = np.empty((nn, 2), dtype=mach.dtype)
        x[:, 0] = del_mach
        x[:, 1] = ART
        CD3, self.dCD3 = PCARtable.interpolate(x, compute_derivative=True)

        # Negative drag sometimes occurs due to overshoot in the table interp.
        self.clamp_CD3 = np.where(CD3 <= 0)
        CD3[self.clamp_CD3] = 0.0
        self.dCD3[self.clamp_CD3] = 0.0

        compress_drag_coeff = CD3 * (TC ** (5.0 / 3.0) * (1.0 + 0.1 * max_camber_70))

        # Contribution of fuselage.
        if fuse_area > 0.0:
            SOS = 1.0 + base_area / fuse_area
            x[:, 0] = mach
            x[:, 1] = SOS
            CD4, self.dCD4 = BSUPtable.interpolate(x, compute_derivative=True)

            # Negative drag sometimes occurs due to overshoot in the table interp.
            self.clamp_CD4 = np.where(CD4 <= 0)
            CD4[self.clamp_CD4] = 0.0
            self.dCD4[self.clamp_CD4] = 0.0

            fuselage_compress_drag_coeff = CD4 * \
                (fuse_area / wing_area * (1.0 / fuselage_len_to_diam_ratio ** 2))

            compress_drag_coeff += fuselage_compress_drag_coeff

            # Wing fuselage interference.
            idx_mach = np.where(mach >= 1.0)
            if len(idx_mach[0]) > 0:

                x[:, 1] = diam_to_wing_span_ratio
                CD5, self.dCD5 = WFITable.interpolate(
                    x[idx_mach], compute_derivative=True)

                # TODO: is this some kind of override?
                if wing_taper_ratio == 1.0:
                    wing_taper_ratio = 0.5

                int_compress_drag_coeff = CD5 * (1.0 / (1.0 - wing_taper_ratio)
                                                 / np.cos(sweep25 / 57.2958))

                compress_drag_coeff[idx_mach] += int_compress_drag_coeff

            else:
                CD5 = np.zeros((nn), dtype=mach.dtype)
                self.dCD5 = np.zeros((nn, 2), dtype=mach.dtype)

        else:
            CD4 = np.zeros((nn), dtype=mach.dtype)
            self.dCD4 = np.zeros((nn, 2), dtype=mach.dtype)
            CD5 = np.zeros((nn), dtype=mach.dtype)
            self.dCD5 = np.zeros((nn, 2), dtype=mach.dtype)

        self.CD3 = CD3
        self.CD4 = CD4
        self.CD5 = CD5
        return compress_drag_coeff

    def _compute_subsonic(self, inputs, outputs, idx):
        """
        Calculate compressibility drag for subsonic speeds.
        """

        mach = inputs[Dynamic.Mission.MACH][idx]
        nn = len(mach)
        del_mach = mach - inputs[Mission.Design.MACH]
        TC = inputs[Aircraft.Wing.THICKNESS_TO_CHORD]
        max_camber_70 = inputs[Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN]
        fuse_area = inputs[Aircraft.Fuselage.CROSS_SECTION]
        base_area = inputs[Aircraft.Design.BASE_AREA]
        wing_area = inputs[Aircraft.Wing.AREA]
        fuselage_len_to_diam_ratio = inputs[Aircraft.Fuselage.LENGTH_TO_DIAMETER]

        TOC = TC ** (2.0 / 3.0)
        x = np.empty((nn, 2), dtype=mach.dtype)
        x[:, 0] = del_mach
        x[:, 1] = TOC
        CD1, self.dCD1 = PCWtable.interpolate(x, compute_derivative=True)

        # Negative drag sometimes occurs due to overshoot in the table interp.
        self.clamp_CD1 = np.where(CD1 <= 0)
        CD1[self.clamp_CD1] = 0.0
        self.dCD1[self.clamp_CD1] = 0.0

        compress_drag_coeff = CD1 * (TC ** (5.0 / 3.0) * (1.0 + 0.1 * max_camber_70))

        # Contribution of fuselage.
        if fuse_area > 0.0:
            SOS = 1.0 + base_area / fuse_area
            x[:, 0] = mach
            x[:, 1] = SOS
            CD2, self.dCD2 = BSUBtable.interpolate(x, compute_derivative=True)

            # Negative drag sometimes occurs due to overshoot in the table interp.
            self.clamp_CD2 = np.where(CD2 <= 0)
            CD2[self.clamp_CD2] = 0.0
            self.dCD2[self.clamp_CD2] = 0.0

            fuselage_compress_drag_coeff = CD2 * \
                (fuse_area / wing_area * (1.0 / fuselage_len_to_diam_ratio ** 2))

            compress_drag_coeff += fuselage_compress_drag_coeff

        else:
            CD2 = np.zeros((nn), dtype=mach.dtype)
            self.dCD2 = np.zeros((nn, 2), dtype=mach.dtype)

        self.CD1 = CD1
        self.CD2 = CD2
        return compress_drag_coeff

    def compute_partials(self, inputs, partials):
        """
        Calculate partials of compressibility drag.

        :param inputs: _description_
        :type inputs: _type_
        :param partials: _description_
        :type partials: _type_
        """

        del_mach = inputs[Dynamic.Mission.MACH] - inputs[Mission.Design.MACH]

        idx_super = np.where(del_mach > 0.05)
        idx_sub = np.where(del_mach <= 0.05)
        self._compute_partials_supersonic(inputs, partials, idx_super)
        self._compute_partials_subsonic(inputs, partials, idx_sub)

    def _compute_partials_supersonic(self, inputs, partials, idx):

        mach = inputs[Dynamic.Mission.MACH][idx]
        nn = len(mach)
        AR = inputs[Aircraft.Wing.ASPECT_RATIO]
        TC = inputs[Aircraft.Wing.THICKNESS_TO_CHORD]
        max_camber_70 = inputs[Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN]
        sweep25 = inputs[Aircraft.Wing.SWEEP]
        wing_taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
        fuse_area = inputs[Aircraft.Fuselage.CROSS_SECTION]
        base_area = inputs[Aircraft.Design.BASE_AREA]
        wing_area = inputs[Aircraft.Wing.AREA]
        fuselage_len_to_diam_ratio = inputs[Aircraft.Fuselage.LENGTH_TO_DIAMETER]

        CD3 = self.CD3
        CD4 = self.CD4
        CD5 = self.CD5

        dCd3_dCD3 = TC ** (5.0 / 3.0) * (1.0 + max_camber_70 / 10.0)
        dCd4_dCD4 = (fuse_area / wing_area) * (1.0 / fuselage_len_to_diam_ratio ** 2)
        dCd5_dCD5 = 1.0 / (1.0 - wing_taper_ratio) / np.cos(sweep25 / 57.2958)

        ddel_mach_ddesign_Mach = -1.0
        ddel_mach_dMach = 1.0

        # Derivatives from table interpolation
        dCD3_ddel_mach = self.dCD3[:, 0]
        dCD3_dART = self.dCD3[:, 1]
        dCD4_dMach = self.dCD4[:, 0]
        dCD4_dSOS = self.dCD4[:, 1]
        dCD5_dMach = self.dCD5[:, 0]
        dCD5_ddiam_to_wing_span_ratio = self.dCD5[:, 1]

        # wrt Mach
        dCd3_dMach = dCd3_dCD3 * dCD3_ddel_mach * ddel_mach_dMach

        # wrt design_Mach
        dCd_ddesign_Mach = dCd3_dCD3 * dCD3_ddel_mach * ddel_mach_ddesign_Mach

        # wrt TC
        dCd3_dTC = (5.0 / 3.0) * CD3 * TC**(2.0 / 3.0) * (1.0 + 0.1 * max_camber_70)
        dCd_dTC = dCd3_dTC

        # wrt max_camber_70
        dCd3_dmax_camber_70 = 0.1 * CD3 * TC**(5.0 / 3.0)
        dCd_dmax_camber_70 = dCd3_dmax_camber_70

        # wrt AR
        dART_dAR = np.tan(sweep25 / 57.2958)
        dCd_dAR = dCd3_dCD3 * dCD3_dART * dART_dAR

        dART_dwing_taper_ratio = -(1 - wing_taper_ratio) / (wing_taper_ratio + 1)**2 \
            - 1.0 / (wing_taper_ratio + 1)

        dART_dsweep25 = AR * (np.tan(sweep25 / 57.2958)**2 + 1) / 57.2958
        dCd3_dsweep25 = dCd3_dCD3 * dCD3_dART * dART_dsweep25

        dCd_ddiam_to_wing_span_ratio = 0

        if fuse_area > 0.0:

            # wrt Mach
            dCd4_dMach = dCd4_dCD4 * dCD4_dMach
            dCd_dMach = dCd3_dMach + dCd4_dMach

            # wrt fuse_area
            dSOS_dfuse_area = -base_area / fuse_area**2
            dCd4_dfuse_area = CD4 / (wing_area * fuselage_len_to_diam_ratio**2)
            dCd_dfuse_area = dCd4_dfuse_area + dCd4_dCD4 * dCD4_dSOS * dSOS_dfuse_area

            # wrt base_area
            dSOS_dbase_area = 1.0 / fuse_area
            dCd_dbase_area = dCd4_dCD4 * dCD4_dSOS * dSOS_dbase_area

            # wrt wing_area
            dCd_dwing_area = -CD4 * fuse_area / \
                (wing_area * fuselage_len_to_diam_ratio)**2

            # wrt fuselage_len_to_diam_ratio
            dCd_dfuselage_len_to_diam_ratio = -2.0 * CD4 * fuse_area \
                / (wing_area * fuselage_len_to_diam_ratio**3)

            # wrt diam_to_wing_span_ratio
            dCd_ddiam_to_wing_span_ratio = np.zeros(dCd_dwing_area.shape,
                                                    dtype=dCd_dwing_area.dtype)

            # wrt wing_taper_ratio
            dCd_dwing_taper_ratio = dCd3_dCD3 * dCD3_dART * dART_dwing_taper_ratio

            # wrt SW25
            dCd_dsweep25 = dCd3_dsweep25

            # Wing fuselage interference.
            idx_mach = np.where(mach >= 1.0)
            if len(idx_mach[0]) > 0:

                dCd_dMach[idx_mach] += dCd5_dCD5 * dCD5_dMach

                dCd5_dwing_taper_ratio = CD5 / \
                    ((1.0 - wing_taper_ratio)**2 * np.cos(sweep25 / 57.2958))
                dCd_dwing_taper_ratio[idx_mach] += dCd5_dwing_taper_ratio

                # wrt diam_to_wing_span_ratio
                dCd_ddiam_to_wing_span_ratio[idx_mach] = dCd5_dCD5 * \
                    dCD5_ddiam_to_wing_span_ratio

                dCd5_dsweep25 = CD5 * np.sin(sweep25 / 57.2958) / \
                    (57.2958 * (1.0 - wing_taper_ratio) * np.cos(sweep25 / 57.2958)**2)
                dCd_dsweep25[idx_mach] += dCd5_dsweep25

        else:

            dCd_dMach = dCd3_dMach
            dCd_dfuse_area = 0.0
            dCd_dbase_area = 0.0
            dCd_dwing_area = 0.0
            dCd_dfuselage_len_to_diam_ratio = 0.0
            dCd_dwing_taper_ratio = dCd3_dCD3 * dCD3_dART * dART_dwing_taper_ratio
            dCd_dsweep25 = dCd3_dsweep25

        partials["compress_drag_coeff", Dynamic.Mission.MACH][idx] = dCd_dMach
        partials["compress_drag_coeff", Mission.Design.MACH][idx, 0] = dCd_ddesign_Mach
        partials["compress_drag_coeff", Aircraft.Wing.THICKNESS_TO_CHORD][idx, 0] = dCd_dTC
        partials["compress_drag_coeff",
                 Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN][idx, 0] = dCd_dmax_camber_70
        partials["compress_drag_coeff",
                 Aircraft.Fuselage.CROSS_SECTION][idx, 0] = dCd_dfuse_area
        partials["compress_drag_coeff",
                 Aircraft.Design.BASE_AREA][idx,
                                            0] = dCd_dbase_area
        partials["compress_drag_coeff", Aircraft.Wing.AREA][idx, 0] = dCd_dwing_area
        partials["compress_drag_coeff",
                 Aircraft.Fuselage.LENGTH_TO_DIAMETER][idx,
                                                       0] = dCd_dfuselage_len_to_diam_ratio
        partials["compress_drag_coeff",
                 Aircraft.Wing.TAPER_RATIO][idx, 0] = dCd_dwing_taper_ratio
        partials["compress_drag_coeff", Aircraft.Wing.SWEEP][idx, 0] = dCd_dsweep25
        partials["compress_drag_coeff", Aircraft.Wing.ASPECT_RATIO][idx, 0] = dCd_dAR
        partials["compress_drag_coeff",
                 Aircraft.Fuselage.DIAMETER_TO_WING_SPAN][idx,
                                                          0] = dCd_ddiam_to_wing_span_ratio

    def _compute_partials_subsonic(self, inputs, partials, idx):

        mach = inputs[Dynamic.Mission.MACH][idx]
        nn = len(mach)
        TC = inputs[Aircraft.Wing.THICKNESS_TO_CHORD]
        max_camber_70 = inputs[Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN]
        fuse_area = inputs[Aircraft.Fuselage.CROSS_SECTION]
        base_area = inputs[Aircraft.Design.BASE_AREA]
        wing_area = inputs[Aircraft.Wing.AREA]
        fuselage_len_to_diam_ratio = inputs[Aircraft.Fuselage.LENGTH_TO_DIAMETER]
        CD1 = self.CD1
        CD2 = self.CD2

        dCd1_dCD1 = TC ** (5.0 / 3.0) * (1.0 + 0.1 * max_camber_70)
        dCd2_dCD2 = fuse_area / wing_area * (1.0 / fuselage_len_to_diam_ratio ** 2)
        dTOC_dTC = (2.0 / 3.0) * TC ** (-1.0 / 3.0)

        ddel_mach_dMach = 1.0

        # Derivatives from table interpolation
        dCD1_ddel_mach = self.dCD1[:, 0]
        dCD1_dTOC = self.dCD1[:, 1]
        dCD2_dMach = self.dCD2[:, 0]
        dCD2_dSOS = self.dCD2[:, 1]

        # wrt Mach
        dCd1_dMach = dCd1_dCD1 * dCD1_ddel_mach * ddel_mach_dMach
        dCd2_dMach = dCd2_dCD2 * dCD2_dMach
        dCd_dMach = dCd1_dMach + dCd2_dMach

        # wrt design_Mach
        dCd_ddesign_Mach = -dCd1_dCD1 * dCD1_ddel_mach

        # wrt TC
        dCd1_dTC = (5.0 / 3.0) * CD1 * (1.0 + 0.1 * max_camber_70) * TC**(2.0 / 3.0)
        dCD1_dTC = dCD1_dTOC * dTOC_dTC
        dCd_dTC = dCd1_dTC + dCd1_dCD1 * dCD1_dTC

        # wrt max_camber_70
        dCd_dmax_camber_70 = CD1 * (1.0 / 10.0) * TC**(5.0 / 3.0)

        partials["compress_drag_coeff", Dynamic.Mission.MACH][idx] = dCd_dMach
        partials["compress_drag_coeff", Mission.Design.MACH][idx, 0] = dCd_ddesign_Mach
        partials["compress_drag_coeff", Aircraft.Wing.THICKNESS_TO_CHORD][idx, 0] = dCd_dTC
        partials["compress_drag_coeff",
                 Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN][idx, 0] = dCd_dmax_camber_70

        # Contribution of fuselage.
        if fuse_area > 0.0:

            dCd2_dfuse_area = CD2 / (wing_area * fuselage_len_to_diam_ratio**2)
            dSOS_dfuse_area = -base_area / fuse_area**2
            dCd_dfuse_area = dCd2_dfuse_area + dCd2_dCD2 * dCD2_dSOS * dSOS_dfuse_area

            # wrt base_area
            dSOS_dbase_area = 1.0 / fuse_area
            dCd_dbase_area = dCd2_dCD2 * dCD2_dSOS * dSOS_dbase_area

            # wrt wing_area
            dCd_dwing_area = -CD2 * fuse_area / \
                (wing_area * fuselage_len_to_diam_ratio)**2

            # wrt fuselage_len_to_diam_ratio
            dCd_dfuselage_len_to_diam_ratio = -2.0 * CD2 * fuse_area \
                / (wing_area * fuselage_len_to_diam_ratio**3)

            partials["compress_drag_coeff",
                     Aircraft.Fuselage.CROSS_SECTION][idx, 0] = dCd_dfuse_area
            partials["compress_drag_coeff",
                     Aircraft.Design.BASE_AREA][idx, 0] = dCd_dbase_area
            partials["compress_drag_coeff", Aircraft.Wing.AREA][idx, 0] = dCd_dwing_area
            partials["compress_drag_coeff",
                     Aircraft.Fuselage.LENGTH_TO_DIAMETER][idx,
                                                           0] = dCd_dfuselage_len_to_diam_ratio

        else:
            partials["compress_drag_coeff",
                     Aircraft.Fuselage.CROSS_SECTION][idx, 0] = 0.0
            partials["compress_drag_coeff", Aircraft.Design.BASE_AREA][idx, 0] = 0.0
            partials["compress_drag_coeff", Aircraft.Wing.AREA][idx, 0] = 0.0
            partials["compress_drag_coeff",
                     Aircraft.Fuselage.LENGTH_TO_DIAMETER][idx, 0] = 0.0

        partials["compress_drag_coeff", Aircraft.Wing.TAPER_RATIO][idx, 0] = 0.0
        partials["compress_drag_coeff", Aircraft.Wing.SWEEP][idx, 0] = 0.0
        partials["compress_drag_coeff", Aircraft.Wing.ASPECT_RATIO][idx, 0] = 0.0
        partials["compress_drag_coeff",
                 Aircraft.Fuselage.DIAMETER_TO_WING_SPAN][idx, 0] = 0.0


# Tables
PCW = np.array([[13007.0, .100, .120, .140, .160, .180, .220, .300],
                [-.800, .00, .00, .00, .00, .00, .00, .00],
                [-.200, .0600, .040, .020, .0200, .0100, .0080, .0020],
                [-.160, .0720, .050, .030, .0260, .0170, .0160, .0060],
                [-.120, .1000, .060, .040, .0380, .0250, .0240, .0120],
                [-.080, .1250, .080, .050, .0490, .0350, .0330, .0190],
                [-.040, .1600, .120, .080, .0680, .0540, .0470, .0300],
                [-.020, .2000, .160, .120, .1100, .0700, .0590, .0390],
                [0.000, .2800, .220, .160, .1200, .0930, .0770, .0520],
                [.010, .3400, .270, .200, .1520, .1180, .0930, .0610],
                [.020, .4400, .330, .240, .1970, .1530, .1170, .0730],
                [.030, .6400, .450, .310, .2550, .2030, .1480, .0870],
                [.040, 1.1000, .660, .410, .3250, .2700, .1870, .1030],
                [.050, 1.9000, 1.020, .560, .4000, .3500, .2350, .1270]])

BSUB = np.array([[17004.0, 1.00, 1.20, 1.40, 1.50],
                 [.2000, 0.00, 0.00, 0.00, 0.00],
                 [.5000, 0.00, 0.00, 0.00, 0.00],
                 [.7000, 0.00, 0.00, 0.00, 0.00],
                 [.7800, 0.00, 0.00, 0.00, 0.00],
                 [.8200, 0.00, 0.00, 0.150, 0.210],
                 [.8400, 0.00, 0.150, 0.200, 0.350],
                 [.8600, 0.090, 0.220, 0.400, 0.520],
                 [.8800, 0.200, 0.380, 0.610, 0.780],
                 [.9000, 0.380, 0.580, 0.910, 1.100],
                 [.9100, 0.530, 0.750, 1.100, 1.330],
                 [.9200, 0.730, 0.950, 1.300, 1.600],
                 [.9300, 0.950, 1.200, 1.650, 1.930],
                 [.9400, 1.300, 1.550, 2.050, 2.490],
                 [.9500, 1.750, 2.200, 2.900, 3.650],
                 [.9600, 2.450, 3.250, 4.500, 6.400],
                 [.9650, 3.000, 4.220, 6.300, 8.450],
                 [.9700, 3.900, 5.600, 9.500, 11.500]])

PCAR = np.array([[16009.0, 1.00, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00, 5.00, 6.00],
                 [.050, 2.400, 1.700, 1.170, .850, .730, .670, .600, .540, .520],
                 [.070, 3.100, 2.250, 1.580, 1.100, .890, .770, .700, .620, .600],
                 [.090, 3.550, 2.610, 1.880, 1.240, .990, .870, .750, .670, .650],
                 [.110, 3.850, 2.880, 2.030, 1.330, 1.070, .920, .800, .710, .680],
                 [.130, 3.970, 3.050, 2.140, 1.410, 1.120, .960, .840, .740, .710],
                 [.150, 4.000, 3.100, 2.170, 1.480, 1.160, .990, .860, .750, .720],
                 [.200, 3.900, 3.000, 2.200, 1.550, 1.200, 1.000, .860, .740, .700],
                 [.250, 3.680, 2.850, 2.160, 1.570, 1.200, 1.000, .830, .700, .650],
                 [.300, 3.430, 2.700, 2.100, 1.550, 1.170, .920, .770, .630, .580],
                 [.400, 3.030, 2.450, 1.900, 1.470, 1.100, .880, .730, .590, .530],
                 [.500, 2.750, 2.220, 1.710, 1.370, 1.020, .840, .730, .570, .520],
                 [.600, 2.490, 2.000, 1.550, 1.260, .970, .810, .740, .560, .510],
                 [.700, 2.250, 1.800, 1.410, 1.170, .910, .790, .710, .550, .510],
                 [.800, 1.990, 1.620, 1.300, 1.100, .880, .750, .700, .550, .500],
                 [.900, 1.800, 1.500, 1.200, 1.000, .840, .700, .660, .540, .500],
                 [1.000, 1.650, 1.400, 1.100, .950, .800, .700, .660, .540, .500]])

BSUP = np.array([[14006.0, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50],
                 [1.000, 24.50, 20.00, 16.20, 13.40, 11.10, 9.50],
                 [1.050, 30.70, 23.60, 20.00, 16.00, 12.90, 10.50],
                 [1.100, 33.00, 26.20, 21.50, 17.40, 14.00, 11.10],
                 [1.150, 34.30, 27.30, 22.30, 18.20, 14.80, 11.60],
                 [1.200, 34.70, 27.70, 22.50, 18.50, 15.00, 11.90],
                 [1.250, 34.50, 27.50, 22.40, 18.20, 14.90, 11.90],
                 [1.300, 33.80, 27.00, 22.00, 17.60, 14.50, 11.70],
                 [1.350, 32.90, 26.40, 21.70, 17.30, 14.20, 11.40],
                 [1.400, 32.40, 25.90, 21.40, 17.20, 14.10, 11.00],
                 [1.500, 32.00, 25.60, 21.10, 17.00, 14.10, 10.90],
                 [1.600, 32.00, 25.60, 21.00, 17.00, 14.10, 10.90],
                 [1.800, 32.00, 25.60, 21.00, 17.00, 14.20, 11.40],
                 [2.000, 32.00, 25.60, 21.00, 17.10, 14.40, 11.80],
                 [2.200, 32.00, 25.60, 21.00, 17.30, 14.60, 12.00]])

# called BINT in FLOPS.
WFI = np.array([[13010.0, .10, .120, .140, .150, .160, .170, .180, .190, .200, .220],
                [1.000, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0],
                [1.050, 0.0, 0.0, .00040, -.00030, -.00080, -.00110, -.00100, -.00040, .00030, .00180],
                [1.100, 0.0, 0.0, .00060, -.00060, -.00140, -.00180, -.00140, -.00060, .00040, .00260],
                [1.150, 0.0, 0.0, .00030, -.00080, -.00170, -.00200, -.00150, -.00060, .00040, .00240],
                [1.200, 0.0, 0.0, .00020, -.00080, -.00170, -.00180, -.00140, -.00060, .00030, .00200],
                [1.300, 0.0, 0.0, .00020, -.00060, -.00100, -.00100, -.00080, -.00050, .00010, .00120],
                [1.400, 0.0, 0.0, .00010, -.00030, -.00030, -.00030, -.00020, -.00010, .00030, .00090],
                [1.500, 0.0, 0.0, .00010, .00000, .00030, .00030, .00040, .00040, .00050, .00070],
                [1.600, 0.0, 0.0, .00000, .00040, .00050, .00090, .00090, .00080, .00070, .00050],
                [1.700, 0.0, 0.0, .00000, .00050, .00070, .00120, .00110, .00100, .00080, .00050],
                [1.800, 0.0, 0.0, .00000, .00060, .00090, .00120, .00110, .00100, .00080, .00050],
                [1.900, 0.0, 0.0, .00000, .00060, .00090, .00100, .00100, .00090, .00080, .00050],
                [2.000, 0.0, 0.0, .00000, .00050, .00090, .00110, .00100, .00090, .00070, .00050]])

PCWtable = InterpND(method='lagrange2', points=(
    PCW[1:, 0], PCW[0, 1:]), values=PCW[1:, 1:], extrapolate=True)
BSUBtable = InterpND(method='lagrange2', points=(
    BSUB[1:, 0], BSUB[0, 1:]), values=BSUB[1:, 1:], extrapolate=True)
PCARtable = InterpND(method='lagrange2', points=(
    PCAR[1:, 0], PCAR[0, 1:]), values=PCAR[1:, 1:], extrapolate=True)
BSUPtable = InterpND(method='lagrange2', points=(
    BSUP[1:, 0], BSUP[0, 1:]), values=BSUP[1:, 1:], extrapolate=True)
WFITable = InterpND(method='lagrange2', points=(
    WFI[1:, 0], WFI[0, 1:]), values=WFI[1:, 1:], extrapolate=True)
