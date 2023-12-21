import numpy as np
import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, get_units
from aviary.variable_info.variables import Aircraft


class SkinFrictionDrag(om.ExplicitComponent):
    """
    Computes the total coefficient of drag due to skin friction from the skin friction
    coefficients of each component surface.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Form factor fit coefficients.
        self.F = np.array([
            4.34255, -1.14281, .171203, -.0138334, .621712e-3, .137442e-6, -.145532e-4,
            2.94206, 7.16974, 48.8876, -1403.02, 8598.76, -15834.3, 4.275])

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

        # TODO: Convert these into aviary_options entries.
        self.options.declare(
            'excrescences_drag', default=0.06,
            desc='Drag contribution of excrescences as a percentage.')

    def setup(self):
        aviary_options: AviaryValues = self.options['aviary_options']
        nn = self.options['num_nodes']

        zero_count = (0, None)
        nvtail, _ = aviary_options.get_item(Aircraft.VerticalTail.NUM_TAILS, zero_count)
        nfuse, _ = aviary_options.get_item(Aircraft.Fuselage.NUM_FUSELAGES, zero_count)
        num_engines, _ = aviary_options.get_item(Aircraft.Engine.NUM_ENGINES, zero_count)
        self.nc = nc = 2 + nvtail + nfuse + int(sum(num_engines))

        # Computed by other components in drag group.
        self.add_input('skin_friction_coeff', np.zeros((nn, nc)), units='unitless')
        self.add_input('Re', np.ones((nn, nc)), units='unitless')

        # These have been assembled from the individually-titled component variables.
        self.add_input(
            'fineness_ratios', np.ones(nc),
            desc='Vector of component fineness ratios.',
            units='unitless')
        self.add_input(
            'wetted_areas', np.ones(nc), units=get_units(Aircraft.Wing.AREA),
            desc='Vector of component wetted areas.')
        self.add_input(
            'laminar_fractions_upper', np.ones(nc), units=get_units(Aircraft.Wing.LAMINAR_FLOW_UPPER),
            desc='Vector of component upper-surface laminar-flow fractions.')
        self.add_input(
            'laminar_fractions_lower', np.ones(nc), units=get_units(Aircraft.Wing.LAMINAR_FLOW_LOWER),
            desc='Vector of component lower-surface laminar-flow fractions.')

        # Aircraft design inputs
        add_aviary_input(self, Aircraft.Wing.AREA, 0.0)

        # Output
        self.add_output(
            'skin_friction_drag_coeff', np.zeros(nn), units='unitless',
            desc='Skin friction drag coefficient.')

    def setup_partials(self):
        nn = self.options["num_nodes"]
        nc = self.nc
        n = nn * nc

        self.declare_partials(
            of='skin_friction_drag_coeff',
            wrt=[Aircraft.Wing.AREA])

        rows = np.repeat(np.arange(nn), nc)
        cols = np.tile(np.arange(nc), nn)
        self.declare_partials(
            of='skin_friction_drag_coeff',
            wrt=['fineness_ratios', 'wetted_areas',
                 'laminar_fractions_upper', 'laminar_fractions_lower'], rows=rows, cols=cols)

        cols = np.arange(n)
        self.declare_partials(
            of='skin_friction_drag_coeff',
            wrt=['skin_friction_coeff', 'Re'], rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        nc = self.nc
        aviary_options: AviaryValues = self.options['aviary_options']

        cf = inputs['skin_friction_coeff']
        Re = inputs['Re']

        fineness = inputs['fineness_ratios']
        wetted_area = inputs['wetted_areas']
        lam_up = inputs['laminar_fractions_upper']
        lam_low = inputs['laminar_fractions_lower']
        mission_wing_area = inputs[Aircraft.Wing.AREA]

        laminar_flow = np.any(lam_up > 0.0) or np.any(lam_low > 0.0)

        if laminar_flow:
            laminar_upper = _calc_laminar_flow(lam_up)
            laminar_lower = _calc_laminar_flow(lam_low)
            cf = cf - 0.5 * (cf - 1.328 / np.sqrt(Re)) * \
                (laminar_lower + laminar_upper)

        form_factor = np.empty(nc, dtype=cf.dtype)

        F = self.F

        # Form factor for bodies.
        idx_body = np.where(fineness > 0.5)[0]
        fine = fineness[idx_body]

        # Note: this equation is implemented exactly as it is in FLOPS. Terms 5 and 6 in the
        # Horner expansion seem to be out of order (cf. F[5] + fine * F[6]), and the origin
        # of this equation is not clear.
        # However, if you swap the terms, you end up with negative skin friction coef.
        form_factor[idx_body] = F[0] + fine * \
            (F[1] + fine * (F[2] + fine * (F[3] + fine * (F[4] + fine * (F[5] * fine + F[6])))))

        idx_max = np.where(fineness >= 20.0)
        form_factor[idx_max] = 1.0

        # Form factor for surfaces.
        idx_surf = np.where(fineness <= 0.5)[0]
        fine = fineness[idx_surf]
        airfoil = aviary_options.get_val(Aircraft.Wing.AIRFOIL_TECHNOLOGY)

        FF1 = 1.0 + fine * (F[7] + fine * (F[8] + fine *
                            (F[9] + fine * (F[10] + fine * (F[11] + fine * F[12])))))
        FF2 = 1.0 + fine * self.F[13]

        form_factor[idx_surf] = FF1 * (2.0 - airfoil) + FF2 * (airfoil - 1.0)

        CDF = np.einsum('j,ij,j->i', wetted_area, cf, form_factor) * \
            (1.0 / mission_wing_area)

        # Add drag for excrescences.

        # TODO - Per component not completely implemented in aviary 1.0
        # Var mission_skin_friction_drag_corrections_count is a vector over components and is added
        # to the drag.
        # This may be "dead weight" from FLOPS - D.J.

        # An additional six percent of the skin friction drag is added to for excrescences
        # (miscellaneous).
        CDF *= 1.0 + self.options['excrescences_drag']

        outputs['skin_friction_drag_coeff'] = CDF

    def compute_partials(self, inputs, partials):
        nc = self.nc
        nn = self.options["num_nodes"]
        aviary_options: AviaryValues = self.options['aviary_options']

        cf = inputs['skin_friction_coeff']
        Re = inputs['Re']

        fineness = inputs['fineness_ratios']
        wetted_area = inputs['wetted_areas']
        lam_up = inputs['laminar_fractions_upper']
        lam_low = inputs['laminar_fractions_lower']
        mission_wing_area = inputs[Aircraft.Wing.AREA]

        laminar_flow = np.any(lam_up > 0.0) or np.any(lam_low > 0.0)

        den = 1.0 / np.sqrt(Re)
        lam_lam = -0.5 * (cf - 1.328 * den)
        if laminar_flow:
            laminar_upper = _calc_laminar_flow(lam_up)
            laminar_lower = _calc_laminar_flow(lam_low)
            lam_sum = laminar_lower + laminar_upper
            lam_cf = 1.0 - 0.5 * lam_sum
            lam_Re = -0.25 * 1.328 / Re ** 1.5 * lam_sum

            cf = cf - 0.5 * (cf - 1.328 * den) * lam_sum

        form_factor = np.empty(nc, dtype=cf.dtype)
        dform_dfine = np.empty(nc, dtype=cf.dtype)

        F = self.F

        # Form factor for bodies.
        idx_body = np.where(fineness > 0.5)[0]
        fine = fineness[idx_body]
        form_factor[idx_body] = F[0] + fine * \
            (F[1] + fine * (F[2] + fine * (F[3] + fine * (F[4] + fine * (F[5] * fine + F[6])))))
        dform_dfine[idx_body] = F[1] + fine * (2.0 * F[2] + fine * (
            3.0 * F[3] + fine * (4.0 * F[4] + fine * (6.0 * F[5] * fine + 5.0 * F[6]))))

        # When pinned above max fineness, deriv is zero.
        idx_max = np.where(fineness >= 20.0)
        form_factor[idx_max] = 1.0
        dform_dfine[idx_max] = 0.0

        # Form factor for surfaces.
        idx_surf = np.where(fineness <= 0.5)[0]
        fine = fineness[idx_surf]

        airfoil = aviary_options.get_val(Aircraft.Wing.AIRFOIL_TECHNOLOGY)

        FF1 = 1.0 + fine * (
            F[7] + fine
            * (F[8] + fine * (F[9] + fine * (F[10] + fine * (F[11] + fine * F[12])))))
        FF2 = 1.0 + fine * self.F[13]
        dFF1 = F[7] + fine * (
            2.0 * F[8] + fine * (
                3.0 * F[9] + fine
                * (4.0 * F[10] + fine * (5.0 * F[11] + fine * 6.0 * F[12]))))
        dFF2 = self.F[13]

        form_factor[idx_surf] = FF1 * (2.0 - airfoil) + FF2 * (airfoil - 1.0)
        dform_dfine[idx_surf] = dFF1 * (2.0 - airfoil) + dFF2 * (airfoil - 1.0)

        den = 1.0 / mission_wing_area
        CDF = np.einsum('j,ij,j->i', wetted_area, cf, form_factor) * den

        excr = 1.0 + self.options['excrescences_drag']
        CDF *= excr
        DCDF_dwet = excr * np.einsum('ij,j->ij', cf, form_factor) * den
        DCDF_dcf = excr * wetted_area * form_factor * den
        DCDF_dform = excr * np.einsum('j,ij->ij', wetted_area, cf) * den
        DCDF_dmwa = -excr * np.einsum('j,ij,j->i', wetted_area,
                                      cf, form_factor) * den ** 2

        DCDF_dlamup = DCDF_dcf * lam_lam * _calc_laminar_flow_deriv(lam_up)
        DCDF_dlamlow = DCDF_dcf * lam_lam * _calc_laminar_flow_deriv(lam_low)

        if laminar_flow:
            DCDF_dRe = np.einsum('j,ij->ij', DCDF_dcf, lam_Re)
            DCDF_dcf *= lam_cf

        partials['skin_friction_drag_coeff', 'skin_friction_coeff'] = np.tile(
            DCDF_dcf, nn)

        if laminar_flow:
            partials['skin_friction_drag_coeff', 'Re'] = DCDF_dRe.ravel()

        partials['skin_friction_drag_coeff', 'fineness_ratios'] = np.einsum(
            'ij,j->ij', DCDF_dform, dform_dfine).ravel()

        partials['skin_friction_drag_coeff', 'wetted_areas'] = DCDF_dwet.ravel()

        partials['skin_friction_drag_coeff',
                 'laminar_fractions_upper'] = DCDF_dlamup.ravel()

        partials['skin_friction_drag_coeff',
                 'laminar_fractions_lower'] = DCDF_dlamlow.ravel()

        partials['skin_friction_drag_coeff', Aircraft.Wing.AREA] = DCDF_dmwa.ravel()


def _calc_laminar_flow(lam):
    return lam * (0.0064164 + lam * (0.48087e-4 - 0.12234e-6 * lam))


def _calc_laminar_flow_deriv(lam):
    return 0.0064164 + lam * (0.96174e-4 - 0.36702e-6 * lam)
