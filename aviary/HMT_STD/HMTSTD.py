import math
import numpy as np
# from ambiance import Atmosphere
# from dymos.models.atmosphere.atmos_1976 import USatm1976Comp
import openmdao.api as om
from aviary.constants import RHO_SEA_LEVEL_ENGLISH, TSLS_DEGR
import pdb

def prop_perform():
    """
    Propeller Performance Routine based on Hamliton Standard Report NASA CR-2066
    Inputs:
        diam_prop = Prop Diameter (ft.)
        aft = Actitivty Factor per Blade (Range: 80 to 200)
        num_blade = numbr of Blades (Range: 2 to 8)
        CLI = Blade intergrated lift coeff (Range: 0.3 to 0.8)
              Code sometimes has trouble with CLI's greater than 0.5 at high J's 
              (J: advance ratio, ZJI, or adv_ratio)
        Install_loss_factor = Installation Loss Factor (Input: 0 for no Loss (or .le. 1.0) or Computed: Input -1.)
        DiamNac_DiamProp = Nacalle Diam to Prop Diam Ratio (Used only for Install_loss_factor = -1.)
        alt = Alitude (ft)
        VKTAS =  True Airspeed (knots)
        VTIP = Prop Tip Speed (ft/sec)
        SHP = Shaft Horsepower to Propeller (HP)
    Outputs: English units
        CT: Thrust Coeff (~Thrust/(Rhorat*D^4*N^2))
        XFT:compressibility tip loss factor 
        CTX: thrust coeff with compress losses
        ang_blade: 3/4 Blade Angle (deg)
        Thrust (lbf): thrust including both compressibility and installation losses
	    EFFP: Propeller Efficiency
	    Install_loss_factor: Installation Loss factor (if input -1.)
	    EFFPI: Installation efficiency
    """

    RHOsls = RHO_SEA_LEVEL_ENGLISH  # Air density is 0.002378 slugs per cubic foot at sea level
    # Setup for entering J and CP (iw = 1 for now)
    iw = 1
    # Print Control (set = 1 for more detailed output)
    kwrite = 1

    print("Input propeller data:")
    diam_prop, num_blade, aft, cli = [
        float(x) for x in input("  diam_prop (ft), num_blade, aft, cli: ").split()]
    num_blade = int(num_blade)
    if(diam_prop <= 0.):
        exit()

    DiamNac_DiamProp = float(input("Input Nacalle Diam to Prop Diam Ratio: "))
    sqa = DiamNac_DiamProp**2
    pdb.set_trace()

    alt = 0.0
    while (alt >= 0):
        alt, vktas, tipspd, SHP, Install_loss_factor = [
            float(x) for x in input("Input alt (ft), VKTAS (knots), VTIP (ft/sec), SHP (HP) & Install_loss_factor: ").split()]
        print()
        if (SHP <= 0):
            exit()
        if (Install_loss_factor == -1):
            Install_loss_factor = tloss(sqa, vktas, tipspd)

        temp0, pres0, rho0, rmu, a = atmos(alt)
        # try ambiance
        # pdb.set_trace()
        # atmos1 = Atmosphere(alt*0.3048) # feet to meter
        # rho1 = atmos1.density[0]*0.00194032033  # kg/m**3 to slugs/ft^3
        # temp1 = atmos1.temperature[0]  # kevin
        # temp1 = temp1*1.8  # R
        # try USatm1976Comp

        # ROR0 = RHOsls/rho0
        density_ratio = rho0/RHOsls
        FC = math.sqrt(TSLS_DEGR/temp0)  # 518.67 is TSLS_DEGR
        mach = .00150933 * vktas * FC  # at sea level, 1 knot = .00150933 (actually 0.00149984 Mach)
        tip_mach = tipspd*FC/1118.21948771  # 1118.21948771 is speed of sound at sea level
        adv_ratio = 5.309*vktas/tipspd
        cp = SHP*10.E10/density_ratio/(2.*tipspd**3*diam_prop**2*6966.)
        if (kwrite == 1):
            print(f"Inputs to PERFM:")
            print(f"  Prop Diameter: {diam_prop}")
            print(f"  Power Coeff = {cp}")
            print(f"  Advance Ratio = {adv_ratio}")
            print(f"  Actitivty Factor per Blade = {aft}")
            print(f"  number of blade = {num_blade}")
            print(f"  Blade intergrated lift coeff = {cli}")
            print(f"  Mach = {mach}")
            print(f"  Tip Mach = {tip_mach}")
            print(f"  Install Loss Factor = {Install_loss_factor}")

        # Call PERFM
        ct, ang_blade, xft, limit = perfm(
            iw, cp, adv_ratio, aft, num_blade, cli, mach, tip_mach, kwrite)

        ctx = ct*xft
        Thrust = ctx*tipspd**2*diam_prop**2*density_ratio/(1.515E06)*364.76*(1. - Install_loss_factor)
        EFFP = adv_ratio*ctx/cp
        EFFPI = EFFP*(1. - Install_loss_factor)

        print(f"Performance:")
        print(f"  Thrust Coeff = {ct}")
        print(f"  compressibility tip loss factor = {xft}")
        print(f"  thrust coeff with compress losses = {ctx}")
        print(f"  Thrust = {Thrust} (lbf)")
        print(f"  3/4 angle of blade = {ang_blade} (deg)")
        print(f"  Prop Eff. = {EFFP}")
        print(f"  Instll Loss factor = {Install_loss_factor}")
        print(f"  Install Eff. = {EFFPI}")
        if (limit > 0):
            print(f"  table look-up error = {limit} (if you go outside the tables.)")
        print()


def perfm(iw, cp, adv_ratio, act_fac, num_blade, cli, mach, tip_mach, kwrite):
    """
    IW=1      SHP INPUT (CP)
    IW=2      THRUST INPUT
    IW=3      REVERSE THRUST COMPUTATION
    IW=4      BLADE ANGLE INPUT
    cp        (input)  Power Coeff (~SHP/(Rhorat*diam_prop^5*tipspd^3).
    adv_ratio (input)  ADVANCE RATIO
    act_fac   (input)  Actitivty Factor per Blade (ref. 150, range 80 - 200). 
              installation lost = thrust loss due to propeller installed ahead of nacelle (verses a propeller all by itself)
    num_blade (input)  number of blade (range 2 - 8)
    cli       (input)  blade integrated lift coeff (ref. 0.5, range 0.3 - 0.8)
    mach      (input)  Mach
    tip_mach  (input)  Tip Mach
    CT        (output) Thrust Coeff (~Thrust/(Rhorat*diam_prop^4*tipspd^2))
    ang_blade (output) 3/4 Blade Angle (if IW != 4)
    XFT       (output) compressibility tip loss factor
    LIMIT     (output) a table look-up error (warning), if you go outside the tables.
    """
    CP_Angles = [
        [  # 2 blades
            [0.0158, 0.0165, .0188, .0230, .0369, .0588, .0914, .1340, .1816, .2273],
            [0.0215, 0.0459, .0829, .1305, .1906, .2554, 0.0, 0.0, 0.0, 0.0],
            [-.0149, -.0088, .0173, .0744, .1414, .2177, .3011, .3803, 0.0, 0.0],
            [-.0670, -.0385, .0285, .1304, .2376, .3536, .4674, .5535, 0.0, 0.0],
            [-.1150, -.0281, .1086, .2646, .4213, .5860, .7091, 0.0, 0.0, 0.0],
            [-.1151, 0.0070, .1436, .2910, .4345, .5744, .7142, .8506, .9870, 1.1175],
            [-.2427, 0.0782, .4242, .7770, 1.1164, 1.4443, 0.0, 0.0, 0.0, 0.0],
        ],
        [  # 4 blades
            [.0311, .0320, .0360, .0434, .0691, .1074, .1560, .2249, .3108, .4026],
            [.0380, .0800, .1494, .2364, .3486, .4760, 0.0, 0.0, 0.0, 0.0],
            [-.0228, -.0109, .0324, .1326, .2578, .399, .5664, .7227, 0.0, 0.0],
            [-.1252, -.0661, .0535, .2388, .4396, .6554, .8916, 1.0753, 0.0, 0.0],
            [-.2113, -.0480, .1993, .4901, .7884, 1.099, 1.3707, 0.0, 0.0, 0.0],
            [-.2077, .0153, .2657, .5387, .8107, 1.075, 1.3418, 1.5989, 1.8697, 2.1238],
            [-.4508, .1426, .7858, 1.448, 2.0899, 2.713, 0.0, 0.0, 0.0, 0.0],
        ],
        [  # 6 blades
            [.0450, .0461, .0511, .0602, .0943, .1475, .2138, .2969, .4015, .5237],
            [.0520, .1063, .2019, .3230, .4774, .6607, 0.0, 0.0, 0.0, 0.0],
            [-.0168, -.0085, .0457, .1774, .3520, .5506, .7833, 1.0236, 0.0, 0.0],
            [-.1678, -.0840, .0752, .3262, .6085, .9127, 1.2449, 1.5430, 0.0, 0.0],
            [-.2903, -.0603, .2746, .6803, 1.0989, 1.5353, 1.9747, 0.0, 0.0, 0.0],
            [-.2783, .0259, .3665, .7413, 1.1215, 1.4923, 1.8655, 2.2375, 2.6058, 2.9831],
            [-.6181, .1946, 1.0758, 1.9951, 2.8977, 3.7748, 0.0, 0.0, 0.0, 0.0],
        ],
        [  # 8 blades
            [.0577, .0591, .0648, .0751, .1141, .1783, .2599, .3551, .4682, .5952],
            [.0650, .1277, .2441, .3947, .5803, .8063, 0.0, 0.0, 0.0, 0.0],
            [-.0079, -.0025, .0595, .2134, .4266, .6708, .9519, 1.2706, 0.0, 0.0],
            [-.1894, -.0908, .0956, .3942, .7416, 1.1207, 1.5308, 1.9459, 0.0, 0.0],
            [-.3390, -.0632, .3350, .8315, 1.3494, 1.890, 2.4565, 0.0, 0.0, 0.0],
            [-.3267, .0404, .4520, .9088, 1.3783, 1.8424, 2.306, 2.7782, 3.2292, 3.7058],
            [-.7508, .2395, 1.315, 2.4469, 3.5711, 4.6638, 0.0, 0.0, 0.0, 0.0],
        ],
    ]
    CT_Angles = [
        [  # 2 blades
            [.0303, .0444, .0586, .0743, .1065, .1369, .1608, .1767, .1848, .1858],
            [.0205, .0691, .1141, .1529, .1785, .1860, 0.0, 0.0, 0.0, 0.0],
            [-.0976, -.0566, .0055, .0645, .1156, .1589, .1864, .1905, 0.0, 0.0],
            [-.1133, -.0624, .0111, .0772, .1329, .1776, .202, .2045, 0.0, 0.0],
            [-.1132, -.0356, .0479, .1161, .1711, .2111, .2150, 0.0, 0.0, 0.0],
            [-.0776, -.0159, .0391, .0868, .1279, .1646, .1964, .2213, .2414, .2505,],
            [-.1228, -.0221, .0633, .1309, .1858, .2314, 0.0, 0.0, 0.0, 0.0],
        ],
        [  # 4 blades
            [.0426, .0633, .0853, .1101, .1649, .2204, .2678, .3071, .3318, .3416],
            [.0318, .1116, .1909, .2650, .3241, .3423, 0.0, 0.0, 0.0, 0.0],
            [-.1761, -.0960, .0083, .1114, .2032, .2834, .3487, .3596, 0.0, 0.0],
            [-.2155, -.1129, .0188, .1420, .2401, .3231, .3850, .390, 0.0, 0.0],
            [-.2137, -.0657, .0859, .2108, .3141, .3894, .4095, 0.0, 0.0, 0.0],
            [-.1447, -.0314, .0698, .1577, .2342, .3013, .3611, .4067, .4457, .4681],
            [-.2338, -.0471, .1108, .2357, .3357, .4174, 0.0, 0.0, 0.0, 0.0],
        ],
        [  # 6 blades
            [.0488, .0732, .0999, .1301, .2005, .2731, .3398, .3982, .4427, .4648],
            [.0375, .1393, .2448, .3457, .4356, .4931, 0.0, 0.0, 0.0, 0.0],
            [-.2295, -.1240, .0087, .1443, .2687, .3808, .4739, .5256, 0.0, 0.0],
            [-.2999, -.1527, .0235, .1853, .3246, .4410, .5290, .5467, 0.0, 0.0],
            [-.3019, -.0907, .1154, .2871, .429, .5338, .5954, 0.0, 0.0, 0.0],
            [-.2012, -.0461, .0922, .2125, .3174, .4083, .4891, .5549, .6043, .6415],
            [-.3307, -.0749, .1411, .3118, .4466, .5548, 0.0, 0.0, 0.0, 0.0],
        ],
        [  # 8 blades
            [.0534, .0795, .1084, .1421, .2221, .3054, .3831, .4508, .5035, .5392],
            [.0423, .1588, .2841, .4056, .5157, .6042, 0.0, 0.0, 0.0, 0.0],
            [-.2606, -.1416, .0097, .1685, .3172, .4526, .5655, .6536, 0.0, 0.0],
            [-.3615, -.1804, .0267, .2193, .3870, .5312, .6410, .7032, 0.0, 0.0],
            [-.3674, -.1096, .1369, .3447, .5165, .6454, .7308, 0.0, 0.0, 0.0],
            [-.2473, -.0594, .1086, .2552, .3830, .4933, .5899, .6722, .7302, .7761],
            [-.4165, -.1040, .1597, .3671, .5289, .6556, 0.0, 0.0, 0.0, 0.0],
        ],
    ]
    AFCPC = np.array([
        [1.67, 1.37, 1.165, 1.0, .881, .81],
        [1.55, 1.33, 1.149, 1.0, .890, .82]
    ])
    AFCTC = np.array([
        [1.39, 1.27, 1.123, 1.0, .915, .865],
        [1.49, 1.30, 1.143, 1.0, .915, .865]
    ])
    Act_Factor_table = [80., 100., 125., 150., 175., 200.]
    Blade_angle_table = [
        [0.0, 2.0, 4.0, 6.0, 10., 14., 18., 22., 26., 30.],
        [10., 15., 20., 25., 30., 35., 0.0, 0.0, 0.0, 0.0],
        [10., 15., 20., 25., 30., 35., 40., 45., 0.0, 0.0],
        [20., 25., 30., 35., 40., 45., 50., 55., 0.0, 0.0],
        [30., 35., 40., 45., 50., 55., 60., 0.0, 0.0, 0.0],
        [45., 47.5, 50., 52.5, 55., 57.5, 60., 62.5, 65., 67.5],
        [57.5, 60., 62.5, 65., 67.5, 70., 0.0, 0.0, 0.0, 0.0],
    ]
    BLDCR = [
        [1.84, 1.775, 1.75, 1.74, 1.76, 1.78, 1.80, 1.81, 1.835, 1.85, 1.865, 1.875, 1.88, 1.88],
        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        [.585, .635, .675, .710, .738, .745, .758, .755, .705, .735, .710, .725, .725, .725],
        [.415, .460, .505, .535, .560, .575, .600, .610, .630, .630, .610, .605, .600, .600]
    ]
    BTDCR = [
        [1.58, 1.685, 1.73, 1.758, 1.777, 1.802, 1.828, 1.839, 1.848, 1.850, 1.850, 1.850, 1.850, 1.850],
        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        [.918, .874, .844, .821, .802, .781, .764, .752, .750, .750, .750, .750, .750, .750],
        [.864, .797, .758, .728, .701, .677, .652, .640, .630, .622, .620, .620, .620, .620]
    ]
    CL_List = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    CPCLI = [
        [.0114, .0294, .0491, .0698, .0913, .1486, .2110, .2802, .3589, .4443, .5368, .6255, 0.0, 0.0],
        [.016, .020, .0294, .0478, .0678, .0893, .1118, .1702, .2335, .3018, .3775, .4610, .5505, .6331],
        [0.0, .0324, .0486, .0671, .0875, .1094, .1326, .1935, .2576, .3259, .3990, .4805, .5664, .6438],
        [.029, .043, .048, .049, .0524, .0684, .0868, .1074, .1298, .1537, .2169, .3512, .5025, .6605],
        [.0510, .0743, .0891, .1074, .1281, .1509, .1753, .2407, .3083, .3775, .4496, .5265, .6065, .6826],
        [.0670, .0973, .1114, .1290, .1494, .1723, .1972, .2646, .3345, .4047, .4772, .5532, .6307, .7092]
    ]
    CPEC = [.01, .02, .03, .04, .05, .06, .08, .10, .15, .20, .25, .30, .35, .40]
    CPSTAL = [
        [.05, .12, .22, .35, .490, .650, .820, 1.01, 1.19],
        [.16, .29, .49, .75, 1.05, 1.37, 1.74, 2.13, 2.53],
        [.30, .47, .75, 1.1, 1.51, 1.96, 2.41, 2.86, 3.30],
        [.45, .71, 1.03, 1.40, 1.89, 2.45, 2.96, 3.55, 4.1]
    ]
    CTCLI = [
        [.0013, .0211, .0407, .0600, .0789, .1251, .1702, .2117, .2501, .2840, .3148, .3316, 0.00, 0.00],
        [.005, .010, .0158, .0362, .0563, .0761, .0954, .1419, .1868, .2278, .2669, .3013, .3317, .3460],
        [.00, .0083, .0297, .0507, .0713, .0916, .1114, .1585, .2032, .2456, .2834, .3191, .3487, .3626],
        [.0130, .0208, .0428, .0645, .0857, .1064, .1267, .1748, .2195, .2619, .2995, .3350, .3647, .3802],
        [.026, .0331, .0552, .0776, .0994, .1207, .1415, .1907, .2357, .2778, .3156, .3505, .3808, .3990],
        [.0365, .0449, .0672, .0899, .1125, .1344, .1556, .2061, .2517, .2937, .3315, .3656, .3963, .4186]
    ]
    CTEC = [.01, .03, .05, .07, .09, .12, .16, .20, .24, .28, .32, .36, .40, .44]
    CTSTAL = [
        [.125, .151, .172, .187, .204, .218, .233, .243, .249],
        [.268, .309, .343, .369, .387, .404, .420, .435, .451],
        [.401, .457, .497, .529, .557, .582, .605, .629, .651],
        [.496, .577, .628, .665, .695, .720, .742, .764, .785]
    ]
    INN = [10, 6, 8, 8, 7, 10, 6]
    NCLX = [12, 14, 14, 14, 14, 14]
    PFCLI = [1.68, 1.405, 1.0, .655, .442, .255, .102]
    TFCLI = [1.22, 1.105, 1.0, .882, .792, .665, .540]
    XLB = [2.0, 4.0, 6.0, 8.0]
    XPCLI = [
        [4.26, 2.285, 1.780, 1.568, 1.452, 1.300, 1.220, 1.160, 1.110, 1.085, 1.054, 1.048, 0.0, 0.0],
        [2.0, 1.88, 1.652, 1.408, 1.292, 1.228, 1.188, 1.132, 1.105, 1.08, 1.058, 1.042, 1.029, 1.022],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, .40, .52, .551, .619, .712, .775, .815, .845, .865, .891,.928, .958, .975],
        [0.00, .436, .545, .625, .682, .726, .755, .804, .835, .864, .889, .914, .935, .944],
        [0.00, .333, .436, .520, .585, .635, .670, .730, .770, .807, .835, .871, .897, .909]
    ]
    XTCLI = [
        [22.85, 2.40, 1.75, 1.529, 1.412, 1.268, 1.191, 1.158, 1.130, 1.122, 1.108, 1.108, 0.0, 0.0],
        [5.5, 2.27, 1.880, 1.40, 1.268, 1.208, 1.170, 1.110, 1.089, 1.071, 1.060, 1.054, 1.051, 1.048],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [.000, .399, .694, .787, .831, .860, .881, .908, .926, .940, .945, .951, .958, .958],
        [.000, .251, .539, .654, .719, .760, .788, .831, .865, .885, .900, .910, .916, .916],
        [0.0, .1852, .442, .565, .635, .681, .716, .769, .809, .838, .855, .874, .881, .881]
    ]
    ZJCL = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    ZJJ = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    ZJSTAL = [0.00, 0.400, 0.800, 1.200, 1.600, 2.00, 2.4, 2.8, 3.2]
    ZMCRO = [0.928, 0.916, 0.901, 0.884, 0.865, 0.845]
    ZMCRL = [
        [.0, .151, .299, .415, .505, .578, .620, .630, .630, .630, .630],
        [.0, .146, .287, .400, .487, .556, .595, .605, .605, .605, .605],
        [.0, .140, .276, .387, .469, .534, .571, .579, .579, .579, .579],
        [.0, .135, .265, .372, .452, .512, .547, .554, .554, .554, .554],
        [.0, .130, .252, .357, .434, .490, .522, .526, .526, .526, .526],
        [.0, .125, .240, .339, .416, .469, .498, .500, .500, .500, .500]
    ]
    ZMMMC = [
        1, 9, 12, .0, .02, .04, .06, .08, .10, .15, .20, .30,
        .01, .02, .04, .08, .12, .16, .20, .24, .28, .32, .36, .40,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        .979, .981, .984, .987, .990, .993, .996, 1.00, 1.00, 1.00, 1.00, 1.00,
        .944, .945, .950, .958, .966, .975, .984, .990, .996, .999, 1.00, 1.00,
        .901, .905, .912, .927, .942, .954, .964, .974, .984, .990, .900, .900,
        .862, .866, .875, .892, .909, .926, .942, .957, .970, .980, .984, .984,
        .806, .813, .825, .851, .877, .904, .924, .939, .952, .961, .971, .976,
        .675, .685, .700, .735, .777, .810, .845, .870, .890, .905, .920, .930,
        .525, .540, .565, .615, .670, .710, .745, .790, .825, .860, .880, .895,
        .225, .260, .320, .375, .430, .495, .550, .610, .660, .710, .740, .775
    ]
    limit = 0
    xft = 1.0
    ichck = 0
    AFCP = np.zeros(7)
    AFCT = np.zeros(7)
    CTT = np.zeros(7)
    CPP = np.zeros(7)
    BLL = np.zeros(7)
    BLLL = np.zeros(7)
    PXCLI = np.zeros(7)
    CTA = np.zeros(7)  # 6, why 7?
    CTA1 = np.zeros(7)  # 5, why 7?
    CTN = np.zeros(7)
    XFT1 = np.zeros(7)
    XFFT = np.zeros(6)
    CTG = np.zeros(6)
    CTG1 = np.zeros(6)
    TXCLI = np.zeros(6)
    CPG = np.zeros(6)
    CPG1 = np.zeros(6)
    CTTT = np.zeros(4)
    CPPP = np.zeros(4)
    XXXFT = np.zeros(4)
    # an adjustment for cp and CT for AF (for the 7 J's)
    for k in range(2):
        AFCP[k], limit = unint(Act_Factor_table, AFCPC[k], act_fac)
        AFCT[k], limit = unint(Act_Factor_table, AFCTC[k], act_fac)
    for k in range(2, 7):
        AFCP[k] = AFCP[1]
        AFCT[k] = AFCT[1]
    if (adv_ratio <= 0.5):
        AFCPE = 2.*adv_ratio*(AFCP[1] - AFCP[0]) + AFCP[0]
        AFCTE = 2.*adv_ratio*(AFCT[1] - AFCT[0]) + AFCT[0]
    else:
        AFCPE = AFCP[1]
        AFCTE = AFCT[1]

    # bounding J for setting up interpolation
    if (adv_ratio <= 1.0):
        NBEG = 0
        NEND = 3
    elif (adv_ratio <= 1.5):
        NBEG = 1
        NEND = 4
    elif (adv_ratio <= 2.0 or iw == 3):
        NBEG = 2
        NEND = 5
    else:
        NBEG = 3
        NEND = 6

    # flag that given lift coeff (cli) does not fall on a node point of CL_List
    NCL_flg = 0
    ifnd = 0
    for ii in range(6):
        iz = ii
        if (abs(cli - CL_List[ii]) <= 0.0009):
            ifnd = 1
            break
    if (ifnd == 0):
        if (cli <= 0.6):
            NCLT = 0
            NCLTT = 3
        elif (cli <= 0.7):
            NCLT = 1
            NCLTT = 4
        else:
            NCLT = 2
            NCLTT = 5
    else:
        NCLT = iz
        NCLTT = iz
        NCL_flg = 1  # flag that given lift coeff (cli) falls on a node point of CL_List

    lmod = (num_blade % 2) + 1
    if (lmod == 1):
        nbb = 1
        idx_blade = int(num_blade/2.0)
        # even number of blades idx_blade = 1 if 2 blades;
        #                       idx_blade = 2 if 4 blades;
        #                       idx_blade = 3 if 6 blades;
        #                       idx_blade = 4 if 8 blades.
        idx_blade = idx_blade - 1
    else:
        nbb = 4
        # odd number of blades
        idx_blade = 0  # start from first blade

    for ibb in range(nbb):
        # nbb = 1 even number of blades. No interpolation needed
        # nbb = 4 odd number of blades. So, interpolation done
        #       using 4 sets of even J (advance ratio) interpolation
        for k in range(NBEG, NEND+1):
            if (iw == 1 or iw == 3):
                if (iw == 3):
                    CTT[k], limit = unint(ZJSTAL, CTSTAL[idx_blade], ZJJ[k])
                    CPP[k], limit = unint(ZJSTAL, CPSTAL[idx_blade], ZJJ[k])
                    ndx = INN[k]
                    BLL[k], limit = unint(
                        CP_Angles[idx_blade][k][:ndx], Blade_angle_table[k], CPP[k])
                CPE = cp*AFCP[k]
                PBL, limit = unint(CPEC, BLDCR[idx_blade], CPE)
                # PBL = number of blades correction for cp
                CPE1 = CPE*PBL*PFCLI[k]
                # PFCLI = camber factor adjust on cp for advance ratio
                NNCLT = NCLT
                for kl in range(NCLT, NCLTT+1):
                    ndx = NCLX[NNCLT]
                    PXCLI[kl], limit = unint(CPCLI[NNCLT][:ndx], XPCLI[NNCLT], CPE1)
                    if (limit == 1):
                        ichck = ichck + 1
                    if (kwrite != 1):
                        if (ichck <= 1):
                            if (limit == 1):
                                print(
                                    f"+++warning = Mach,VTMACH,J,cp,CPE =: {mach},{tip_mach},{adv_ratio},{cp},{CPE}")
                            if (kl == 4 and CPE1 < 0.049):
                                print(
                                    f"Extrapolated data is being used for CLI=.6--CPE1,PXCLI,L= , {CPE1},{PXCLI[kl]},{idx_blade}   Suggest inputting CLI=.5")
                            if (kl == 5 and CPE1 < 0.0705):
                                print(
                                    f"Extrapolated data is being used for CLI=.7--CPE1,PXCLI,L= , {CPE1},{PXCLI[kl]},{idx_blade}   Suggest inputting CLI=.5")
                            if (kl == 6 and CPE1 < 0.0915):
                                print(
                                    f"Extrapolated data is being used for CLI=.8--CPE1,PXCLI,L= , {CPE1},{PXCLI[kl]},{idx_blade}   Suggest inputting CLI=.5")
                    else:
                        if (limit == 1):
                            print(
                                f"+++warning = Mach,VTMACH,J,cp,CPE =: {mach},{tip_mach},{adv_ratio},{cp},{CPE}")
                        if (kl == 4 and CPE1 < 0.049):
                            print(
                                f"Extrapolated data is being used for CLI=.6--CPE1,PXCLI,L= , {CPE1},{PXCLI[kl]},{idx_blade}   Suggest inputting CLI=.5")
                        if (kl == 5 and CPE1 < 0.0705):
                            print(
                                f"Extrapolated data is being used for CLI=.7--CPE1,PXCLI,L= , {CPE1},{PXCLI[kl]},{idx_blade}   Suggest inputting CLI=.5")
                        if (kl == 6 and CPE1 < 0.0915):
                            print(
                                f"Extrapolated data is being used for CLI=.8--CPE1,PXCLI,L= , {CPE1},{PXCLI[kl]},{idx_blade}   Suggest inputting CLI=.5")
                    NERPT = 1
                    NNCLT = NNCLT+1
                if (NCL_flg != 1):
                    PCLI, limit = unint(CL_List[NCLT:NCLT+4], PXCLI[NCLT:NCLT+4], cli)
                else:
                    PCLI = PXCLI[NCLT]
                    # PCLI = CLI ADJUSTMENT TO cp
                CPE = CPE*PCLI
                ndx = INN[k]
                BLL[k], limit = unint(
                    CP_Angles[idx_blade][k][:ndx], Blade_angle_table[k], CPE)
                CTT[k], limit = unint(
                    Blade_angle_table[k], CT_Angles[idx_blade][k][:ndx], BLL[k])
                if (limit > 1):
                    NERPT = 2
                    print(f"ERROR IN PROP. PERF.-- NERPT={NERPT}, LIMIT={limit}")
            elif (iw == 2):
                NNCLT = NCLT
                for kl in range(NCLT, NCLTT+1):
                    CTA[0] = ct
                    CTA[1] = 1.5*ct
                    # next, compute CTA[2],...,CTA[5]
                    # do we need CTA[6]?
                    ifnd = 0
                    for kj in range(5):
                        NFTX = kj
                        CTE1 = CTA[kj]*AFCT[k]
                        TBL, limit = unint(CTEC, BTDCR[idx_blade], CTE1)
                        CTE1 = CTE1*TBL*TFCLI[k]
                        ndx = NCLX(NNCLT)
                        TXCLI[kl], limit = unint(
                            CTCLI[NNCLT][:ndx], XTCLI[NNCLT][:ndx], CTE1)
                        NERPT = 3
                        if (limit == 1):
                            print(f"ERROR IN PROP. PERF.-- NERPT={NERPT}, LIMIT={limit}")
                        if (ZJJ[k] != 0):
                            ZMCRT, limit = unint(ZJCL, ZMCRL[NNCLT], ZJJ[k])
                            DMN = mach - ZMCRT
                        else:
                            ZMCRT = ZMCRO[NNCLT]
                            DMN = tip_mach - ZMCRT
                        XFFT[kl] = 1.0
                        if (DMN > 0.0):
                            CTE2 = CTE1*TXCLI[kl]/TFCLI[k]
                            XFFT[kl], limit = biquad(ZMMMC, 1, DMN, CTE2)
                        CTA1[kj] = ct - CTA[kj]*XFFT[kl]
                        if (CTA1[kj] == 0.0 and kj == 0):
                            ifnd = 1
                            break
                        if (kj > 0):
                            if (abs(CTA1[kj]/ct) <= 0.0007):
                                ifnd = 1
                                break
                            CTA[kj+1] = -CTA1[kj-1] * \
                                (CTA[kj]-CTA[kj-1])/(CTA1[kj]-CTA1[kj-1]) + CTA[kj-1]
                    if (ifnd == 0):
                        print(
                            "Integrated design cl adjustment not working properly for ct definition")
                    CTN[kl] = CTA[NFTX]
                    NNCLT = NNCLT + 1
                if (NCL_flg != 1):
                    TCLI, limit = unint(CL_List[NCLT:NCLT+4], TXCLI[NCLT:NCLT+4], cli)
                    XFT1[k], limit = unint(CL_List[NCLT:NCLT+4], XFFT[NCLT:NCLT+4], cli)
                    CTT[k], limit = unint(CL_List[NCLT:NCLT+4], CTN[NCLT:NCLT+4], cli)
                else:
                    TCLI = TXCLI[NCLT]
                    XFT1[k] = XFFT[NCLT]
                    CTT[k] = CTN[NCLT]
                CTE = CTT[k]*AFCT[k]*TCLI
                ndx = INN[k]
                BLL[k], limit = unint(
                    CT_Angles[idx_blade][k][:ndx], Blade_angle_table[k][:ndx], CTE)
                CPP[k], limit = unint(
                    Blade_angle_table[k][:ndx], CP_Angles[idx_blade][k][:ndx], BLL[k])
                if (limit > 0):
                    NERPT = 4
                    print(f"ERROR IN PROP. PERF.-- NERPT={NERPT}, LIMIT={limit}")
            elif (iw == 4):
                ndx = INN[k]
                CPP[ibb], limit = unint(
                    INN[k], Blade_angle_table[k][:ndx], CP_Angles[idx_blade][k][:ndx], ang_blade)
                CTT[k], limit = unint(
                    INN[k], Blade_angle_table[k][:ndx], CT_Angles[idx_blade][k][:ndx], ang_blade)

        if (iw == 1 or iw == 3):
            # print(f"BLL: {BLL}")  # blade angle
            # print(f"CTT: {CTT}")  # trust coefficient
            pass
        elif (iw == 2):
            # print(f"BLL: {BLL}")  # blade angle
            # print(f"CPP: {CPP}")  # power coefficient
            pass
        elif (iw == 4):
            # print(f"CPP: {CPP}")  # power coefficient
            # print(f"CTT: {CTT}")  # trust coefficient
            pass

        if (iw < 4):
            BLLL[ibb], limit = unint(ZJJ[NBEG:NBEG+4], BLL[NBEG:NBEG+4], adv_ratio)
            ang_blade = BLLL[ibb]
        if (iw == 4):
            CPPP[ibb], limit = unint(ZJJ[NBEG:NBEG+4], CPP[NBEG:NBEG+4], adv_ratio)
        if (iw != 2):
            CTTT[ibb], limit = unint(ZJJ[NBEG:NBEG+4], CTT[NBEG:NBEG+4], adv_ratio)
            CTG[0] = .100
            CTG[1] = .200
            TFCLII, limit = unint(ZJJ, TFCLI, adv_ratio)
            ifnd1 = 0
            ifnd2 = 0
            for il in range(5):
                ct = CTG[il]
                CTE = CTG[il]*AFCTE
                TBL, limit = unint(CTEC, BTDCR[idx_blade], CTE)
                CTE1 = CTE*TBL*TFCLII
                NNCLT = NCLT
                for kl in range(NCLT, NCLTT+1):
                    ndx = NCLX[NNCLT]
                    TXCLI[kl], limit = unint(
                        CTCLI[NNCLT][:ndx], XTCLI[NNCLT][:ndx], CTE1)
                    NERPT = 5
                    if (limit == 1):
                        # off lower bound only.
                        print(f"ERROR IN PROP. PERF.-- NERPT={NERPT}, LIMIT={limit}, il = {il}, kl = {kl}")
                    if (adv_ratio != 0.0):
                        ZMCRT, limit = unint(ZJCL, ZMCRL[NNCLT], adv_ratio)
                        DMN = mach - ZMCRT
                    else:
                        ZMCRT = ZMCRO[NNCLT]
                        DMN = tip_mach - ZMCRT
                    XFFT[kl] = 1.0
                    if (DMN > 0.0):
                        CTE2 = CTE*TXCLI[kl]*TBL
                        XFFT[kl], limit = biquad(ZMMMC, 1, DMN, CTE2)
                    NNCLT = NNCLT + 1
                if (NCL_flg != 1):
                    TCLII, limit = unint(CL_List[NCLT:NCLT+4], TXCLI[NCLT:NCLT+4], cli)
                    xft, limit = unint(CL_List[NCLT:NCLT+4], XFFT[NCLT:NCLT+4], cli)
                else:
                    TCLII = TXCLI[NCLT]
                    xft = XFFT[NCLT]
                ct = CTG[il]
                CTE = CTG[il]*AFCTE*TCLII
                CTG1[il] = CTE - CTTT[ibb]
                if (abs(CTG1[il]/CTTT[ibb]) < 0.001):
                    ifnd1 = 1
                    break
                if (il > 0):
                    CTG[il+1] = -CTG1[il-1] * \
                        (CTG[il] - CTG[il-1])/(CTG1[il] - CTG1[il-1]) + CTG[il-1]
                    if (CTG[il+1] <= 0):
                        ifnd2 = 1
                        break

            if (ifnd1 == 0 and ifnd2 == 0):
                print(
                    f"Integrated design cl adjustment not working properly for ct definition (ibb={ibb})")
            if (ifnd1 == 0 and ifnd2 == 1):
                ct = 0.0
            CTTT[ibb] = ct
            XXXFT[ibb] = xft
        else:
            xft, limit = unint(ZJJ[NBEG][:4], XFT1[NBEG][:4], adv_ratio)
            if (xft > 1):
                xft = 1
        if (iw == 2 or iw == 3):
            CPPP[ibb], limit = unint(ZJJ[NBEG][:4], CPP[NBEG][:4], adv_ratio)
        if (iw > 1):
            CPG[0] = .150
            CPG[1] = .200
            PFCLII, limit = unint(ZJJ[NBEG][:4], PFCLI[NBEG][:4], adv_ratio)
            for il in range(5):
                cp = CPG[il]
                CPE = CPG[il]*AFCPE
                PBL, limit = unint(CPEC, BLDCR[idx_blade], CPE)
                CPE1 = CPE*PBL*PFCLII
                NNCLT = NCLT
                ifnd = 0
                for kl in range(NCLT, NCLTT+1):
                    if (cli == 0.5):
                        PXCLI[kl] = 1.0
                    else:
                        PXCLI[kl], limit = unint(
                            NCLX[NNCLT], CPCLI[NNCLT], XPCLI[NNCLT], CPE1)
                        NERPT = 6
                        if (limit == 1):
                            print(
                                f"ERROR IN PROP. PERF.-- NERPT={NERPT}, LIMIT={limit}")
                    NNCLT = NNCLT + 1
                    if (NCL_flg != 1):
                        PCLII, limit = unint(
                            CL_List[NCLT:NCLT+4], PXCLI[NCLT:NCLT+4], cli)
                    else:
                        PCLII = PXCLI[NCLT]
                    cp = CPG[il]
                    CPE = CPE*PCLII
                    CPG1[il] = CPE - CPPP[ibb]
                    if (abs(CPG1[il]/CPPP[ibb]) <= .0005):
                        ifnd = 1
                        break
                    if (il != 1):
                        CPG[il+1] = -CPG1[il-1] * \
                            (CPG[il]-CPG[il-1])/(CPG1[il]-CPG1[il-1]) + CPG[il-1]
            if (ifnd == 0):
                print("Integrated design cl adjustment not working properly for ct definition")
            CPPP[ibb] = cp
            XXXFT[ibb] = xft
        idx_blade = idx_blade + 1

    if (nbb != 1):
        # interpolation by the number of blades if odd number
        if (iw == 1 or iw == 3):
            ang_blade, limit = unint(XLB, BLLL[:4], num_blade)
            ct, limit = unint(XLB, CTTT, num_blade)
        elif (iw == 2):
            ang_blade, limit = unint(XLB, BLLL[:4], num_blade)
            cp, limit = unint(XLB, CPPP, num_blade)
        elif (iw == 4):
            ct, limit = unint(XLB, CTTT, num_blade)
            cp, limit = unint(XLB, CPPP, num_blade)
        xft, limit = unint(XLB, XXXFT, num_blade)

        if (iw == 1 or iw == 3):
            # print(f"final blade_angle = {ang_blade}")
            # print(f"final coef_thrust = {ct}")
            pass
        elif (iw == 2):
            # print(f"final blade_angle = {ang_blade}")
            # print(f"final coeff_power = {cp}")
            pass
        elif (iw == 4):
            # print(f"final coef_thrust = {ct}")
            # print(f"final coeff_power = {cp}")
            pass

    return ct, ang_blade, xft, limit


def unint(xa, ya, x):
    """
    univariate table routine with seperate arrays for x and y
    This routine interpolates over a 4 point interval using a
    variation of 3nd degree interpolation to produce a continuity
    of slope between adjacent intervals.
    """

    Lmt = 0
    n = len(xa)
    # test for off low end
    if (xa[0] > x):
        Lmt = 1  # off low end
        y = ya[0]
    elif (xa[0] == x):
        y = ya[0]  # at low end
    else:
        ifnd = 0
        idx = 0
        for i in range(1, n):
            if (xa[i] == x):
                ifnd = 1  # at a node
                idx = i
                break
            elif (xa[i] > x):
                ifnd = 2  # between (xa[i-1],xa[i])
                idx = i
                break
        if (ifnd == 0):
            idx = n
            Lmt = 2  # off high end
            y = ya[n-1]
        elif (ifnd == 1):
            y = ya[idx]
        elif (ifnd == 2):
            # jx1: the first point of four points
            if (idx == 1):
                # first interval
                jx1 = 0
                ra = 1.0
            elif (idx == n-1):
                # last interval
                jx1 = n - 4
                ra = 0.0
            else:
                jx1 = idx - 2
                ra = (xa[idx] - x)/(xa[idx] - xa[idx-1])
            rb = 1.0 - ra

            # get coefficeints and results
            p1 = xa[jx1+1] - xa[jx1]
            p2 = xa[jx1+2] - xa[jx1+1]
            p3 = xa[jx1+3] - xa[jx1+2]
            p4 = p1 + p2
            p5 = p2 + p3
            d1 = x - xa[jx1]
            d2 = x - xa[jx1+1]
            d3 = x - xa[jx1+2]
            d4 = x - xa[jx1+3]
            c1 = ra/p1*d2/p4*d3
            c2 = -ra/p1*d1/p2*d3 + rb/p2*d3/p5*d4
            c3 = ra/p2*d1/p4*d2 - rb/p2*d2/p3*d4
            c4 = rb/p5*d2/p3*d3
            y = ya[jx1]*c1 + ya[jx1+1]*c2 + ya[jx1+2]*c3 + ya[jx1+3]*c4

    return y, Lmt


def biquad(T, i, xi, yi):
    """
    This routine interpolates over a 4 point interval using a 
    variation of 2nd degree interpolation to produce a continuity
    of slope between adjacent intervals.

    Table set up:
    T(i)   = table number
    T(i+1) = number of x values in xi array
    T(i+2) = number of y values in yi array
    T(i+3) = values of x in ascending order
    """
    nx = int(T[i])
    ny = int(T[i+1])
    j1 = int(i + 2)
    j2 = j1 + nx - 1
    x = xi
    # search in x sense
    # jx1 = subscript of 1st x
    # search routine - input j1, j2, xi
    #                - output ra_x, rb_x, kx, jx1
    z = 0.0
    x = xi
    kx = 0
    ifnd_x = 0
    jn = 0
    xc = [0, 0, 0, 0]
    for j in range(j1, j2+1):
        if (T[j] >= x):
            ifnd_x = 1
            jn = j
            break
    if (ifnd_x == 0):
        # off high end
        x = T[j2]
        kx = 2
        # the last 4 points and curve B
        jx1 = j2 - 3
        ra_x = 0.0
    else:
        # test for -- off low end, first interval, other
        if (jn < j1+1):
            if (T[jn] != x):
                kx = 1
                x = T[j1]
        if (jn <= j1+1):
            jx1 = j1
            ra_x = 1.0
        else:
            # test for last interval
            if (j == j2):
                jx1 = j2 - 3
                ra_x = 0.0
            else:
                jx1 = jn - 2
                ra_x = (T[jn] - x)/(T[jn] - T[jn-1])
        rb_x = 1. - ra_x

        # return here from search of x
        lmt = kx
        jx = jx1
        # The following code puts x values in xc blocks
        for j in range(4):
            xc[j] = T[jx1+j]
        # get coeff. in x sense
        # coefficient routine - input x,x1,x2,x3,x4,ra_x,rb_x
        p1 = xc[1] - xc[0]
        p2 = xc[2] - xc[1]
        p3 = xc[3] - xc[2]
        p4 = p1 + p2
        p5 = p2 + p3
        d1 = x - xc[0]
        d2 = x - xc[1]
        d3 = x - xc[2]
        d4 = x - xc[3]
        cx1 = ra_x/p1*d2/p4*d3
        cx2 = -ra_x/p1*d1/p2*d3 + rb_x/p2*d3/p5*d4
        cx3 = ra_x/p2*d1/p4*d2 - rb_x/p2*d2/p3*d4
        cx4 = rb_x/p5*d2/p3*d3
        # return to main body

        # return here with coeff. test for univariate or bivariate
        if (ny == 0):
            z = 0.0
            jy = jx + nx
            z = cx1*T[jy] + cx2*T[jy+1] + cx3*T[jy+2] + cx4*T[jy+3]
        else:
            # vibariate table
            y = yi
            j3 = j2 + 1
            j4 = j3 + ny - 1
            # search in y sense
            # jy1 = subscript of 1st y
            # search routine - input j3,j4,y
            #                - output ra_y,rb_y,ky,,jy1
            ky = 0
            ifnd_y = 0
            for j in range(j3, j4+1):
                if (T[j] >= y):
                    ifnd_y = 1
                    break
            if (ifnd_y == 0):
                # off high end
                y = T[j4]
                ky = 2
                # use last 4 points and curve B
                jy1 = j4 - 3
                ra_y = 0.0
            else:
                # test for off low end, first interval
                if (j < j3 + 1):
                    if (T[j] != y):
                        ky = 1
                        y = T[j3]
                if (j <= j3 + 1):
                    jy1 = j3
                    ra_y = 1.0
                else:
                    # test for last interval
                    if (j == j4):
                        jy1 = j4 - 3
                        ra_y = 0.0
                    else:
                        jy1 = j - 2
                        ra_y = (T[j] - y)/(T[j] - T[j-1])
            rb_y = 1.0 - ra_y

            lmt = lmt + 3*ky
            # interpolate in y sense
            # subscript - base, num. of col., num. of y's
            jy = (j4 + 1) + (jx - i - 2)*ny + (jy1 - j3)
            yt = [0, 0, 0, 0]
            for m in range(4):
                jx = jy
                yt[m] = cx1*T[jx] + cx2*T[jx+ny] + cx3*T[jx+2*ny] + cx4*T[jx+3*ny]
                jy = jy + 1

            # the following code puts y values in yc block
            yc = [0, 0, 0, 0]
            for j in range(4):
                yc[j] = T[jy1]
                jy1 = jy1 + 1
            # get coeff. in y sense
            # coeffient routine - input y, y1, y2, y3, y4, ra_y, rb_y
            p1 = yc[1] - yc[0]
            p2 = yc[2] - yc[1]
            p3 = yc[3] - yc[2]
            p4 = p1 + p2
            p5 = p2 + p3
            d1 = y - yc[0]
            d2 = y - yc[1]
            d3 = y - yc[2]
            d4 = y - yc[3]
            cy1 = ra_y/p1*d2/p4*d3
            cy2 = -ra_y/p1*d1/p2*d3 + rb_y/p2*d3/p5*d4
            cy3 = ra_y/p2*d1/p4*d2 - rb_y/p2*d2/p3*d4
            cy4 = rb_y/p5*d2/p3*d3
            z = cy1*yt[0] + cy2*yt[1] + cy3*yt[2] + cy4*yt[3]

    return z, lmt


def tloss(sqa, vktas, tipspd):
    """
    install_loss_factor: Installation Loss Factor (output)
    """
    aje = [0., 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    asqa = [0.00, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32]
    array_blockage_factors = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
           0.992, 0.991, 0.988, 0.983, 0.976, 0.970, 0.963,
           0.986, 0.982, 0.977, 0.965, 0.953, 0.940, 0.927,
           0.979, 0.974, 0.967, 0.948, 0.929, 0.908, 0.887,
           0.972, 0.965, 0.955, 0.932, 0.905, 0.872, 0.835,
           0.964, 0.954, 0.943, 0.912, 0.876, 0.834, 0.786,
           0.955, 0.943, 0.928, 0.892, 0.848, 0.801, 0.751,
           0.948, 0.935, 0.917, 0.872, 0.820, 0.763, 0.706,
           0.940, 0.924, 0.902, 0.848, 0.790, 0.726, 0.662]

    zji = 5.309 * vktas/tipspd
    if (sqa > 0.32):
        sqa = 0.32
    equiv_adv_ratio = (1.0 - 0.254 * sqa) * zji
    if (equiv_adv_ratio > 5.0):
        equiv_adv_ratio = 5.0
    blockage_factor = biv(equiv_adv_ratio, sqa, aje, asqa, array_blockage_factors)
    install_loss_factor = 1.0 - blockage_factor

    return install_loss_factor


def atmos(alt):
    """
    routine to find atmospheric properties as a function of altitude
    REF: 1962 standard atmosphere + sutherlands law for viscosity
         1976 standard atmosphere above 180K ft.

    units: alt = ft, t = deg (R), p = psi, rho = slugs/ft^3, a = ft/sec (sos)
    """

    p0 = 14.696
    # table of altitude in kft
    ah = [0., 10., 20., 30.,
          36.089, 40., 50., 60., 70., 80., 90., 100., 110., 120.,
          130., 140., 150., 160., 170., 180., 190., 200., 210., 220., 230.,
          240., 250., 260., 270., 280., 300.2, 328., 351.1, 393.7, 492.1]
    # table of temps
    at = [518.67, 483., 447.4, 411.84,
          389.97, 389.97, 389.97, 389.97, 392.25, 397.7, 403.1,
          408.57, 418.4, 433.6, 448.8, 463.9, 479.1, 487.2, 485.2, 470.1,
          454.97, 439.88, 424.8, 409.77, 394.7, 381.6, 370.9, 360.2,
          349.49, 338.8, 336.4, 351.2, 391.73, 648., 1142.]
    # table of log pressure ratio
    alogp = [0.0000, -.16254, -.33724, -.52645,
             -0.6511, -0.7310, -0.93885, -1.1465,
             -1.3537, -1.5583, -1.76, -1.95873, -2.1542, -2.3432,
             -2.5255, -2.7016, -2.8719, -3.03735, -3.2018, -3.3695,
             -3.5424, -3.721, -3.90567, -4.09677, -4.29486, -4.5002,
             -4.71157, -4.92893, -5.15262, -5.38305, -5.85717, -6.5005,
             -6.97424, -7.6012, -8.34845]

    h = alt/1000.0
    rlogp = itrln(ah, alogp, h)
    p = pow(10.0, rlogp)*p0
    t = itrln(ah, at, h)  # absolute temperature in R
    alt = h*1000.0
    rho = 0.08393*p/t
    a = 49.02*math.sqrt(t)  # speed of sound
    rmu = 2.27E-08*(pow(t, 1.5)/(t + 198.6))  # kinematic viscosity

    return t, p, rho, rmu, a


def itrln(ax, ay, x):
    """
    linear 1-D table interpolation
    ax must be monotonically increasing
    routine will extrapolate from endpoints
    """

    nx = len(ax)
    y = 0.0

    if (x <= ax[0]):
        dx = ax[1] - ax[0]
        dy = ay[1] - ay[0]
        s = dy/dx
        y = ay[0] + s*(x - ax[0])
    elif (x > ax[nx-1]):
        m = nx - 2
        dx = ax[nx-1] - ax[m]
        dy = ay[nx-1] - ay[m]
        s = ay[nx] + s*(x - ax[nx])
    else:
        ifnd = 0
        for i in range(nx):
            k = i
            z = x - ax[i]
            if (z < 0):
                ifnd = 1
                break
        if (ifnd == 1):
            j = k-1
            dx = ax[k] - ax[j]
            dy = ay[k] - ay[j]
            s = dy/dx
            y = ay[j] + s*(x - ax[j])
        else:
            raise Exception("x is not in the right range.")

    return y


def biv(x, y, ax, ay, az1):
    """
    linear 2-D table interpolation
    ax must be monotonically increasing
    ay must be monotonically increasing
    """

    nx = len(ax)
    ny = len(ay)
    z = 0.0
    # nerr = 0

    if (x >= ax[0] and x <= ax[-1] and y >= ay[0] and y <= ay[-1]):
        jn = 0
        kn = 0
        for j in range(0, nx):
            if (x < ax[j]):
                jn = j
                fx = (x - ax[j-1])/(ax[j] - ax[j-1])
                kn = 1
                break
            elif (x == ax[j]):
                jn = j
                fx = 0
                if (j == nx-1):
                    fx = 1.0
                if (j < nx-1):
                    jn = j + 1
                break

        for k in range(0, ny):
            if (y < ay[k]):
                fy = (y - ay[k-1])/(ay[k] - ay[k-1])
                kn = k
                break
            elif (y == ay[k]):
                fy = 0.0
                if (k == ny-1):
                    fy = 1.0
                if (k < ny-1):
                    kn = k + 1
                break
            else:
                kn = k

        m1 = kn*nx + jn
        m2 = kn*nx + jn - 1
        m3 = (kn - 1)*nx + jn
        m4 = m3 - 1
        zk = az1[m2] + fx*(az1[m1] - az1[m2])
        zkm1 = az1[m4] + fx*(az1[m3] - az1[m4])
        z = zkm1 + fy*(zk - zkm1)
        #nerr = 1
    else:
        #nerr = 2
        print(
            f"x = {x}, ax[0] = {ax[0]}, ax[-1] = {ax[-1]}, y = {y}, ay[0] = {ay[0]}, ay[-1] = {ay[-1]}")
        raise Exception("x or y is not in the right range.")

    return z


if __name__ == "__main__":
    prop_perform()
