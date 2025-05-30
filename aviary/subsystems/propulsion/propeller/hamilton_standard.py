import math
import warnings

import numpy as np
import openmdao.api as om

from aviary.constants import RHO_SEA_LEVEL_ENGLISH
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Dynamic, Settings


def _unint(xa, ya, x):
    """
    Univariate table routine with separate arrays for x and y
    This routine interpolates over a 4 point interval using a
    variation of 3nd degree interpolation to produce a continuity
    of slope between adjacent intervals.
    """
    Lmt = 0
    n = len(xa)
    # test for off low end
    if xa[0] > x:
        Lmt = 1  # off low end
        y = ya[0]
    elif xa[0] == x:
        y = ya[0]  # at low end
    else:
        ifnd = 0
        idx = 0
        for i in range(1, n):
            if xa[i] == x:
                ifnd = 1  # at a node
                idx = i
                break
            elif xa[i] > x:
                ifnd = 2  # between (xa[i-1],xa[i])
                idx = i
                break
        if ifnd == 0:
            idx = n
            Lmt = 2  # off high end
            y = ya[n - 1]
        elif ifnd == 1:
            y = ya[idx]
        elif ifnd == 2:
            # jx1: the first point of four points
            if idx == 1:
                # first interval
                jx1 = 0
                ra = 1.0
            elif idx == n - 1:
                # last interval
                jx1 = n - 4
                ra = 0.0
            else:
                jx1 = idx - 2
                ra = (xa[idx] - x) / (xa[idx] - xa[idx - 1])
            rb = 1.0 - ra

            # get coefficients and results
            p1 = xa[jx1 + 1] - xa[jx1]
            p2 = xa[jx1 + 2] - xa[jx1 + 1]
            p3 = xa[jx1 + 3] - xa[jx1 + 2]
            p4 = p1 + p2
            p5 = p2 + p3
            d1 = x - xa[jx1]
            d2 = x - xa[jx1 + 1]
            d3 = x - xa[jx1 + 2]
            d4 = x - xa[jx1 + 3]
            c1 = ra / p1 * d2 / p4 * d3
            c2 = -ra / p1 * d1 / p2 * d3 + rb / p2 * d3 / p5 * d4
            c3 = ra / p2 * d1 / p4 * d2 - rb / p2 * d2 / p3 * d4
            c4 = rb / p5 * d2 / p3 * d3
            y = ya[jx1] * c1 + ya[jx1 + 1] * c2 + ya[jx1 + 2] * c3 + ya[jx1 + 3] * c4

            # we don't want y to be an array
            try:
                y = y[0]
            # floats/ints will give TypeError, numpy versions give IndexError
            except (TypeError, IndexError):
                pass

    return y, Lmt


def _biquad(T, i, xi, yi):
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
    lmt = 0
    nx = int(T[i])
    ny = int(T[i + 1])
    j1 = int(i + 2)
    j2 = j1 + nx - 1
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
    for j in range(j1, j2 + 1):
        if T[j] >= x:
            ifnd_x = 1
            jn = j
            break
    if ifnd_x == 0:
        # off high end
        x = T[j2]
        kx = 2
        # the last 4 points and curve B
        jx1 = j2 - 3
        ra_x = 0.0
    else:
        # test for -- off low end, first interval, other
        if jn < j1 + 1:
            if T[jn] != x:
                kx = 1
                x = T[j1]
        if jn <= j1 + 1:
            jx1 = j1
            ra_x = 1.0
        else:
            # test for last interval
            if j == j2:
                jx1 = j2 - 3
                ra_x = 0.0
            else:
                jx1 = jn - 2
                ra_x = (T[jn] - x) / (T[jn] - T[jn - 1])
        rb_x = 1.0 - ra_x

        # return here from search of x
        lmt = kx
        jx = jx1
        # The following code puts x values in xc blocks
        for j in range(4):
            xc[j] = T[jx1 + j]
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
        cx1 = ra_x / p1 * d2 / p4 * d3
        cx2 = -ra_x / p1 * d1 / p2 * d3 + rb_x / p2 * d3 / p5 * d4
        cx3 = ra_x / p2 * d1 / p4 * d2 - rb_x / p2 * d2 / p3 * d4
        cx4 = rb_x / p5 * d2 / p3 * d3
        # return to main body

        # return here with coeff. test for univariate or bivariate
        if ny == 0:
            z = 0.0
            jy = jx + nx
            z = cx1 * T[jy] + cx2 * T[jy + 1] + cx3 * T[jy + 2] + cx4 * T[jy + 3]
        else:
            # bivariate table
            y = yi
            j3 = j2 + 1
            j4 = j3 + ny - 1
            # search in y sense
            # jy1 = subscript of 1st y
            # search routine - input j3,j4,y
            #                - output ra_y,rb_y,ky,,jy1
            ky = 0
            ifnd_y = 0
            for j in range(j3, j4 + 1):
                if T[j] >= y:
                    ifnd_y = 1
                    break
            if ifnd_y == 0:
                # off high end
                y = T[j4]
                ky = 2
                # use last 4 points and curve B
                jy1 = j4 - 3
                ra_y = 0.0
            else:
                # test for off low end, first interval
                if j < j3 + 1:
                    if T[j] != y:
                        ky = 1
                        y = T[j3]
                if j <= j3 + 1:
                    jy1 = j3
                    ra_y = 1.0
                else:
                    # test for last interval
                    if j == j4:
                        jy1 = j4 - 3
                        ra_y = 0.0
                    else:
                        jy1 = j - 2
                        ra_y = (T[j] - y) / (T[j] - T[j - 1])
            rb_y = 1.0 - ra_y

            lmt = lmt + 3 * ky
            # interpolate in y sense
            # subscript - base, num. of col., num. of y's
            jy = (j4 + 1) + (jx - i - 2) * ny + (jy1 - j3)
            yt = [0, 0, 0, 0]
            for m in range(4):
                jx = jy
                yt[m] = cx1 * T[jx] + cx2 * T[jx + ny] + cx3 * T[jx + 2 * ny] + cx4 * T[jx + 3 * ny]
                jy = jy + 1

            # the following code puts y values in yc block
            yc = [0, 0, 0, 0]
            for j in range(4):
                yc[j] = T[jy1]
                jy1 = jy1 + 1
            # get coeff. in y sense
            # coefficient routine - input y, y1, y2, y3, y4, ra_y, rb_y
            p1 = yc[1] - yc[0]
            p2 = yc[2] - yc[1]
            p3 = yc[3] - yc[2]
            p4 = p1 + p2
            p5 = p2 + p3
            d1 = y - yc[0]
            d2 = y - yc[1]
            d3 = y - yc[2]
            d4 = y - yc[3]
            cy1 = ra_y / p1 * d2 / p4 * d3
            cy2 = -ra_y / p1 * d1 / p2 * d3 + rb_y / p2 * d3 / p5 * d4
            cy3 = ra_y / p2 * d1 / p4 * d2 - rb_y / p2 * d2 / p3 * d4
            cy4 = rb_y / p5 * d2 / p3 * d3
            z = cy1 * yt[0] + cy2 * yt[1] + cy3 * yt[2] + cy4 * yt[3]

    return z, lmt


# block auto-formatting of tables
# fmt: off
CP_Angle_table = np.array([
    [  # 2 blades
        [0.0158, 0.0165, .0188, .0230, .0369, .0588, .0914, .1340, .1816, .22730],  # advance_ratio = 0.0
        [0.0215, 0.0459, .0829, .1305, .1906, .2554, 0.000, 0.000, 0.000, 0.0000],  # advance_ratio = 0.5
        [-.0149, -.0088, .0173, .0744, .1414, .2177, .3011, .3803, 0.000, 0.0000],  # advance_ratio = 1.0
        [-.0670, -.0385, .0285, .1304, .2376, .3536, .4674, .5535, 0.000, 0.0000],  # advance_ratio = 1.5
        [-.1150, -.0281, .1086, .2646, .4213, .5860, .7091, 0.000, 0.000, 0.0000],  # advance_ratio = 2.0
        [-.1151, 0.0070, .1436, .2910, .4345, .5744, .7142, .8506, .9870, 1.1175],  # advance_ratio = 3.0
        [-.2427, 0.0782, .4242, .7770, 1.1164, 1.4443, 0.000, 0.000, 0.000, 0.000],  # advance_ratio = 5.0
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
])
CT_Angle_table = np.array([
    [  # 2 blades
        [.0303, .0444, .0586, .0743, .1065, .1369, .1608, .1767, 0.1848, 0.1858],
        [.0205, .0691, .1141, .1529, .1785, .1860, 0.000, 0.000, 0.0000, 0.0000],
        [-.0976, -.0566, .0055, .0645, .1156, .1589, .1864, .1905, 0.000, 0.000],
        [-.1133, -.0624, .0111, .0772, .1329, .1776, .202, .2045, 0.000, 0.0000],
        [-.1132, -.0356, .0479, .1161, .1711, .2111, .2150, 0.000, 0.000, 0.000],
        [-.0776, -.0159, .0391, .0868, .1279, .1646, .1964, .2213, .2414, .2505],
        [-.1228, -.0221, .0633, .1309, .1858, .2314, 0.000, 0.000, 0.000, 0.000],
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
])
AFCPC = np.array([
    [1.67, 1.37, 1.165, 1.0, .881, .81],
    [1.55, 1.33, 1.149, 1.0, .890, .82],
])
AFCTC = np.array([
    [1.39, 1.27, 1.123, 1.0, .915, .865],
    [1.49, 1.30, 1.143, 1.0, .915, .865],
])
Act_Factor_arr = np.array([80., 100., 125., 150., 175., 200.])
Blade_angle_table = np.array([
    [0.0, 2.0, 4.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.],  # advance_ratio = 0.0
    [10.0, 15.0, 20.0, 25.0, 30.0, 35., 0.0, 0.0, 0.0, 0.0],  # advance_ratio = 0.5
    [10.0, 15.0, 20.0, 25.0, 30.0, 35., 40., 45., 0.0, 0.0],  # advance_ratio = 1.0
    [20.0, 25.0, 30.0, 35.0, 40.0, 45., 50., 55., 0.0, 0.0],  # advance_ratio = 1.5
    [30.0, 35.0, 40.0, 45.0, 50.0, 55., 60., 0.0, 0.0, 0.0],  # advance_ratio = 2.0
    [45., 47.5, 50., 52.5, 55., 57.5, 60., 62.5, 65., 67.5],  # advance_ratio = 3.0
    [57.5, 60.0, 62.5, 65., 67.5, 70.0, 0.0, 0.0, 0.0, 0.0],  # advance_ratio = 5.0
])
BL_P_corr_table = np.array([
    [1.84, 1.775, 1.75, 1.74, 1.76, 1.78, 1.80,
        1.81, 1.835, 1.85, 1.865, 1.875, 1.88, 1.88],  # 2 blades
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
        1.00, 1.00, 1.00, 1.00, 1.000, 1.000, 1.000],  # 4 blades
    [.585, .635, .675, .710, .738, .745, .758,
        .755, .705, .735, .710, .7250, .7250, .7250],  # 6 blades
    [.415, .460, .505, .535, .560, .575, .600,
        .610, .630, .630, .610, .6050, .6000, .6000],  # 8 blades
])
BL_T_corr_table = np.array([
    [1.58, 1.685, 1.73, 1.758, 1.777, 1.802, 1.828,
        1.839, 1.848, 1.850, 1.850, 1.850, 1.850, 1.850],  # 2 blades
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
        1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],  # 4 blades
    [.918, .874, .844, .821, .802, .781, .764,
        0.752, 0.750, 0.750, 0.750, 0.750, 0.750, 0.750],  # 6 blades
    [.864, .797, .758, .728, .701, .677, .652,
        0.640, 0.630, 0.622, 0.620, 0.620, 0.620, 0.620],  # 8 blades
])
CL_arr = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
CP_CLi_table = np.array([
    [0.0114, 0.0294, .0491, .0698, .0913, .1486, .2110,
        .2802, .3589, .4443, 0.5368, 0.6255, 0.00, 0.00, 0.00],  # CLI = 0.3
    [0.016, 0.020, .0294, .0478, .0678, .0893, .1118,
        .1702, .2335, .3018, .3775, .4610, .5505, .6331, 0.00],  # CLI = 0.4
    [0.00, 0.0324, .0486, .0671, .0875, .1094, .1326,
        .1935, .2576, .3259, .3990, .4805, .5664, .6438, 0.00],  # CLI = 0.5
    [0.00, 0.029, 0.043, 0.048, 0.049, .0524, .0684, .0868,
        .1074, .1298, .1537, .2169, .3512, .5025, .6605],  # CLI = 0.6
    [0.00, .0510, .0743, .0891, .1074, .1281, .1509, .1753,
        .2407, .3083, .3775, .4496, .5265, .6065, .6826],  # CLI = 0.7
    [0.00, .0670, .0973, .1114, .1290, .1494, .1723, .1972,
        .2646, .3345, .4047, .4772, .5532, .6307, .7092],  # CLI = 0.8
])
CPEC = np.array(
    [.01, .02, .03, .04, .05, .06, .08, .10, .15, .20, .25, .30, .35, .40])
CT_CLi_table = np.array([
    [0.0013, 0.0211, .0407, .0600, .0789, .1251, .1702,
        .2117, .2501, .2840, 0.3148, 0.3316, 0.00, 0.00],  # CLI = 0.3
    [0.005, 0.010, .0158, .0362, .0563, .0761, .0954,
        .1419, .1868, .2278, .2669, .3013, .3317, .3460],  # CLI = 0.4
    [0.00, 0.0083, .0297, .0507, .0713, .0916, .1114,
        .1585, .2032, .2456, .2834, .3191, .3487, .3626],  # CLI = 0.5
    [.0130, .0208, .0428, .0645, .0857, .1064, .1267,
        .1748, .2195, .2619, .2995, .3350, .3647, .3802],  # CLI = 0.6
    [0.026, .0331, .0552, .0776, .0994, .1207, .1415,
        .1907, .2357, .2778, .3156, .3505, .3808, .3990],  # CLI = 0.7
    [.0365, .0449, .0672, .0899, .1125, .1344, .1556,
        .2061, .2517, .2937, .3315, .3656, .3963, .4186],  # CLI = 0.8
])
CTEC = np.array(
    [.01, .03, .05, .07, .09, .12, .16, .20, .24, .28, .32, .36, .40, .44])
# array length for CP_Angle_table and CT_Angle_table
ang_arr_len = np.array([10, 6, 8, 8, 7, 10, 6])
# array length for CP_CLi_table and CT_CLi_table
cli_arr_len = np.array([12, 14, 14, 15, 15, 15])
# integrated design lift coefficient adjustment factor to power coefficient
PF_CLI_arr = np.array([1.68, 1.405, 1.0, .655, .442, .255, .102])
# integrated design lift coefficient adjustment factor to thrust coefficient
TF_CLI_arr = np.array([1.22, 1.105, 1.0, .882, .792, .665, .540])
num_blades_arr = np.array([2.0, 4.0, 6.0, 8.0])
XPCLI = np.array([
    [4.26, 2.285, 1.780, 1.568, 1.452, 1.300, 1.220,
        1.160, 1.110, 1.085, 1.054, 1.048, 0.000, 0.000, 0.0],  # CL = 0.3
    [2.0, 1.88, 1.652, 1.408, 1.292, 1.228, 1.188,
        1.132, 1.105, 1.08, 1.058, 1.042, 1.029, 1.0220, 0.0],  # CL = 0.4
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 0.0],  # CL = 0.5
    [0.0, 0.065, .40, .52, .551, .619, .712, .775,
        0.815, 0.845, 0.865, 0.891, 0.928, 0.958, 0.975],  # CL = 0.6
    [0.00, 0.250, .436, .545, .625, .682, .726, .755,
        0.804, 0.835, 0.864, 0.889, 0.914, 0.935, 0.944],  # CL = 0.7
    [0.00, 0.110, .333, .436, .520, .585, .635, .670,
        0.730, 0.770, 0.807, 0.835, 0.871, 0.897, 0.909],  # CL = 0.8
])
XTCLI = np.array([
    [22.85, 2.40, 1.75, 1.529, 1.412, 1.268, 1.191,
        1.158, 1.130, 1.122, 1.108, 1.108, 0.000, 0.000],  # CL = 0.3
    [5.5, 2.27, 1.880, 1.40, 1.268, 1.208, 1.170,
        1.110, 1.089, 1.071, 1.060, 1.054, 1.051, 1.048],  # CL = 0.4
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],  # CL = 0.5
    [.295, .399, .694, .787, .831, .860, .881,
        0.908, 0.926, 0.940, 0.945, 0.951, 0.958, 0.958],  # CL = 0.6
    [.166, .251, .539, .654, .719, .760, .788,
        0.831, 0.865, 0.885, 0.900, 0.910, 0.916, 0.916],  # CL = 0.7
    [0.042, .1852, .442, .565, .635, .681, .716,
        0.769, 0.809, 0.838, 0.855, 0.874, 0.881, 0.881],  # CL = 0.8
])
advance_ratio_array2 = ([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
advance_ratio_array = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
mach_tip_corr_arr = np.array([0.928, 0.916, 0.901, 0.884, 0.865, 0.845])
mach_corr_table = np.array([  # ZMCRL
    [.0, .151, .299, .415, .505, .578, .620, .630, .630, .630, .630],  # CL = 0.3
    [.0, .146, .287, .400, .487, .556, .595, .605, .605, .605, .605],  # CL = 0.4
    [.0, .140, .276, .387, .469, .534, .571, .579, .579, .579, .579],  # CL = 0.5
    [.0, .135, .265, .372, .452, .512, .547, .554, .554, .554, .554],  # CL = 0.6
    [.0, .130, .252, .357, .434, .490, .522, .526, .526, .526, .526],  # CL = 0.7
    [.0, .125, .240, .339, .416, .469, .498, .500, .500, .500, .500],  # CL = 0.8
])
comp_mach_CT_arr = np.array([
    # table number, number of X array, number of Y array, X array
    1, 9, 12, .0, .02, .04, .06, .08, .10, .15, .20, .30,
    # Y array (CTE)
    0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40,
    # Mach
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,  # X = 0.00
    .979, .981, .984, .987, .990, .993, .996, 1.00, 1.00, 1.00, 1.00, 1.00,  # X = 0.02
    .944, .945, .950, .958, .966, .975, .984, .990, .996, .999, 1.00, 1.00,  # X = 0.04
    .901, .905, .912, .927, .942, .954, .964, .974, .984, .990, .900, .900,  # X = 0.06
    .862, .866, .875, .892, .909, .926, .942, .957, .970, .980, .984, .984,  # X = 0.08
    .806, .813, .825, .851, .877, .904, .924, .939, .952, .961, .971, .976,  # X = 0.10
    .675, .685, .700, .735, .777, .810, .845, .870, .890, .905, .920, .930,  # X = 0.15
    .525, .540, .565, .615, .670, .710, .745, .790, .825, .860, .880, .895,  # X = 0.20
    .225, .260, .320, .375, .430, .495, .550, .610, .660, .710, .740, .775,  # X = 0.30
])
# fmt: on


class PreHamiltonStandard(om.ExplicitComponent):
    """Pre-process parameters needed by HamiltonStandard component."""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Aircraft.Engine.Propeller.DIAMETER, val=0.0, units='ft')
        add_aviary_input(
            self,
            Dynamic.Vehicle.Propulsion.PROPELLER_TIP_SPEED,
            val=np.zeros(nn),
            units='ft/s',
        )
        add_aviary_input(self, Dynamic.Vehicle.Propulsion.SHAFT_POWER, val=np.zeros(nn), units='hp')
        add_aviary_input(self, Dynamic.Atmosphere.DENSITY, val=np.zeros(nn), units='slug/ft**3')
        add_aviary_input(self, Dynamic.Mission.VELOCITY, val=np.zeros(nn), units='ft/s')
        add_aviary_input(self, Dynamic.Atmosphere.SPEED_OF_SOUND, val=np.zeros(nn), units='ft/s')

        self.add_output('power_coefficient', val=np.zeros(nn), units='unitless')
        self.add_output('advance_ratio', val=np.zeros(nn), units='unitless')
        self.add_output('tip_mach', val=np.zeros(nn), units='unitless')
        # TODO this conflicts with 2DOF phases that also output density ratio
        # Right now repeating calculation in post-HS component where it is also used
        # self.add_output('density_ratio', val=np.zeros(nn), units='unitless')

    def setup_partials(self):
        arange = np.arange(self.options['num_nodes'])

        # self.declare_partials(
        #     'density_ratio', Dynamic.Atmosphere.DENSITY, rows=arange, cols=arange)
        self.declare_partials(
            'tip_mach',
            [
                Dynamic.Vehicle.Propulsion.PROPELLER_TIP_SPEED,
                Dynamic.Atmosphere.SPEED_OF_SOUND,
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(
            'advance_ratio',
            [
                Dynamic.Mission.VELOCITY,
                Dynamic.Vehicle.Propulsion.PROPELLER_TIP_SPEED,
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(
            'power_coefficient',
            [
                Dynamic.Vehicle.Propulsion.SHAFT_POWER,
                Dynamic.Atmosphere.DENSITY,
                Dynamic.Vehicle.Propulsion.PROPELLER_TIP_SPEED,
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials('power_coefficient', Aircraft.Engine.Propeller.DIAMETER)

    def compute(self, inputs, outputs):
        diam_prop = inputs[Aircraft.Engine.Propeller.DIAMETER]
        shp = inputs[Dynamic.Vehicle.Propulsion.SHAFT_POWER]
        vtas = inputs[Dynamic.Mission.VELOCITY]
        tipspd = inputs[Dynamic.Vehicle.Propulsion.PROPELLER_TIP_SPEED]
        sos = inputs[Dynamic.Atmosphere.SPEED_OF_SOUND]

        # arbitrarily small number to keep advance ratio nonzero, which allows for static thrust prediction
        vtas[np.where(vtas <= 1e-6)] = 1e-6
        density_ratio = inputs[Dynamic.Atmosphere.DENSITY] / RHO_SEA_LEVEL_ENGLISH

        if diam_prop <= 0.0:
            raise om.AnalysisError('Aircraft.Engine.Propeller.DIAMETER must be positive.')
        if any(tipspd) <= 0.0:
            raise om.AnalysisError(
                'Dynamic.Vehicle.Propulsion.PROPELLER_TIP_SPEED must be positive.'
            )
        if any(sos) <= 0.0:
            raise om.AnalysisError('Dynamic.Atmosphere.SPEED_OF_SOUND must be positive.')
        if any(density_ratio) <= 0.0:
            raise om.AnalysisError('Dynamic.Atmosphere.DENSITY must be positive.')
        if any(shp) < 0.0:
            raise om.AnalysisError('Dynamic.Vehicle.Propulsion.SHAFT_POWER must be non-negative.')

        # outputs['density_ratio'] = density_ratio
        # TODO tip mach was already calculated, revisit this
        outputs['tip_mach'] = tipspd / sos
        outputs['advance_ratio'] = math.pi * vtas / tipspd
        # TODO back out what is going on with unit conversion factor 10e10/(2*6966)

        outputs['power_coefficient'] = (
            shp * 10.0e10 / (2 * 6966.0) / density_ratio / (tipspd**3 * diam_prop**2)
        )

    def compute_partials(self, inputs, partials):
        vtas = inputs[Dynamic.Mission.VELOCITY]
        tipspd = inputs[Dynamic.Vehicle.Propulsion.PROPELLER_TIP_SPEED]
        rho = inputs[Dynamic.Atmosphere.DENSITY]
        diam_prop = inputs[Aircraft.Engine.Propeller.DIAMETER]
        shp = inputs[Dynamic.Vehicle.Propulsion.SHAFT_POWER]
        sos = inputs[Dynamic.Atmosphere.SPEED_OF_SOUND]

        unit_conversion_const = 10.0e10 / (2 * 6966.0)

        # partials["density_ratio", Dynamic.Atmosphere.DENSITY] = 1 / RHO_SEA_LEVEL_ENGLISH
        partials['tip_mach', Dynamic.Vehicle.Propulsion.PROPELLER_TIP_SPEED] = 1 / sos
        partials['tip_mach', Dynamic.Atmosphere.SPEED_OF_SOUND] = -tipspd / sos**2
        partials['advance_ratio', Dynamic.Mission.VELOCITY] = math.pi / tipspd
        partials['advance_ratio', Dynamic.Vehicle.Propulsion.PROPELLER_TIP_SPEED] = (
            -math.pi * vtas / (tipspd * tipspd)
        )
        partials['power_coefficient', Dynamic.Vehicle.Propulsion.SHAFT_POWER] = (
            unit_conversion_const * RHO_SEA_LEVEL_ENGLISH / (rho * tipspd**3 * diam_prop**2)
        )
        partials['power_coefficient', Dynamic.Atmosphere.DENSITY] = (
            -unit_conversion_const
            * shp
            * RHO_SEA_LEVEL_ENGLISH
            / (rho * rho * tipspd**3 * diam_prop**2)
        )
        partials['power_coefficient', Dynamic.Vehicle.Propulsion.PROPELLER_TIP_SPEED] = (
            -3
            * unit_conversion_const
            * shp
            * RHO_SEA_LEVEL_ENGLISH
            / (rho * tipspd**4 * diam_prop**2)
        )
        partials['power_coefficient', Aircraft.Engine.Propeller.DIAMETER] = (
            -2
            * unit_conversion_const
            * shp
            * RHO_SEA_LEVEL_ENGLISH
            / (rho * tipspd**3 * diam_prop**3)
        )


class HamiltonStandard(om.ExplicitComponent):
    """
    This is Hamilton Standard component rewritten from Fortran code.
    The original documentation is available at
    https://ntrs.nasa.gov/api/citations/19720010354/downloads/19720010354.pdf
    It computes the thrust coefficient of a propeller blade.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

        add_aviary_option(self, Aircraft.Engine.Propeller.NUM_BLADES)
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('power_coefficient', val=np.zeros(nn), units='unitless')
        self.add_input('advance_ratio', val=np.zeros(nn), units='unitless')
        add_aviary_input(self, Dynamic.Atmosphere.MACH, val=np.zeros(nn), units='unitless')
        self.add_input('tip_mach', val=np.zeros(nn), units='unitless')
        add_aviary_input(
            self, Aircraft.Engine.Propeller.ACTIVITY_FACTOR, val=0.0, units='unitless'
        )  # Actitivty Factor per Blade
        add_aviary_input(
            self,
            Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT,
            val=0.0,
            units='unitless',
        )  # blade integrated lift coeff

        self.add_output('thrust_coefficient', val=np.zeros(nn), units='unitless')
        # propeller tip compressibility loss factor
        self.add_output('comp_tip_loss_factor', val=np.zeros(nn), units='unitless')

        self.declare_partials('*', '*', method='fd', form='forward')

    def compute(self, inputs, outputs):
        verbosity = self.options[Settings.VERBOSITY]
        num_blades = self.options[Aircraft.Engine.Propeller.NUM_BLADES]

        act_factor = inputs[Aircraft.Engine.Propeller.ACTIVITY_FACTOR][0]
        cli = inputs[Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT][0]

        # TODO verify this works with multiple engine models (i.e. prop mission is
        #      properly slicing these inputs)
        # ensure num_blades is an int, so it can be used as array index later
        try:
            len(num_blades)
        except TypeError:
            num_blades = int(num_blades)
        else:
            num_blades = int(num_blades[0])

        for i_node in range(self.options['num_nodes']):
            ichck = 0
            run_flag = 0
            xft = 1.0
            AF_adj_CP = np.zeros(7)  # AFCP: an AF adjustment of CP to be assigned
            AF_adj_CT = np.zeros(7)  # AFCT: an AF adjustment of CT to be assigned
            CTT = np.zeros(7)
            BLL = np.zeros(7)
            BLLL = np.zeros(7)
            PXCLI = np.zeros(7)
            XFFT = np.zeros(6)
            CTG = np.zeros(11)
            CTG1 = np.zeros(11)
            TXCLI = np.zeros(6)
            CTTT = np.zeros(4)
            XXXFT = np.zeros(4)

            for k in range(2):
                AF_adj_CP[k], run_flag = _unint(Act_Factor_arr, AFCPC[k], act_factor)
                AF_adj_CT[k], run_flag = _unint(Act_Factor_arr, AFCTC[k], act_factor)
            for k in range(2, 7):
                AF_adj_CP[k] = AF_adj_CP[1]
                AF_adj_CT[k] = AF_adj_CT[1]
            if inputs['advance_ratio'][i_node] <= 0.5:
                AFCTE = (
                    2.0 * inputs['advance_ratio'][i_node] * (AF_adj_CT[1] - AF_adj_CT[0])
                    + AF_adj_CT[0]
                )
            else:
                AFCTE = AF_adj_CT[1]

            # bounding J (advance ratio) for setting up interpolation
            if inputs['advance_ratio'][i_node] <= 1.0:
                J_begin = 0
                J_end = 3
            elif inputs['advance_ratio'][i_node] <= 1.5:
                J_begin = 1
                J_end = 4
            elif inputs['advance_ratio'][i_node] <= 2.0:
                J_begin = 2
                J_end = 5
            else:
                J_begin = 3
                J_end = 6

            CL_tab_idx_begin = 0  # NCLT
            CL_tab_idx_end = 0  # NCLTT
            # flag that given lift coeff (cli) does not fall on a node point of CL_arr
            CL_tab_idx_flg = 0  # NCL_flg
            ifnd = 0

            power_coefficient = inputs['power_coefficient'][i_node]
            for ii in range(6):
                cl_idx = ii
                if abs(cli - CL_arr[ii]) <= 0.0009:
                    ifnd = 1
                    break
            if ifnd == 0:
                if cli <= 0.6:
                    CL_tab_idx_begin = 0
                    CL_tab_idx_end = 3
                elif cli <= 0.7:
                    CL_tab_idx_begin = 1
                    CL_tab_idx_end = 4
                else:
                    CL_tab_idx_begin = 2
                    CL_tab_idx_end = 5
            else:
                CL_tab_idx_begin = cl_idx
                CL_tab_idx_end = cl_idx
                # flag that given lift coeff (cli) falls on a node point of CL_arr
                CL_tab_idx_flg = 1

            lmod = (num_blades % 2) + 1
            if lmod == 1:
                nbb = 1
                idx_blade = int(num_blades / 2)
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
                for kdx in range(J_begin, J_end + 1):
                    CP_Eff = power_coefficient * AF_adj_CP[kdx]
                    PBL, run_flag = _unint(CPEC, BL_P_corr_table[idx_blade], CP_Eff)
                    # PBL = number of blades correction for power_coefficient
                    CPE1 = CP_Eff * PBL * PF_CLI_arr[kdx]
                    CL_tab_idx = CL_tab_idx_begin
                    for kl in range(CL_tab_idx_begin, CL_tab_idx_end + 1):
                        CPE1X = CPE1
                        if CPE1 < CP_CLi_table[CL_tab_idx][0]:
                            CPE1X = CP_CLi_table[CL_tab_idx][0]
                        cli_len = cli_arr_len[CL_tab_idx]
                        PXCLI[kl], run_flag = _unint(
                            CP_CLi_table[CL_tab_idx][:cli_len], XPCLI[CL_tab_idx], CPE1X
                        )
                        if run_flag == 1:
                            ichck = ichck + 1
                        if verbosity == Verbosity.DEBUG or ichck <= Verbosity.BRIEF:
                            if run_flag == 1:
                                warnings.warn(
                                    f'Mach = {inputs[Dynamic.Atmosphere.MACH][i_node]}\n'
                                    f'VTMACH = {inputs["tip_mach"][i_node]}\n'
                                    f'J = {inputs["advance_ratio"][i_node]}\n'
                                    f'power_coefficient = {power_coefficient}\n'
                                    f'CP_Eff = {CP_Eff}'
                                )
                            if kl == 4 and CPE1 < 0.010:
                                print(
                                    f'Extrapolated data is being used for CLI=.6--CPE1,PXCLI,L= , {CPE1},{PXCLI[kl]},{idx_blade}   Suggest inputting CLI=.5'
                                )
                            if kl == 5 and CPE1 < 0.010:
                                print(
                                    f'Extrapolated data is being used for CLI=.7--CPE1,PXCLI,L= , {CPE1},{PXCLI[kl]},{idx_blade}   Suggest inputting CLI=.5'
                                )
                            if kl == 6 and CPE1 < 0.010:
                                print(
                                    f'Extrapolated data is being used for CLI=.8--CPE1,PXCLI,L= , {CPE1},{PXCLI[kl]},{idx_blade}   Suggest inputting CLI=.5'
                                )
                        NERPT = 1
                        CL_tab_idx = CL_tab_idx + 1
                    if CL_tab_idx_flg != 1:
                        PCLI, run_flag = _unint(
                            CL_arr[CL_tab_idx_begin : CL_tab_idx_begin + 4],
                            PXCLI[CL_tab_idx_begin : CL_tab_idx_begin + 4],
                            inputs[Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT][0],
                        )
                    else:
                        PCLI = PXCLI[CL_tab_idx_begin]
                        # PCLI = CLI adjustment to power_coefficient
                    CP_Eff = CP_Eff * PCLI  # the effective CP at baseline point for kdx
                    ang_len = ang_arr_len[kdx]
                    BLL[kdx], run_flag = _unint(
                        # blade angle at baseline point for kdx
                        CP_Angle_table[idx_blade][kdx][:ang_len],
                        Blade_angle_table[kdx],
                        CP_Eff,
                    )
                    try:
                        CTT[kdx], run_flag = _unint(
                            # thrust coeff at baseline point for kdx
                            Blade_angle_table[kdx][:ang_len],
                            CT_Angle_table[idx_blade][kdx][:ang_len],
                            BLL[kdx],
                        )
                    except IndexError:
                        raise om.AnalysisError(
                            'interp failed for CTT (thrust coefficient) in hamilton_standard.py'
                        )
                    if run_flag > 1:
                        NERPT = 2
                        print(f'ERROR IN PROP. PERF.-- NERPT={NERPT}, run_flag={run_flag}')

                BLLL[ibb], run_flag = _unint(
                    advance_ratio_array[J_begin : J_begin + 4],
                    BLL[J_begin : J_begin + 4],
                    inputs['advance_ratio'][i_node],
                )
                ang_blade = BLLL[ibb]
                CTTT[ibb], run_flag = _unint(
                    advance_ratio_array[J_begin : J_begin + 4],
                    CTT[J_begin : J_begin + 4],
                    inputs['advance_ratio'][i_node],
                )

                # make extra correction. CTG is an "error" function, and the iteration (loop counter = "IL") tries to drive CTG/CT to 0
                # ERR_CT = CTG1[il]/CTTT[ibb], where CTG1 =CT_Eff - CTTT(IBB).
                CTG[0] = 0.100
                CTG[1] = 0.200
                TFCLII, run_flag = _unint(
                    advance_ratio_array, TF_CLI_arr, inputs['advance_ratio'][i_node]
                )
                NCTG = 10
                ifnd1 = 0
                ifnd2 = 0
                for il in range(NCTG):
                    ct = CTG[il]
                    CT_Eff = CTG[il] * AFCTE
                    TBL, run_flag = _unint(CTEC, BL_T_corr_table[idx_blade], CT_Eff)
                    # TBL = number of blades correction for thrust_coefficient
                    CTE1 = CT_Eff * TBL * TFCLII
                    CL_tab_idx = CL_tab_idx_begin
                    for kl in range(CL_tab_idx_begin, CL_tab_idx_end + 1):
                        CTE1X = CTE1
                        if CTE1 < CT_CLi_table[CL_tab_idx][0]:
                            CTE1X = CT_CLi_table[CL_tab_idx][0]
                        cli_len = cli_arr_len[CL_tab_idx]
                        TXCLI[kl], run_flag = _unint(
                            CT_CLi_table[CL_tab_idx][:cli_len], XTCLI[CL_tab_idx][:cli_len], CTE1X
                        )
                        NERPT = 5
                        if run_flag == 1:
                            # off lower bound only.
                            print(
                                f'ERROR IN PROP. PERF.-- NERPT={NERPT}, '
                                f'run_flag={run_flag}, il={il}, kl = {kl}'
                            )
                        if inputs['advance_ratio'][i_node] != 0.0:
                            ZMCRT, run_flag = _unint(
                                advance_ratio_array2,
                                mach_corr_table[CL_tab_idx],
                                inputs['advance_ratio'][i_node],
                            )
                            DMN = inputs[Dynamic.Atmosphere.MACH][i_node] - ZMCRT
                        else:
                            ZMCRT = mach_tip_corr_arr[CL_tab_idx]
                            DMN = inputs['tip_mach'][i_node] - ZMCRT
                        XFFT[kl] = 1.0  # compressibility tip loss factor
                        if DMN > 0.0:
                            CTE2 = CT_Eff * TXCLI[kl] * TBL
                            XFFT[kl], run_flag = _biquad(comp_mach_CT_arr, 1, DMN, CTE2)
                        CL_tab_idx = CL_tab_idx + 1
                    if CL_tab_idx_flg != 1:
                        cli = inputs[Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT][0]
                        TCLII, run_flag = _unint(
                            CL_arr[CL_tab_idx_begin : CL_tab_idx_begin + 4],
                            TXCLI[CL_tab_idx_begin : CL_tab_idx_begin + 4],
                            cli,
                        )
                        xft, run_flag = _unint(
                            CL_arr[CL_tab_idx_begin : CL_tab_idx_begin + 4],
                            XFFT[CL_tab_idx_begin : CL_tab_idx_begin + 4],
                            cli,
                        )
                    else:
                        TCLII = TXCLI[CL_tab_idx_begin]
                        xft = XFFT[CL_tab_idx_begin]
                    ct = CTG[il]
                    CT_Eff = CTG[il] * AFCTE * TCLII
                    CTG1[il] = CT_Eff - CTTT[ibb]
                    if abs(CTG1[il] / CTTT[ibb]) < 0.001:
                        ifnd1 = 1
                        break
                    if il > 0:
                        CTG[il + 1] = (
                            -CTG1[il - 1] * (CTG[il] - CTG[il - 1]) / (CTG1[il] - CTG1[il - 1])
                            + CTG[il - 1]
                        )
                        if CTG[il + 1] <= 0:
                            ifnd2 = 1
                            break

                if ifnd1 == 0 and ifnd2 == 0:
                    raise ValueError(
                        'Integrated design cl adjustment not working properly for ct '
                        f'definition (ibb={ibb})'
                    )
                if ifnd1 == 0 and ifnd2 == 1:
                    ct = 0.0
                CTTT[ibb] = ct
                XXXFT[ibb] = xft
                idx_blade = idx_blade + 1

            if nbb != 1:
                # interpolation by the number of blades if odd number
                ang_blade, run_flag = _unint(num_blades_arr, BLLL[:4], num_blades)
                ct, run_flag = _unint(num_blades_arr, CTTT, num_blades)
                xft, run_flag = _unint(num_blades_arr, XXXFT, num_blades)

            # NOTE this could be handled via the metamodel comps (extrapolate flag)
            if ichck > 0:
                print(f'  table look-up error = {ichck} (if you go outside the tables.)')

            outputs['thrust_coefficient'][i_node] = ct
            outputs['comp_tip_loss_factor'][i_node] = xft


class PostHamiltonStandard(om.ExplicitComponent):
    """Post-process after HamiltonStandard run to get thrust and compressibility."""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Aircraft.Engine.Propeller.DIAMETER, val=0.0, units='ft')
        self.add_input('install_loss_factor', val=np.zeros(nn), units='unitless')
        self.add_input('thrust_coefficient', val=np.zeros(nn), units='unitless')
        self.add_input('comp_tip_loss_factor', val=np.zeros(nn), units='unitless')
        add_aviary_input(
            self,
            Dynamic.Vehicle.Propulsion.PROPELLER_TIP_SPEED,
            val=np.zeros(nn),
            units='ft/s',
        )
        self.add_input(Dynamic.Atmosphere.DENSITY, val=np.zeros(nn), units='slug/ft**3')
        self.add_input('advance_ratio', val=np.zeros(nn), units='unitless')
        self.add_input('power_coefficient', val=np.zeros(nn), units='unitless')

        self.add_output('thrust_coefficient_comp_loss', val=np.zeros(nn), units='unitless')
        add_aviary_output(self, Dynamic.Vehicle.Propulsion.THRUST, val=np.zeros(nn), units='lbf')
        # keep them for reporting but don't seem to be required
        self.add_output('propeller_efficiency', val=np.zeros(nn), units='unitless')
        self.add_output('install_efficiency', val=np.zeros(nn), units='unitless')

    def setup_partials(self):
        arange = np.arange(self.options['num_nodes'])

        self.declare_partials(
            'thrust_coefficient_comp_loss',
            [
                'thrust_coefficient',
                'comp_tip_loss_factor',
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(
            Dynamic.Vehicle.Propulsion.THRUST,
            [
                'thrust_coefficient',
                'comp_tip_loss_factor',
                Dynamic.Vehicle.Propulsion.PROPELLER_TIP_SPEED,
                Dynamic.Atmosphere.DENSITY,
                'install_loss_factor',
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(
            Dynamic.Vehicle.Propulsion.THRUST,
            [
                Aircraft.Engine.Propeller.DIAMETER,
            ],
        )
        self.declare_partials(
            'propeller_efficiency',
            [
                'advance_ratio',
                'power_coefficient',
                'thrust_coefficient',
                'comp_tip_loss_factor',
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(
            'install_efficiency',
            [
                'advance_ratio',
                'power_coefficient',
                'thrust_coefficient',
                'comp_tip_loss_factor',
                'install_loss_factor',
            ],
            rows=arange,
            cols=arange,
        )

    def compute(self, inputs, outputs):
        ctx = inputs['thrust_coefficient'] * inputs['comp_tip_loss_factor']
        outputs['thrust_coefficient_comp_loss'] = ctx
        diam_prop = inputs[Aircraft.Engine.Propeller.DIAMETER]
        tipspd = inputs[Dynamic.Vehicle.Propulsion.PROPELLER_TIP_SPEED]
        install_loss_factor = inputs['install_loss_factor']
        density_ratio = inputs[Dynamic.Atmosphere.DENSITY] / RHO_SEA_LEVEL_ENGLISH
        outputs[Dynamic.Vehicle.Propulsion.THRUST] = (
            ctx
            * tipspd**2
            * diam_prop**2
            * density_ratio
            / (1.515e06)
            * 364.76
            * (1.0 - install_loss_factor)
        )

        # avoid divide by zero when shaft power is zero
        calc_idx = np.where(inputs['power_coefficient'] > 1e-6)  # index where CP > 1e-5
        prop_eff = np.zeros(self.options['num_nodes'])
        prop_eff[calc_idx] = (
            inputs['advance_ratio'][calc_idx]
            * ctx[calc_idx]
            / inputs['power_coefficient'][calc_idx]
        )
        outputs['propeller_efficiency'] = prop_eff
        outputs['install_efficiency'] = outputs['propeller_efficiency'] * (
            1.0 - install_loss_factor
        )

    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        XFT = inputs['comp_tip_loss_factor']
        ctx = inputs['thrust_coefficient'] * XFT
        diam_prop = inputs[Aircraft.Engine.Propeller.DIAMETER]
        install_loss_factor = inputs['install_loss_factor']
        tipspd = inputs[Dynamic.Vehicle.Propulsion.PROPELLER_TIP_SPEED]
        density_ratio = inputs[Dynamic.Atmosphere.DENSITY] / RHO_SEA_LEVEL_ENGLISH

        unit_conversion_factor = 364.76 / 1.515e06
        partials['thrust_coefficient_comp_loss', 'thrust_coefficient'] = XFT
        partials['thrust_coefficient_comp_loss', 'comp_tip_loss_factor'] = inputs[
            'thrust_coefficient'
        ]
        partials[Dynamic.Vehicle.Propulsion.THRUST, 'thrust_coefficient'] = (
            XFT
            * tipspd**2
            * diam_prop**2
            * density_ratio
            * unit_conversion_factor
            * (1.0 - install_loss_factor)
        )
        partials[Dynamic.Vehicle.Propulsion.THRUST, 'comp_tip_loss_factor'] = (
            inputs['thrust_coefficient']
            * tipspd**2
            * diam_prop**2
            * density_ratio
            * unit_conversion_factor
            * (1.0 - install_loss_factor)
        )
        partials[
            Dynamic.Vehicle.Propulsion.THRUST, Dynamic.Vehicle.Propulsion.PROPELLER_TIP_SPEED
        ] = (
            2
            * ctx
            * tipspd
            * diam_prop**2
            * density_ratio
            * unit_conversion_factor
            * (1.0 - install_loss_factor)
        )
        partials[Dynamic.Vehicle.Propulsion.THRUST, Aircraft.Engine.Propeller.DIAMETER] = (
            2
            * ctx
            * tipspd**2
            * diam_prop
            * density_ratio
            * unit_conversion_factor
            * (1.0 - install_loss_factor)
        )
        partials[Dynamic.Vehicle.Propulsion.THRUST, Dynamic.Atmosphere.DENSITY] = (
            ctx
            * tipspd**2
            * diam_prop**2
            * unit_conversion_factor
            * (1.0 - install_loss_factor)
            / RHO_SEA_LEVEL_ENGLISH
        )
        partials[Dynamic.Vehicle.Propulsion.THRUST, 'install_loss_factor'] = (
            -ctx * tipspd**2 * diam_prop**2 * density_ratio * unit_conversion_factor
        )

        calc_idx = np.where(inputs['power_coefficient'] > 1e-6)
        pow_coeff = inputs['power_coefficient']
        adv_ratio = inputs['advance_ratio']

        deriv_propeff_adv = np.zeros(nn, dtype=pow_coeff.dtype)
        deriv_propeff_ct = np.zeros(nn, dtype=pow_coeff.dtype)
        deriv_propeff_tip = np.zeros(nn, dtype=pow_coeff.dtype)
        deriv_propeff_cp = np.zeros(nn, dtype=pow_coeff.dtype)

        deriv_propeff_adv[calc_idx] = ctx[calc_idx] / pow_coeff[calc_idx]
        deriv_propeff_ct[calc_idx] = adv_ratio[calc_idx] * XFT[calc_idx] / pow_coeff[calc_idx]
        deriv_propeff_tip[calc_idx] = (
            adv_ratio[calc_idx] * inputs['thrust_coefficient'][calc_idx] / pow_coeff[calc_idx]
        )
        deriv_propeff_cp[calc_idx] = -adv_ratio[calc_idx] * ctx[calc_idx] / pow_coeff[calc_idx] ** 2

        partials['propeller_efficiency', 'advance_ratio'] = deriv_propeff_adv
        partials['propeller_efficiency', 'thrust_coefficient'] = deriv_propeff_ct
        partials['propeller_efficiency', 'comp_tip_loss_factor'] = deriv_propeff_tip
        partials['propeller_efficiency', 'power_coefficient'] = deriv_propeff_cp

        deriv_insteff_adv = np.zeros(nn, dtype=pow_coeff.dtype)
        deriv_insteff_ct = np.zeros(nn, dtype=pow_coeff.dtype)
        deriv_insteff_tip = np.zeros(nn, dtype=pow_coeff.dtype)
        deriv_insteff_cp = np.zeros(nn, dtype=pow_coeff.dtype)
        deriv_insteff_lf = np.zeros(nn, dtype=pow_coeff.dtype)

        deriv_insteff_adv[calc_idx] = (
            ctx[calc_idx] / pow_coeff[calc_idx] * (1.0 - install_loss_factor[calc_idx])
        )
        deriv_insteff_ct[calc_idx] = (
            adv_ratio[calc_idx]
            * XFT[calc_idx]
            / pow_coeff[calc_idx]
            * (1.0 - install_loss_factor[calc_idx])
        )
        deriv_insteff_tip[calc_idx] = (
            adv_ratio[calc_idx]
            * inputs['thrust_coefficient'][calc_idx]
            / pow_coeff[calc_idx]
            * (1.0 - install_loss_factor[calc_idx])
        )
        deriv_insteff_cp[calc_idx] = (
            -adv_ratio[calc_idx]
            * ctx[calc_idx]
            / pow_coeff[calc_idx] ** 2
            * (1.0 - install_loss_factor[calc_idx])
        )
        deriv_insteff_lf[calc_idx] = -adv_ratio[calc_idx] * ctx[calc_idx] / pow_coeff[calc_idx]

        partials['install_efficiency', 'advance_ratio'] = deriv_insteff_adv
        partials['install_efficiency', 'thrust_coefficient'] = deriv_insteff_ct
        partials['install_efficiency', 'comp_tip_loss_factor'] = deriv_insteff_tip
        partials['install_efficiency', 'power_coefficient'] = deriv_insteff_cp
        partials['install_efficiency', 'install_loss_factor'] = deriv_insteff_lf
