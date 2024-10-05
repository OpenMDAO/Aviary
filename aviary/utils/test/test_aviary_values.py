import unittest

import numpy as np
from openmdao.utils.assert_utils import assert_near_equal

from aviary.examples.variable_meta_data_extension import ExtendedMetaData
from aviary.examples.variables_extension import Aircraft as ExtendedAircraft
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import get_path
from aviary.variable_info.enums import FlapType, GASPEngineType
from aviary.variable_info.variables import Aircraft, Mission


class TestTypes(unittest.TestCase):
    """
    Test Aviary variables have correct data types.
    """

    def test_aircraft(self):
        vals = AviaryValues()

        try:
            # if we allow units=None, this test will fail.
            vals.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY, val=3, units=None)
        except TypeError as err:
            self.assertEqual(
                str(err),
                f"AviaryValues: set_val({Aircraft.Design.PART25_STRUCTURAL_CATEGORY}): unsupported units: None"
            )
        else:
            self.fail('Expecting correct units to pass.')

        try:
            vals.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=5, units='unitless')
        except:
            self.fail('Expecting correct units and type to pass.')

        try:
            vals.set_val(Aircraft.Design.COMPUTE_HTAIL_VOLUME_COEFF,
                         val=True, units='unitless')
        except:
            self.fail('Expecting correct units and type to pass.')

        try:
            vals.set_val(Aircraft.Design.COMPUTE_VTAIL_VOLUME_COEFF,
                         val=True, units='unitless')
        except:
            self.fail('Expecting correct units and type to pass.')

        try:
            vals.set_val(Aircraft.CrewPayload.NUM_PASSENGERS,
                         val='five', units='unitless')
        except TypeError as err:
            self.assertEqual(
                str(err),
                f"{Aircraft.CrewPayload.NUM_PASSENGERS} is of type(s) <class 'int'> but"
                " you have provided a value of type <class 'str'>.")
        else:
            self.fail('Expecting TypeError.')

        try:
            vals.set_val(Aircraft.Design.COMPUTE_HTAIL_VOLUME_COEFF,
                         val='five', units='unitless')
        except TypeError as err:
            self.assertEqual(
                str(err),
                f"{Aircraft.Design.COMPUTE_HTAIL_VOLUME_COEFF} is of type(s) <class"
                " 'bool'> but you have provided a value of type <class 'str'>.")
        else:
            self.fail('Expecting TypeError.')

        try:
            vals.set_val(Aircraft.Design.COMPUTE_VTAIL_VOLUME_COEFF,
                         val='five', units='unitless')
        except TypeError as err:
            self.assertEqual(
                str(err),
                f"{Aircraft.Design.COMPUTE_VTAIL_VOLUME_COEFF} is of type(s) <class"
                " 'bool'> but you have provided a value of type <class 'str'>.")
        else:
            self.fail('Expecting TypeError.')

        try:
            vals.set_val(Aircraft.Engine.TYPE, GASPEngineType.TURBOJET)
            self.assertTrue(vals.get_val(Aircraft.Engine.TYPE)
                            is GASPEngineType.TURBOJET)
        except:
            self.fail('Expecting to be able to set the value of an Enum.')

        try:
            vals.set_val(Aircraft.Engine.TYPE, 'turbojet')
            self.assertTrue(vals.get_val(Aircraft.Engine.TYPE)
                            is GASPEngineType.TURBOJET)
        except:
            self.fail('Expecting to be able to set the value of an Enum from a string.')

        try:
            vals.set_val(Aircraft.Engine.TYPE, 'TURBOJET')
            self.assertTrue(vals.get_val(Aircraft.Engine.TYPE)
                            is GASPEngineType.TURBOJET)
        except:
            self.fail('Expecting to be able to set the value of an Enum from a string.')

        try:
            vals.set_val(Aircraft.Engine.TYPE, 7)
            self.assertTrue(vals.get_val(Aircraft.Engine.TYPE)
                            is GASPEngineType.TURBOJET)
        except:
            self.fail('Expecting to be able to set the value of an Enum from an int.')

        try:
            vals.set_val(Aircraft.Engine.TYPE, FlapType.DOUBLE_SLOTTED)
        except ValueError as err:
            self.assertEqual(str(err),
                             "<FlapType.DOUBLE_SLOTTED: 4> is not a valid GASPEngineType")
        else:
            self.fail("Expecting ValueError.")

        try:
            vals.set_val(Aircraft.Engine.DATA_FILE, np.array([]))
        except IndexError as err:
            self.fail("Expecting to be able to set the value of an empty numpy array.")
        else:
            # NOTE comparing two empty numpy arrays will fail (unlike empty lists), so
            #      arrays are appended to for this test
            self.assertEqual(np.append(vals.get_val(
                Aircraft.Engine.DATA_FILE), 0), np.array([0]))

    def test_mission(self):
        vals = AviaryValues()

        try:
            vals.set_val(Mission.Design.CRUISE_ALTITUDE, val=35000, units='ft')
        except:
            self.fail('Expecting correct units and type to pass.')

        try:
            vals.set_val(Mission.Design.CRUISE_ALTITUDE, val=35000.01, units='ft')
        except:
            self.fail('Expecting correct units and type to pass.')

        try:
            vals.set_val(Mission.Design.CRUISE_ALTITUDE, val=True, units='ft')
        except TypeError as err:
            self.assertEqual(
                str(err),
                f"{Mission.Design.CRUISE_ALTITUDE} is of type(s) [<class 'int'>, <class"
                " 'float'>] but you have provided a value of type <class 'bool'>.")
        else:
            self.fail('Expecting TypeError.')

        try:
            vals.set_val(Mission.Design.CRUISE_ALTITUDE, val='fail', units='ft')
        except TypeError as err:
            self.assertEqual(
                str(err),
                f"{Mission.Design.CRUISE_ALTITUDE} is of type(s) [<class 'int'>, <class"
                " 'float'>] but you have provided a value of type <class 'str'>.")
        else:
            self.fail('Expecting TypeError.')


class TestUnits(unittest.TestCase):
    """
    Test Aviary variables have correct units.
    """

    def test_aircraft(self):
        vals = AviaryValues()

        try:
            vals.set_val(Aircraft.Wing.LOADING, val=20, units='N/cm**2')
        except:
            self.fail('Expecting correct units and type to pass.')

        try:
            vals.set_val(Aircraft.Wing.LOADING, val=20, units='kgf/cm**2')
        except ValueError as err:
            self.assertEqual(
                str(err),
                'The units kgf/cm**2 which you have provided for'
                f' {Aircraft.Wing.LOADING} are invalid.')
        else:
            self.fail('Expecting ValueError.')

        try:
            vals.set_val(Aircraft.Wing.LOADING, val=20, units='inch**2/NM')
        except TypeError as err:
            self.assertEqual(
                str(err),
                f"The base units of {Aircraft.Wing.LOADING} are lbf/ft**2, and you have"
                f" tried to set {Aircraft.Wing.LOADING} with units of inch**2/NM, which"
                " are not compatible.")
        else:
            self.fail('Expecting TypeError.')

        try:
            vals.set_val(Aircraft.Wing.LOADING, val=20, units='unitless')
        except TypeError as err:
            self.assertEqual(
                str(err),
                f"The base units of {Aircraft.Wing.LOADING} are lbf/ft**2, and you have"
                f" tried to set {Aircraft.Wing.LOADING} with units of unitless, which"
                " are not compatible.")
        else:
            self.fail('Expecting TypeError.')

    def test_mission(self):
        vals = AviaryValues()

        try:
            vals.set_val(Mission.Takeoff.FINAL_VELOCITY, val=20, units='ft/min')
        except:
            self.fail('Expecting correct units and type to pass.')

        try:
            vals.set_val(Mission.Takeoff.FINAL_VELOCITY, val=20, units='kg')
        except TypeError as err:
            self.assertEqual(
                str(err),
                f"The base units of {Mission.Takeoff.FINAL_VELOCITY} are m/s, and you"
                f" have tried to set {Mission.Takeoff.FINAL_VELOCITY} with units of kg,"
                " which are not compatible.")
        else:
            self.fail('Expecting TypeError.')

        try:
            vals.set_val(Mission.Takeoff.FINAL_VELOCITY, val=20, units='min/ft')
        except TypeError as err:
            self.assertEqual(
                str(err),
                f"The base units of {Mission.Takeoff.FINAL_VELOCITY} are m/s, and you"
                f" have tried to set {Mission.Takeoff.FINAL_VELOCITY} with units of"
                " min/ft, which are not compatible.")
        else:
            self.fail('Expecting TypeError.')


class TestVariableExtension(unittest.TestCase):
    """
    Test set_val function for extended Aviary variables.
    """

    def test_set_val_metadata_extension(self):
        option_defaults = AviaryValues()

        filename = get_path(
            'models/engines/turbofan_23k_1.deck')
        option_defaults.set_val(ExtendedAircraft.Engine.DATA_FILE, filename)
        option_defaults.set_val(ExtendedAircraft.Wing.AERO_CENTER, val=5, units='ft',
                                meta_data=ExtendedMetaData)

        check_val = option_defaults.get_val(
            ExtendedAircraft.Wing.AERO_CENTER, units='inch')

        assert_near_equal(check_val, 60, 1e-9)


if __name__ == "__main__":
    unittest.main()
