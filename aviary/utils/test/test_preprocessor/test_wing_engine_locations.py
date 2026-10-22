import unittest

from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.preprocessors import preprocess_propulsion
from aviary.validation_cases.validation_tests import get_flops_inputs
from aviary.variable_info.variables import Aircraft, Settings

E = Aircraft.Engine


def _make_engine(**options):
    engine_options = AviaryValues()
    engine_options.set_val(E.DATA_FILE, 'models/engines/turbofan_28k.csv')
    engine_options.set_val(Settings.VERBOSITY, 0)
    for key, val in options.items():
        engine_options.set_val(key, val)
    return EngineDeck(options=engine_options)


def _preprocess(*engines, shared=None):
    aviary_options = get_flops_inputs('LargeSingleAisle1FLOPS')
    aviary_options.set_val(Settings.VERBOSITY, 0)
    if shared is not None:
        aviary_options.set_val(E.WING_LOCATIONS, shared)
    preprocess_propulsion(aviary_options, list(engines))
    return aviary_options


class WingEngineLocationCheckTest(unittest.TestCase):
    """Tests for the number of Aircraft.Engine.WING_LOCATIONS values (issue #1189)."""

    def test_matching_count_passes(self):
        engine = _make_engine(**{E.NUM_ENGINES: 2, E.NUM_WING_ENGINES: 2, E.WING_LOCATIONS: [0.3]})
        _preprocess(engine)

    def test_too_many_locations_raises(self):
        engine = _make_engine(
            **{E.NUM_ENGINES: 2, E.NUM_WING_ENGINES: 2, E.WING_LOCATIONS: [0.3, 0.5]}
        )
        with self.assertRaisesRegex(ValueError, r'has 2 value\(s\).*require 1 location'):
            _preprocess(engine)

    def test_too_few_locations_raises(self):
        engine = _make_engine(**{E.NUM_ENGINES: 4, E.NUM_WING_ENGINES: 4, E.WING_LOCATIONS: [0.3]})
        with self.assertRaisesRegex(ValueError, r'has 1 value\(s\).*require 2 location'):
            _preprocess(engine)

    def test_multiple_engine_models_valid(self):
        engine1 = _make_engine(
            **{E.NUM_ENGINES: 4, E.NUM_WING_ENGINES: 4, E.WING_LOCATIONS: [0.2, 0.8]}
        )
        engine2 = _make_engine(**{E.NUM_ENGINES: 2, E.NUM_WING_ENGINES: 2, E.WING_LOCATIONS: [0.5]})
        _preprocess(engine1, engine2)

    def test_multiple_engine_models_second_invalid_raises(self):
        engine1 = _make_engine(
            **{E.NUM_ENGINES: 4, E.NUM_WING_ENGINES: 4, E.WING_LOCATIONS: [0.2, 0.8]}
        )
        engine2 = _make_engine(
            **{E.NUM_ENGINES: 2, E.NUM_WING_ENGINES: 2, E.WING_LOCATIONS: [0.4, 0.5]}
        )
        with self.assertRaises(ValueError):
            _preprocess(engine1, engine2)

    def test_no_wing_engines_ignores_locations(self):
        engine = _make_engine(
            **{
                E.NUM_ENGINES: 2,
                E.NUM_WING_ENGINES: 0,
                E.NUM_FUSELAGE_ENGINES: 2,
                E.WING_LOCATIONS: [0.1, 0.2],
            }
        )
        _preprocess(engine)

    def test_odd_wing_engines_use_floor_of_half(self):
        # three wing engines -> one symmetric pair, the odd engine is ignored
        engine = _make_engine(**{E.NUM_ENGINES: 3, E.NUM_WING_ENGINES: 3, E.WING_LOCATIONS: [0.3]})
        _preprocess(engine)

        engine = _make_engine(
            **{E.NUM_ENGINES: 3, E.NUM_WING_ENGINES: 3, E.WING_LOCATIONS: [0.3, 0.5]}
        )
        with self.assertRaises(ValueError):
            _preprocess(engine)

    def test_locations_not_supplied_by_engine_are_not_checked(self):
        # the location comes from the shared aviary options and is broadcast
        engine1 = _make_engine(**{E.NUM_ENGINES: 2, E.NUM_WING_ENGINES: 2})
        engine2 = _make_engine(**{E.NUM_ENGINES: 4, E.NUM_WING_ENGINES: 4})
        _preprocess(engine1, engine2)

    def test_shared_array_matching_count_passes(self):
        engine = _make_engine(**{E.NUM_ENGINES: 4, E.NUM_WING_ENGINES: 4})
        _preprocess(engine, shared=[0.3, 0.7])

    def test_shared_array_too_many_raises(self):
        engine = _make_engine(**{E.NUM_ENGINES: 2, E.NUM_WING_ENGINES: 2})
        with self.assertRaisesRegex(ValueError, r'has 3 value\(s\).*require 1 location'):
            _preprocess(engine, shared=[0.2, 0.5, 0.8])

    def test_shared_array_too_few_raises(self):
        engine = _make_engine(**{E.NUM_ENGINES: 4, E.NUM_WING_ENGINES: 4})
        with self.assertRaisesRegex(ValueError, r'has 3 value\(s\).*require 2 location'):
            _preprocess(engine, shared=[0.3, 0.7, 0.9])

    def test_shared_single_value_is_not_checked(self):
        engine = _make_engine(**{E.NUM_ENGINES: 4, E.NUM_WING_ENGINES: 4})
        _preprocess(engine, shared=[0.3])

    def test_shared_array_ignored_when_engine_supplies_its_own(self):
        engine = _make_engine(**{E.NUM_ENGINES: 2, E.NUM_WING_ENGINES: 2, E.WING_LOCATIONS: [0.3]})
        _preprocess(engine, shared=[0.2, 0.5, 0.8])

    def test_shared_array_not_checked_with_multiple_models(self):
        engine1 = _make_engine(**{E.NUM_ENGINES: 2, E.NUM_WING_ENGINES: 2})
        engine2 = _make_engine(**{E.NUM_ENGINES: 4, E.NUM_WING_ENGINES: 4})
        _preprocess(engine1, engine2, shared=[0.2, 0.5, 0.8])


if __name__ == '__main__':
    unittest.main()
