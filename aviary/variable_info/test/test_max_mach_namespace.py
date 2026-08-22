import tempfile
import unittest
from pathlib import Path

from aviary.utils.aviary_values import AviaryValues
from aviary.utils.csv_data_file import read_data_file
from aviary.variable_info.legacy_aliases import (
    LEGACY_VARIABLE_NAME_MAP,
    resolve_legacy_variable_name,
)
from aviary.variable_info.variable_meta_data import CoreMetaData
from aviary.variable_info.variables import Aircraft, Mission


class MaxMachNamespaceTest(unittest.TestCase):
    def test_canonical_namespace(self):
        self.assertEqual(Aircraft.Design.MAX_MACH, 'aircraft:design:max_mach')
        self.assertFalse(hasattr(Mission.Constraints, 'MAX_MACH'))
        self.assertIn(Aircraft.Design.MAX_MACH, CoreMetaData)
        self.assertNotIn('mission:constraints:max_mach', CoreMetaData)

    def test_legacy_values_access_canonical_storage(self):
        legacy = 'mission:constraints:max_mach'
        self.assertEqual(resolve_legacy_variable_name(legacy), Aircraft.Design.MAX_MACH)
        values = AviaryValues()
        values.set_val(legacy, 0.86)
        self.assertIn(legacy, values)
        self.assertIn(Aircraft.Design.MAX_MACH, values)
        self.assertEqual(values.get_val(legacy), 0.86)
        self.assertEqual(values.get_val(Aircraft.Design.MAX_MACH), 0.86)
        self.assertEqual(list(values.keys()), [Aircraft.Design.MAX_MACH])

    def test_legacy_csv_header_is_normalized(self):
        legacy = next(iter(LEGACY_VARIABLE_NAME_MAP))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'legacy.csv'
            path.write_text(f'{legacy}\n0.84\n', encoding='utf-8')
            data, _, _ = read_data_file(path, metadata=CoreMetaData)
        self.assertIn(Aircraft.Design.MAX_MACH, data)
        self.assertNotIn(legacy, data)
        self.assertEqual(data.get_val(Aircraft.Design.MAX_MACH), 0.84)

    def test_no_unapproved_legacy_references(self):
        root = Path(__file__).resolve().parents[3]
        allowed = {
            Path('aviary/variable_info/legacy_aliases.py'),
            Path('aviary/variable_info/test/test_max_mach_namespace.py'),
            Path('aviary/variable_info/migrations/max_mach_namespace_manifest.json'),
            Path('tools/migrate_max_mach_namespace.py'),
        }
        suffixes = {
            '.py',
            '.csv',
            '.json',
            '.ipynb',
            '.md',
            '.rst',
            '.txt',
            '.toml',
            '.yaml',
            '.yml',
        }
        failures = []
        for path in root.rglob('*'):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            rel = path.relative_to(root)
            if rel in allowed or '.git' in rel.parts:
                continue
            try:
                text = path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                continue
            if 'Mission.Constraints.MAX_MACH' in text or 'mission:constraints:max_mach' in text:
                failures.append(str(rel))
        self.assertEqual(failures, [], 'Legacy MAX_MACH references remain: ' + ', '.join(failures))


if __name__ == '__main__':
    unittest.main()
