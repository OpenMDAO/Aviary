#!/usr/bin/env python3
"""Apply and audit the repository-wide MAX_MACH namespace migration (#1046)."""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_SYMBOL = 'Mission.Constraints.MAX_MACH'
NEW_SYMBOL = 'Aircraft.Design.MAX_MACH'
OLD_KEY = 'mission:constraints:max_mach'
NEW_KEY = 'aircraft:design:max_mach'
SUFFIXES = {'.py', '.csv', '.json', '.ipynb', '.md', '.rst', '.txt', '.toml', '.yaml', '.yml'}
SKIP = {'.git', '.venv', 'venv', 'build', 'dist', '__pycache__', '.pytest_cache'}
ALLOW = {
    Path('tools/migrate_max_mach_namespace.py'),
    Path('aviary/variable_info/legacy_aliases.py'),
    Path('aviary/variable_info/test/test_max_mach_namespace.py'),
    Path('aviary/variable_info/migrations/max_mach_namespace_manifest.json'),
}


def files():
    for path in ROOT.rglob('*'):
        if path.is_file() and path.suffix.lower() in SUFFIXES:
            rel = path.relative_to(ROOT)
            if not any(part in SKIP for part in rel.parts):
                yield rel


def patch_variables(text):
    old = "        MAX_MACH = 'mission:constraints:max_mach'\n"
    alias = (
        '        # Backward-compatible symbol alias; new code uses Aircraft.Design.MAX_MACH.\n'
        '        MAX_MACH = Aircraft.Design.MAX_MACH\n'
    )
    if old in text:
        text = text.replace(old, alias, 1)
    elif 'MAX_MACH = Aircraft.Design.MAX_MACH' not in text:
        raise RuntimeError('Mission.Constraints.MAX_MACH definition not found')

    anchor = "        MACH = 'aircraft:design:mach'\n"
    canonical = "        MAX_MACH = 'aircraft:design:max_mach'\n"
    if canonical not in text:
        if anchor not in text:
            raise RuntimeError('Aircraft.Design.MACH anchor not found')
        text = text.replace(anchor, anchor + canonical, 1)
    return text


def patch_metadata(text):
    start = text.find('add_meta_data(\n    Aircraft.Design.MAX_MACH,')
    if start < 0:
        raise RuntimeError('MAX_MACH metadata block not found')
    end = text.find('\n)\n', start)
    if end < 0:
        raise RuntimeError('MAX_MACH metadata block end not found')
    block = text[start:end + 3]
    desc = (
        "    desc=(\n"
        "        'Maximum aircraft design Mach number. Used by FLOPS-based air-conditioning, '\n"
        "        'fuel-system, hydraulics, instruments, passenger-service, starter, and surface-'\n"
        "        'control mass correlations, and by FLOPS pre-mission aerodynamics when computing '\n"
        "        'the design lift coefficient.'\n"
        "    ),\n"
    )
    pattern = re.compile(r"    desc=.*?(?=    [a-zA-Z_]+\s*=|\)\n)", re.S)
    match = pattern.search(block)
    if match:
        block = block[:match.start()] + desc + block[match.end():]
    else:
        close = block.rfind(')\n')
        block = block[:close] + desc + block[close:]
    return text[:start] + block + text[end + 3:]


def patch_values(text):
    marker = 'from aviary.variable_info.variable_meta_data import CoreMetaData\n'
    imp = 'from aviary.variable_info.legacy_aliases import resolve_legacy_variable_name\n'
    if imp not in text:
        text = text.replace(marker, imp + marker, 1)

    marker = '        if key in meta_data:\n'
    normalize = '        key = resolve_legacy_variable_name(key)\n\n'
    if normalize + marker not in text:
        text = text.replace(marker, normalize + marker, 1)

    marker = '    def _check_units_compatibility(self, key, val, units, meta_data=CoreMetaData):\n'
    methods = '''    def get_item(self, key):
        return super().get_item(resolve_legacy_variable_name(key))

    def get_val(self, key, units='unitless'):
        return super().get_val(resolve_legacy_variable_name(key), units)

    def delete(self, key):
        return super().delete(resolve_legacy_variable_name(key))

    def __contains__(self, key):
        return super().__contains__(resolve_legacy_variable_name(key))

'''
    if 'def get_item(self, key):' not in text:
        if marker not in text:
            raise RuntimeError('AviaryValues insertion point not found')
        text = text.replace(marker, methods + marker, 1)
    return text


def patch_csv(text):
    marker = 'from aviary.variable_info.enums import Verbosity\n'
    imp = 'from aviary.variable_info.legacy_aliases import resolve_legacy_variable_name\n'
    if imp not in text:
        text = text.replace(marker, marker + imp, 1)
    marker = "                        name = re.sub('\\\\s', '_', item[0])\n"
    add = marker + '                        name = resolve_legacy_variable_name(name)\n'
    if 'name = resolve_legacy_variable_name(name)' not in text:
        if marker not in text:
            raise RuntimeError('CSV header normalization point not found')
        text = text.replace(marker, add, 1)
    return text


def rewrite(rel, text):
    if rel == Path('aviary/variable_info/variables.py'):
        text = patch_variables(text)
    if rel not in ALLOW:
        text = text.replace(OLD_SYMBOL, NEW_SYMBOL).replace(OLD_KEY, NEW_KEY)
    if rel == Path('aviary/variable_info/variable_meta_data.py'):
        text = patch_metadata(text)
    elif rel == Path('aviary/utils/aviary_values.py'):
        text = patch_values(text)
    elif rel == Path('aviary/utils/csv_data_file.py'):
        text = patch_csv(text)
    return text


def apply():
    changed = []
    counts = {'python_symbol': 0, 'serialized_key': 0}
    for rel in list(files()):
        path = ROOT / rel
        try:
            before = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            continue
        after = rewrite(rel, before)
        if before != after:
            counts['python_symbol'] += before.count(OLD_SYMBOL)
            counts['serialized_key'] += before.count(OLD_KEY)
            path.write_text(after, encoding='utf-8')
            changed.append(str(rel))

    manifest = {
        'issue': 'OpenMDAO/Aviary#1046',
        'canonical_symbol': NEW_SYMBOL,
        'canonical_key': NEW_KEY,
        'legacy_symbol': OLD_SYMBOL,
        'legacy_key': OLD_KEY,
        'changed_file_count': len(changed),
        'changed_files': sorted(changed),
        'replacement_counts': counts,
        'compatibility': {
            'python_symbol': 'Mission.Constraints.MAX_MACH aliases Aircraft.Design.MAX_MACH',
            'serialized_inputs': 'legacy raw names normalize before metadata validation/storage',
        },
    }
    out = ROOT / 'aviary/variable_info/migrations/max_mach_namespace_manifest.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return changed, counts


def leftovers():
    found = []
    for rel in files():
        if rel in ALLOW:
            continue
        try:
            text = (ROOT / rel).read_text(encoding='utf-8')
        except UnicodeDecodeError:
            continue
        if OLD_SYMBOL in text or OLD_KEY in text:
            found.append(str(rel))
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.write:
        changed, counts = apply()
        print(f'Migrated {len(changed)} files: {counts}')
    if args.check:
        remain = leftovers()
        if remain:
            print('Legacy references remain:')
            print('\n'.join(remain))
            raise SystemExit(1)
        print('MAX_MACH namespace audit passed')
    if not args.write and not args.check:
        raise SystemExit('use --write and/or --check')


if __name__ == '__main__':
    main()
