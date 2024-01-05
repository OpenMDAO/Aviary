from setuptools import find_packages, setup
from pathlib import Path
import re


# Version info is set in one place; the aviary/__init__.py file
__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.\-dev]*)["']+""",
    open('aviary/__init__.py').read(),
)[0]

with open(Path(__file__).parent / "README.md", encoding="utf-8") as f:
    long_description = f.read()

pkgname = "aviary"
extras_require = {
    "test": ["testflo", "pyxdsm", "pre-commit"],
    "examples": ["openaerostruct", "ambiance", "boring_battery @ git+https://github.com/jcchin/boring.git"],
}

all_packages = []
for packages in extras_require.values():
    all_packages.extend(packages)

extras_require["all"] = all_packages

setup(
    name="om-aviary",
    long_description=long_description,
    long_description_content_type='text/markdown',
    version=__version__,
    packages=find_packages(),
    install_requires=[
        "openmdao>=3.27.0",
        "dymos>=1.8.1",
        "hvplot",
        "numpy",
        "matplotlib",
        "pandas",
        "panel>=1.0.0",
        "parameterized",
        "simupy",
    ],
    extras_require=extras_require,
    package_data={
        pkgname: [
            "subsystems/aerodynamics/gasp_based/data/*",
            "subsystems/aerodynamics/gasp_based/test/data/*",
            "subsystems/aerodynamics/flops_based/test/*.csv",
            "subsystems/aerodynamics/flops_based/test/data/*.csv",
            "subsystems/propulsion/gasp_based/data/*",
            "validation_cases/validation_data/gasp_data/*.dat",
            "validation_cases/validation_data/gasp_data/*.csv",
            "validation_cases/validation_data/flops_data/engine_only/*.deck",
            "utils/legacy_code_data/*default_values*",
            "utils/test/*",
            "models/engines/*.deck",
            "models/engines/*.txt",
            "models/N3CC/*",
            "models/large_single_aisle_1/*",
            "models/large_single_aisle_2/*",
            "models/small_single_aisle/*",
            "models/test_aircraft/*",
            "visualization/assets/aviary_vars/*"
        ],
        f"{pkgname}.subsystems.aero.test.data": ["*.csv"],
        f"{pkgname}.subsystems.prop.test": ["*.csv"],
    },
    entry_points={
        'console_scripts': [
            'aviary=aviary.interface.cmd_entry_points:aviary_cmd',
        ],
        'openmdao_report': [
            'summary=aviary.interface.reports:register_custom_reports',
        ]
    }
)
