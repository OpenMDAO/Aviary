from setuptools import find_packages, setup
import re


# Version info is set in one place; the aviary/__init__.py file
__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.\-dev]*)["']+""",
    open('aviary/__init__.py').read(),
)[0]

pkgname = "aviary"
extras_require = {
    "test": ["testflo", "pyxdsm", "pre-commit"],
    "optimizers": ["pyoptsparse"],
    "examples": ["openaerostruct", "ambiance", "boring_battery @ git+https://github.com/jcchin/boring.git"],
}

all_packages = []
for packages in extras_require.values():
    all_packages.extend(packages)

extras_require["all"] = all_packages

setup(
    name=pkgname,
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
            "subsystems/aero/data/*.txt",
            "subsystems/prop/data/*.eng",
            "validation_cases/validation_data/gasp_data/*.dat",
            "validation_cases/validation_data/gasp_data/*.csv",
            "utils/*default_values.dat",
        ],
        f"{pkgname}.subsystems.aero.test.data": ["*.csv"],
        f"{pkgname}.subsystems.prop.test": ["*.csv"],
    },
    entry_points={
        'console_scripts': [
            'aviary=aviary.interface.cmd_entry_points:aviary_cmd',
        ],
        'openmdao_report': [
            'aviary_reports=aviary.interface.reports:register_custom_reports',
        ]
    }
)
