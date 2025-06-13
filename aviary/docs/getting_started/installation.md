# Installation

Aviary can be installed on any OS, including Windows, Linux, and macOS (as well as integrated shells like unix on macOS or Linux through WSL on Windows). 

We recommend using a conda environment, as it makes it easier to install Aviary with its full feature set, which is explained in the next section. We assume that you have a working Python or conda environment set up for all of our guides. Our recommended package manager is [miniforge](https://github.com/conda-forge/miniforge). Miniforge is an open-source alternative to miniconda, and functions identically.

```{note}
The minimum supported version of Python is 3.9; we recommend using the latest release of Python.
```

The Detailed and Developer's installation guide assume you have git installed. The Developer's Guide also assumes that if you are using your own fork of Aviary, that you have your machine correctly configured with the proper permissions to pull from and push to your fork.

## Which Installation Guide Should I Use?
The following guides are available:

- [Quick Start Guide](#quick-start-guide)
- [Detailed Installation Guide](#detailed-guide)
- [Developer's Guide](#developers-guide)
- Docker Installation Guide (coming soon)

The different installation guides are intended to get users of various levels of experience and analysis need started. Here is a brief explanation of what can differ between Aviary installs.

### Optimizers
Aviary uses the `pyOptSparse` package to get access to a variety of different optimizers. Many of the more powerful and robust optimizers do not come with pyOptSparse by default, and may require additional steps to install. The Quick Start guide does not include pyOptSparse, however you can always "upgrade" your install with pyOptSparse later to get access to more optimizers. Using a conda environment simplifies that installation process.

### Editable Source Code
If you do not follow the Developer's Guide, then you will not have an editable version of Aviary source code. Following that guide installs a copy of Aviary that you can freely modify.

### Access to Development Versions
The Quick Start Guide only grabs the latest version of Aviary, so if you need a new feature or bugfix you will have to wait until the next version release to get it. However, the other guides show how to directly get the latest Aviary code. These versions are generally stable and give you access to the latest features immediately. However, we recommend using tagged releases for studies, which makes documenting and replicating your work much easier.

(quick-start-guide)=
# Quick Start Guide
```
Prerequisites: Python environment
Features: Minimal Aviary install, basic optimizers only
```

Aviary can be very quickly installed via a terminal. This version of Aviary only includes basic optimizers included in the `scipy` package, such as SLSQP.

```
pip install aviary
```

This will install the latest release of Aviary and all of its dependencies. This may take serval minutes. Verify your installation following [these steps](#verify-installation), and you're done! Keep in mind that many of the examples and most tests require optimizers that aren't included in this installation. To run those on your own machine, you will need to follow [this step](#install-pyoptsparse) from the Detailed Installation Guide.

## Updating Aviary
When a new version of Aviary is released, you can update your local copy by running the following command:

```
pip install --upgrade aviary
```

(detailed-guide)=
# Detailed Installation Guide
```
Prerequisites: Python environment (conda recommended), git (optional)
Features: Latest development version of Aviary with additional optimizers
```

This installation guide will show you how to install Aviary with the ability to grab latest development versions, as well as a guide on how to install `pyoptsparse` with additional optimizers.

(install-pyoptsparse)=
## Step 1: Install pyOptSparse
If you installed Aviary following the Quick Start Guide, you can "upgrade" your installation to use more optimizers by following this step. Once you have pyoptsparse installed, proceed to [verifying your installation](#verify-installation).

### Installing with conda
If you have a conda environment, then you can install pyOptSparse with access to the IPOPT optimizer. This is the most commonly used optimizer in Aviary, as it is generally more performant than SLSQP while also being free. Simply install `pyoptsparse` using conda:

```
conda install pyoptsparse
```

```
Note: There is currently an conflict caused by installing Aviary using pip *before* installing pyOptSparse with conda. Conda will overwrite the numpy version installed by pip with the latest available. This can cause problems, as Aviary does not yet support the latest versions of numpy (>=2.0). To resolve this issue, you may need need to re-install an older version of numpy:

conda install "numpy<2.0"
```

### Installing with pip, or if you have SNOPT
To install pyOptSparse without conda, you will need the [build_pyoptsparse](https://github.com/OpenMDAO/build_pyoptsparse) package to help build pyOptSparse correctly. It is possible to install pyOptSparse without this utility package, but that is outside the scope of this guide. 

First install `build_pyoptsparse`, then run it as a command:

```
pip install git+https://github.com/OpenMDAO/build_pyoptsparse
build_pyoptsparse
```

#### Adding SNOPT
[SNOPT](http://ccom.ucsd.edu/~optimizers/solvers/snopt/) is a proprietary, high-performance optimizer that is very good at solving large nonlinear problems and is used by many OpenMDAO users. SNOPT is only supported on Linux.

If you have a copy of SNOPT, instead run the following command to install `pyoptsparse` with both IPOPT and SNOPT available:

```
build_pyoptsparse -s /path_to_SNOPT_dir
```

## Step 2: Install Aviary
Grab the latest version of Aviary directly from the github repository, rather than from PyPi. If you don't have git, you can download the files from github, and install it using pip from inside the top-level Aviary directory (`pip install .`)

```
pip install git+https://github.com/OpenMDAO/Aviary@main
```

### Updating Aviary
To update your copy of Aviary to the latest development version, run this command:

```
pip install --upgrade git+https://github.com/OpenMDAO/Aviary@main
```

## Troubleshooting
There are several places this installation can fail, here are a couple troubleshooting steps to take. First, try running `build_pyoptsparse -v` to get a more verbose output.

*If you see an error for a missing command:*
```
ERROR: Required command swig not found.
```
Then try and install the missing package (in this case `swig`) using pip or conda. Swig is a commonly missing dependency.

*If you see an error where a conda or mamba command failed:*
```
subprocess.CalledProcessError: Command '['mamba', 'info', '--unsafe-channels']' returned non-zero exit status 109.
```
Try running `build_pyoptsparse -m` to disable use of mamba commands during installation. Adding `-e` will do the same for conda if your error specifically mentions `'conda'`.

(developers-guide)=
# Developer's Guide
```
Prerequisites: Python environment (conda recommended), git
Features: Editable install of latest version Aviary with additional optimizers, additional packages for development and testing
```
The Developer's Guide installs an editable version of Aviary source code, cloned from github, as well as additional Python packages needed for Aviary development and contributing code to the project.

## Step 1: Install pyOptSparse
The instructions for installing pyOptSparse are the same as the Detailed Installation Guide [here](#install-pyoptsparse).

## Step 2: Install Aviary
 It is recommended for developers to create their own fork of Aviary and clone that. In these instructions we will use the main [github repository](https://github.com/OpenMDAO/Aviary), if you are using a fork simply substitute the url with your own. Navigate to the folder where you'd like Aviary to be downloaded to, then run:

 ```
git clone https://github.com/OpenMDAO/Aviary
 ```

Next, we will install Aviary in "editable" mode, which means that you can make changes to the code inside this Aviary directory. The `[all]` tag will also install all optional dependencies - Aviary has several installation configurations you can use by adding the correct tag inside brackets after the period. The available tags are detailed [here](#optional-dependencies). From inside the top-level Aviary directory (it should have a file called `pyproject.toml`), run:

```
pip install -e .[all]
```

### Updating Aviary
To update your copy of Aviary to the latest development version, run this command from anywhere inside the Aviary repository folder:
```
git pull
```

(optional-dependencies)=
## Step 3: Install Additional Dependencies
There are many additional packages that are useful for developers, but not required for users. Select the ones that make sense for your use case. Once you have installed the packages you need, [verify your installation](#verify-installation).

You can automatically install these optional packages by specifying a tag inside brackets after "aviary" when pip installing, like so:
```
pip install aviary[<tag>]
```

The following tags are available:
- `docs`: installs [packages for building docs](#docs-packages)
- `dev`: installs the [packages for running tests](#tests-packages) and [contributing code](#contribution-packages)
- `all`: installs all optional packages listed in this section

(tests-packages)=
### Packages For Running Tests
In order to run Aviary's test suite, you will need the following packages:
- *testflo*
- *ambiance*
- *openaerostruct*

`Testflo` is the core package that automates testing. The other two packages, `ambiance` and `openaerostruct`, are needed to run example cases that incorporate them external subsystems. It is useful to be able to run these cases even if you have no interest in those analyses to ensure that the interface for external subsystems is working correctly.

(contribution-packages)=
### Packages For Contributing Code
To contribute code, you will need to follow Aviary's [contribution guidelines](../developer_guide/contributing_guidelines.md). This involves the use of additional packages.
- *pre-commit*

The `pre-commit` package has an additional install step after you get the package through `pip` or `conda` commands.

```
pip install pre-commit
pre-commit install
```

(docs-packages)=
### Packages For Building Docs
Several additional packages are needed to build a copy of the Aviary documentation locally.
- *jupyter-book*
- *itables*


(verify-installation)=
# Verify Your Installation
First, check that Aviary commands can be successfully executed.

```
aviary
```

You should get a printout similar to below:

```
usage: aviary [-h] [--version]  ...

aviary Command Line Tools

options:
  -h, --help            show this help message and exit
  --version             show version and exit

Tools:

    convert_aero_table  Converts FLOPS- or GASP-formatted aero data files into Aviary csv format.
    convert_engine      Converts FLOPS- or GASP-formatted engine decks into Aviary csv format FLOPS
                        decks are changed from column-delimited to csv format with added headers.
                        GASP decks are reorganized into column based csv. T4 is recovered through
                        calculation. Data points whose T4 exceeds T4max are removed.
    convert_prop_table  Converts GASP-formatted propeller map file into Aviary csv format.
    dashboard           Run the Dashboard tool
    draw_mission        Allows users to draw a mission profile for use in Aviary.
    fortran_to_aviary   Converts legacy Fortran input decks to Aviary csv based decks
    hangar              Allows users that pip installed Aviary to download models from the Aviary
                        hangar
    plot_drag_polar     Plot a Drag Polar Graph using a provided polar data csv input
    run_mission         Runs Aviary using a provided input deck
```

Next try running the installation test command:

```
aviary test_installation
```

The command should finish without raising any errors - warnings and other printouts are ok. If you don't get any errors, you are ready to use Aviary!

## Developer Verification
If you followed the Developer's Guide, then you also have access to testflo. To run testflo, run the following command from inside the top level of the Aviary repository. Be advised, running the full test suite may take a significant amount of time to run, on the order of thirty minutes on weaker machines such as laptops.

```
testflo .
```

The tests should begin running. You will see a series of characters printed to the screen, along with normal printouts from running Aviary cases. Periods are finished tests, "S" represents a skipped tests. Skipped tests are not a concern. Tests are flagged to be skipped for a variety of reasons, and do not mean there is a problem with your installation.

If you are missing some optional packages, those tests will simply be skipped, so you should not be seeing any failures. If you run into an MPI error, you can add the `--nompi` option to the testflo command run.

A successful test run should look like the following. The exact number of tests ran will vary as Aviary development continues, but there should be not be any failed tests.

```
OK

Passed:  1392
Failed:  0
Skipped: 10


Ran 1402 tests using 208 processes
Wall clock time:   00:06:6.20
```

You can also run the benchmark tests, which are a suite of full aircraft optimization problems. These are separate from the test suite called with testflo to allow developers to only run the subset of tests they need to save runtime. To run the benchmark tests, execute the `run_all_benchmarks.py` script located in the `aviary` directory in the repository. 

```
python aviary/run_all_benchmarks.py
```

The final printout should look like this:

```
OK

Passed:  13
Failed:  0
Skipped: 1


Ran 14 tests using 208 processes
Wall clock time:   00:02:45.00
```