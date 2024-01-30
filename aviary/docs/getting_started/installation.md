# Installation

## Quick start installation

The simplest installation method for users is to install via pip.
Once you have cloned the Aviary repo, change directories into the top-level Aviary folder (not within the `aviary` folder) and run the following command:

    pip install .

If you also want to install all packages used for the Aviary tests _and_ external subsystem examples, you can instead run:

    pip install .[all]

If you are a developer and plan to modify parts of the Aviary code, install in an "editable mode" with ``pip``:

    pip install -e .

This installs the package in the current environment such that changes to the Python code don't require re-installation.
This command should be performed while in the folder containing ``setup.py``.

```{note}
You can do this editable installation with any of the `[test]` or `[all]` options as well.
```

```{note}
You can install the optional package [pyOptSparse by following the instructions here](https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/install.html). If you do not need the SNOPT optimizer, installing pyOptSparse is as simple as running `conda install -c conda-forge pyoptsparse`.
```

## Installation on Linux for Developers

As an example, let us do a step-by-step installation from scratch on your Linux operating system. We assume that you have [Anaconda](https://www.anaconda.com/distribution) and your new environment will be built on top of it. In this section, we assume you are a developer and hence you will need developer's versions of OpenMDAO and Dymos. As a result, you will need [Git](https://git-scm.com/). We also assume that you have a bash shell.

We will be installing some Python packages from source in this part of the docs.
Depending on the system you're installing on, the OpenMDAO repositories might require a password-protected SSH key.
This means that users need to [generate a new SSH key and add it to the ssh-agent](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux) and then [add the new SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account?platform=linux).
We assume you already have access to the OpenMDAO repos as shown below or that you've already added an SSH key.

### Preparing your Anaconda Environment

On the Linux system, log in to your account and create your working directory.
For this doc we will use `workspace`:

```
$ mkdir ~/workspace
```

Then `cd` to this newly created directory. Please note that `~/workspace` is just an example. In fact, you can install Aviary wherever you want on your system. Save the following as a file named `aviary-linux-dev-modified.yml` to this directory:

```
name: av1
channels:
  - defaults
dependencies:
  - python=3
  - numpy=1
  - scipy=1
  - matplotlib
  - pandas
  - jupyter
  - pip
  - pip:
    - parameterized
    - testflo
    - jupyter-book
    - mdolab-baseclasses
    - sqlitedict
    - f90nml
    - bokeh
```

In this file, the `name` can be anything you like. The version of python is not limited to 3.9, but we recommend that you stay with this version because it is the version that we use to fully test Aviary and that it is required for some packages later on. For example, if you are going to add `OpenVSP` to your environment, you will find that you need this version.

In the list, we see the popular Python packages for scientific computations: `numpy`, `scipy`, `matplotlib` and `pandas`. Aviary follows a standard source code formatting convention. `autopep8` provides an easy way to check your source code for this purpose. `jupyter` and `jupyter-book` are used to create Aviary manual. `parameterized` and `testflo` are for Aviary testing. Aviary uses a lot of packages developed by [MDOLab](https://mdolab.engin.umich.edu/). So, we want to include its base classes. OpenMDAO records data in SQLite database and that is what `sqlitedict` comes for. `f90nml` is A Python module and command line tool for parsing Fortran namelist files. `bokeh` is an interactive visualization library for modern web browsers. It is needed to generate Aviary output (traj_results_report.html).

Since we are going to depend on `OpenMDAO` and `dymos`, we could have included them in the `pip` list. We leave them out because we will install the developer version later. In this way, we will get the latest working copies that Aviary depends on. But we do not intend to make changes to them.

[pre-commit](https://pre-commit.com/) and [autopep8 formatter](https://pypi.org/project/autopep8/) are additionally required for developers who wish to contribute to the Aviary repository. Read our [coding standards](../developer_guide/coding_standards.md) for more information.

Now, run create your new conda environment using this `.yml` file:

```
$ conda env create -n av1 --file aviary-linux-dev-modified.yml
```

Suppose everything runs smoothly. You have a new conda environment `av1`. You can start your conda environment:

```
$ conda activate av1
```

### Installing Additional Dependencies

Aviary can run in MPI. So, let us do:

```
$ conda install -c conda-forge mpi4py petsc4py
```

Download developer version of `OpenMDAO`:

```
$ cd ~/workspace
$ git clone git@github.com:OpenMDAO/OpenMDAO.git
```

You have a new subdirectory `workspace/OpenMDAO`. You are not expected to modify source code of OpenMDAO, but you want to keep up with the latests version of it. The best way to do is to install OpenMDAO in developer mode. This removes the need to reinstall OpenMDAO after changes are made. Go to this directory where you see a file `setup.py`. Run 

```
$ cd OpenMDAO
$ pip install -e .
```

You should see something like the following:

```
Successfully installed networkx-3.1 openmdao-3.29.1.dev0
```

Now, let us install `dymos` in a similar way:

```
$ cd ~/workspace/
$ git clone git@github.com:OpenMDAO/dymos.git
$ cd dymos
$ pip install -e .
```

You should see something like the following:

```
Successfully installed dymos-1.9.2.dev0
```

### Installing pyOptSparse

Next, we will install `pyoptsparse`.
If you want to easily install and use pyOptSparse, follow the [installation instructions on the pyOptSparse docs](https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/install.html).
Specifically, if you do not need the SNOPT optimizer and want to run Aviary with IPOPT, you can install pyOptSparse using the following command:

```
conda install -c conda-forge pyoptsparse
```

The OpenMDAO team provides a [`build_pyoptsparse`](https://github.com/OpenMDAO/build_pyoptsparse) package to help users install MDO Lab's pyOptSparse, optionally including the `SNOPT` and `IPOPT` optimizers.

This process depends on certain libraries.
One of them is the [Lapack](https://www.netlib.org/lapack/) Fortran library.
If you don't have `Lapack`, you can either [build it from source](https://github.com/Reference-LAPACK/lapack) or try one of the [prebuilt binaries](https://www.netlib.org/lapack/archives/).
We are assuming you have Lapack installed correctly.

Now do the following:

```
$ cd ~/workspace/
$ git clone git@github.com:OpenMDAO/build_pyoptsparse.git
$ python -m pip install ./build_pyoptsparse
```

```{note}
`SNOPT` is a commercial optimizer that is free for academic use and available for puchase for commercial use. Users must obtain it themselves.
```

Assuming you have the `SNOPT` source code already, copy it to the `workspace` directory. 
Run:

```
$ build_pyoptsparse -d -s ~/workspace/SNOPT_Source/
```

Note that you should provide the absolute -- not relative -- path to your SNOPT source files.
If successful, you should get the following:

```
---------------------- The pyOptSparse build is complete ----------------------
NOTE: Set the following environment variable before using this installation:

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

Otherwise, you may encounter errors such as:
 "pyOptSparse Error: There was an error importing the compiled IPOPT module"

----------------------------------- SUCCESS! ----------------------------------
```

So, let us add this environment variable to your bash shell:

```
$ export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
```

If you don't have the Lapack package and you don't plan to build it by yourself, you can build `pyOptSparse` with `SNOPT` by adding an option `--no-ipopt` to your `build_pyoptsparse` command.

Alternatively, if `build_pyoptsparse` fails again and you have `SNOPT` source code, you still can build `pyOptSparse` with `SNOPT` by directly building pyOptSparse.
First, clone the `pyOptSparse` repository:

```
$ cd ~/workspace/
$ git clone https://github.com/mdolab/pyoptsparse
```

You will see a `pySNOPT` subdirectory in it. Go to your `SNOPT` source folder and copy all Fortran source code files into this directory:

```
$ cp -a * ~/workspace/pyoptsparse/pyoptsparse/pySNOPT/source/
```

You are ready to install `pyoptsparse` (with `SNOPT`):

```
$ cd ~/workspace/pyoptsparse/
$ pip install -e .
```

You should see something like the following:

```
Successfully built pyoptsparse
Installing collected packages: pyoptsparse
Successfully installed pyoptsparse-2.10.1
```

### Installing Aviary and Running Tests

Now, we are ready to install Aviary. Assuming that you will become a contributor sooner or later, we want to install a copy from the main source. (You will need a GitHub account for this) Let us open `https://github.com/openMDAO/om-aviary/` in a web browser and click [fork](https://github.com/OpenMDAO/Aviary/fork) on the top-right corner. You then have created your own copy of Aviary on GitHub website. Now we create a copy on your local drive (supposing `USER_ID` is your GitHub account ID):

```
$ cd ~/workspace
$ git clone git@github.com:USER_ID/Aviary.git
$ cd Aviary
$ pip install -e .
```

When it is done, let us run test:

```
% cd ../..
% testflo .
```

If you run into an MPI error, you can add `--nompi` option to `testflo` command run. If everything runs, you will get something like the following:

```
Passed:  875
Failed:  0
Skipped: 3


Ran 878 tests using 1 processes
Wall clock time:   00:02:47.31
```

To find which tests are skipped, we can add `--show_skipped` option:

```
The following tests were skipped:
test_conventional_problem.py:TestConventionalProblem.test_conventional_problem_ipopt
test_conventional_problem.py:TestConventionalProblem.test_conventional_problem_snopt
test_cruise.py:TestCruise.test_cruise_result
```

Actually, those three tests were skipped on purpose. Depending on what optimizers are installed, the number of skipped tests may be different. Your installation is successful.

To see the test name before each unit test and push all outputs to standard outputs:

```
$ testflo -s --pre_announce .
```

Run `testflo --help` to see other options. For example, we have a set of longer tests (called bench tests) that perform full trajectory optimizations. To run all the bench tests, you can run this:

```
$ cd ~/Aviary/aviary
$ python run_all_benchmarks.py
```

If you want to test a particular case (e.g. `test_simplified_takeoff.py`):

```
$ testflo test_simplified_takeoff.py
```

```{note}
Installing Aviary via pip here does not install all packages needed for external subsystems.
For example, if you're using [OpenVSP](https://openvsp.org/), [pyCycle](https://github.com/OpenMDAO/pyCycle), or other tools outside of "core Aviary", you would need to install those separately.
```
