# Optimization Algorithms

Optimizers are what numerically solve the aircraft design problem that we pose to Aviary.
Within the context of Aviary and broader OpenMDAO, optimizers are a type of driver that repeatedly query the aircraft and trajectory models.

When we say "optimizer", this is a distinct idea from a "solver" in the context of OpenMDAO.
A solver is a component that numerically solves a system of equations, e.g. a Newton solver or a linear solver.
An optimizer is a driver that finds the optimal values of the design variables that minimize or maximize the objective function.

For more information on OpenMDAO optimization drivers, see the [OpenMDAO documentation](https://openmdao.org/newdocs/versions/latest/features/building_blocks/drivers/index.html).
For a basic introduction to what an optimization problem is and how it is solved, see the [Practical MDO page on "Basic optimization problem](https://openmdao.github.io/PracticalMDO/Notebooks/Optimization/basic_opt_problem_formulation.html).

## Available Algorithms that work with Aviary

Aviary is designed to work well with gradient-based optimizers.
This means that they require the gradient of the objective function and constraints with respect to the design variables.
Gradient-free optimizers are available through OpenMDAO and pyOptSparse, but they are not recommended for use with Aviary due to the optimization problem complexity.

SNOPT is the recommended optimizer for use with Aviary but is not available for free commercial use.
IPOPT and SLSQP are open-source optimizers that are available for commercial use.

### SNOPT

[SNOPT](https://ccom.ucsd.edu/~optimizers/solvers/snopt/) is a sequential quadratic programming (SQP) algorithm that is available in OpenMDAO through [pyOptSparse](https://github.com/mdolab/pyoptsparse).
SNOPT is a commercial software package and has a high cost for commercial use but it is available for free for academic use.
For more information on SNOPT, see the [SNOPT documentation](https://web.stanford.edu/group/SOL/guides/sndoc7.pdf).

### IPOPT

IPOPT is an interior point optimizer that is available in OpenMDAO through [pyOptSparse](https://github.com/mdolab/pyoptsparse).
IPOPT is an open-source software package and will be installed automatically when you install pyOptSparse when using the `conda` [package manager as detailed here](https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/install.html#conda).
For more information on IPOPT, see the [IPOPT documentation](https://coin-or.github.io/Ipopt/).

### SLSQP

[SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html) is a sequential least squares programming algorithm that is available in OpenMDAO through Scipy.
SLSQP is an open-source software package and is bundled with Scipy so it requires no additional packages.
For more information on SLSQP, see the [SLSQP documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html).