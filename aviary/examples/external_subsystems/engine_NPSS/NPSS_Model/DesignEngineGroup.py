import numpy as np
import openmdao.api as om
import subprocess as subprocess
from openmdao.utils.file_wrap import FileParser
from aviary.subsystems.propulsion.engine_sizing import SizeEngine
from aviary.utils.functions import get_path
import time
import os

from aviary.examples.external_subsystems.engine_NPSS.engine_variables import Aircraft, Dynamic

# Number of points in NPSS model deck.
# Will need to be updated if the size of the deck changes.
vec_size = 72


class NPSSExternalCodeComp(om.ExternalCodeComp):
    def setup(self):
        self.add_input('Alt_DES', val=0.0, units='ft')
        self.add_input('MN_DES', val=0.0, units=None)
        self.add_input('W_DES', val=240.0, units='lbm/s')

        self.add_output('Fn_SLS', val=1.0, units='lbf')
        self.add_output('Wf_training_data', val=np.ones(vec_size), units='lbm/s')
        self.add_output('thrust_training_data', val=np.ones(vec_size), units='lbf')
        self.add_output('thrustmax_training_data', val=np.ones(vec_size), units='lbf')

        self.input_file = get_path(
            './examples/external_subsystems/engine_NPSS/NPSS_Model/Design_files/input.int')
        self.output_file = get_path(
            './examples/external_subsystems/engine_NPSS/NPSS_Model/Design_files/output.int')

        self.options['external_input_files'] = [self.input_file]
        self.options['external_output_files'] = [self.output_file]

        # self.options['command'] = ['c:/NPSS.nt.ver32_VC14_64/bin/npss.nt.exe','turbojet.run']

        run_location = get_path(
            './examples/external_subsystems/engine_NPSS/NPSS_Model/turbojet.run')
        self.options['command'] = ['runnpss', run_location]

    def setup_partials(self):
        # this external code does not provide derivatives, use finite difference
        # Note: step size should be larger than NPSS solver tolerance
        self.declare_partials(of='*', wrt='*', method='fd', step=1e-3)

    def compute(self, inputs, outputs):
        Alt_DES = inputs['Alt_DES']
        MN_DES = inputs['MN_DES']
        W_DES = inputs['W_DES']

        # generate the input file for the external code
        with open(self.input_file, 'w') as input_file:
            input_file.write('Alt_DES  = %.16f ;\nMN_DES  = %.16f ;\nW_DES  = %.16f ;'
                             % (Alt_DES, MN_DES, W_DES))

        # the parent compute function actually runs the external code
        super().compute(inputs, outputs)

        # parse the output file from the external code and set the value of f_xy
        parser = FileParser()
        parser.set_file(self.output_file)

        parser.mark_anchor("Fn_SLS")
        Fn_SLS = float(parser.transfer_var(0, 3))
        outputs['Fn_SLS'] = Fn_SLS

        parser.mark_anchor("Wf_training_data")
        parser.set_delimiters(', ')
        Wf_training_data_raw = parser.transfer_array(0, 4, 0, vec_size*3)
        Wf_training_data_snip = [
            a for a in Wf_training_data_raw if a.replace('.', '').isnumeric()]
        Wf_training_data = np.array(Wf_training_data_snip, float)
        outputs['Wf_training_data'] = Wf_training_data

        parser.mark_anchor("thrust_training_data")
        parser.set_delimiters(', ')
        thrust_training_data_raw = parser.transfer_array(0, 4, 0, vec_size*3)
        thrust_training_data_snip = [
            a for a in thrust_training_data_raw if a.replace('.', '').isnumeric()]
        thrust_training_data = np.array(thrust_training_data_snip, float)
        outputs['thrust_training_data'] = thrust_training_data

        parser.mark_anchor("thrustmax_training_data")
        parser.set_delimiters(', ')
        thrustmax_training_data_raw = parser.transfer_array(0, 4, 0, vec_size*3)
        thrustmax_training_data_snip = [
            a for a in thrustmax_training_data_raw if a.replace('.', '').isnumeric()]
        thrustmax_training_data = np.array(thrustmax_training_data_snip, float)
        outputs['thrustmax_training_data'] = thrustmax_training_data
        # print(thrustmax_training_data)


class DesignEngineGroup(om.Group):
    def setup(self):
        self.add_subsystem('DESIGN', NPSSExternalCodeComp(),
                           promotes_inputs=[('W_DES', Aircraft.Engine.DESIGN_MASS_FLOW)],
                           # promotes_outputs=[('Fn_SLS', Aircraft.Engine.SCALED_SLS_THRUST), ('thrust_training_data', Dynamic.Mission.THRUST+'_train'),
                           #                   ('thrustmax_training_data', Dynamic.Mission.THRUST_MAX+'_train'), ('Wf_training_data', 'Wf_td')])
                           promotes_outputs=[('Fn_SLS', Aircraft.Engine.SCALED_SLS_THRUST), ('thrust_training_data', 'Fn_train'),
                                             ('thrustmax_training_data', 'Fn_max_train'), ('Wf_training_data', 'Wf_td')])

        self.add_subsystem('neg_Wf', om.ExecComp('y=-x',
                                                 x={'val': np.ones(
                                                     vec_size), 'units': 'lbm/s'},
                                                 y={'val': np.ones(vec_size), 'units': 'lbm/s'}),
                           promotes_inputs=[('x', 'Wf_td')],
                           promotes_outputs=[('y', 'Wf_inv_train')])

    def configure(self):
        self.set_input_defaults(Aircraft.Engine.DESIGN_MASS_FLOW, 240.0, units='lbm/s')
        self.set_input_defaults('DESIGN.Alt_DES', 0.0, units='ft')
        self.set_input_defaults('DESIGN.MN_DES', 0.0)
        super().configure()


if __name__ == "__main__":
    import openmdao.api as om

    SNOPT_EN = False
    prob = om.Problem()
    model = prob.model

    # add system model
    model.add_subsystem('p', DesignEngineGroup())

    # Add solver
    newton = model.nonlinear_solver = om.NewtonSolver()
    newton.options['atol'] = 1e-3
    newton.options['rtol'] = 1e-4
    newton.options['iprint'] = 2
    newton.options['debug_print'] = False
    newton.options['maxiter'] = 50
    newton.options['solve_subsystems'] = True
    newton.options['max_sub_solves'] = 100
    newton.options['err_on_non_converge'] = False
    newton.linesearch = om.BoundsEnforceLS()
    # newton.linesearch.options['maxiter'] = 1
    newton.linesearch.options['bound_enforcement'] = 'scalar'
    newton.linesearch.options['iprint'] = 2
    newton.linesearch.options['print_bound_enforce'] = False

    # model.nonlinear_solver = om.DirectSolver()

    model.linear_solver = om.DirectSolver(assemble_jac=True)

    prob.setup()

    print('\n\nSTART MDP\n\n')
    st = time.time()
    # use run driver if optimizing.
    # prob.run_driver()
    # use run model if only using balances.
    prob.run_model()
    run_time = time.time() - st
    print('\n\nEND MDP \n run time : ', run_time, '\n\n')

    # print the output
    # print('opt fail : ',prob.driver.fail )
    print('Fn_SLS : ', prob.get_val('p.'+Aircraft.Engine.SCALED_SLS_THRUST))
    print('Fn_train : ', prob.get_val('p.'+Dynamic.Mission.THRUST+'_train'))
    print('Fn_max : ', prob.get_val('p.'+Dynamic.Mission.THRUST_MAX+'_train'))
    print('Wf_td :', prob.get_val('p.Wf_td'))
