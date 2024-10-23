import numpy as np
import openmdao.api as om
import subprocess as subprocess

from openmdao.utils.file_wrap import FileParser

from aviary.examples.external_subsystems.engine_NPSS.engine_variables import Aircraft
from aviary.utils.functions import get_path


class NPSSExternalCodeComp(om.ExternalCodeComp):
    """
    Component that wraps NPSS engine model
    """

    def initialize(self):
        self.options.declare('vec_size', default=72, types=int,
                             desc='number of points in NPSS model deck. Will need to be updated if size of deck changes')

    def setup(self):
        vec_size = self.options['vec_size']
        self.add_input('Alt_DES', val=0.0, units='ft', desc='design altitude')
        self.add_input('MN_DES', val=0.0, units=None, desc='design Mach number')
        self.add_input('W_DES', val=240.0, units='lbm/s', desc='design mass flow')

        self.add_output('Fn_SLS', val=1.0, units='lbf',
                        desc='net thrust at sea-level-static conditions')
        self.add_output('Wf_training_data', val=np.ones(vec_size),
                        units='lbm/s', desc='fuel flow training data')
        self.add_output('thrust_training_data', val=np.ones(
            vec_size), units='lbf', desc='thrust training data')
        self.add_output('thrustmax_training_data', val=np.ones(vec_size),
                        units='lbf', desc='maximum thrust training data')

        self.input_file = get_path(
            './examples/external_subsystems/engine_NPSS/NPSS_Model/Design_files/input.int')
        self.output_file = get_path(
            './examples/external_subsystems/engine_NPSS/NPSS_Model/Design_files/output.int')

        self.options['external_input_files'] = [self.input_file]
        self.options['external_output_files'] = [self.output_file]

        run_location = get_path(
            './examples/external_subsystems/engine_NPSS/NPSS_Model/turbojet.run')
        engine_location = get_path(
            './examples/external_subsystems/engine_NPSS/NPSS_Model/')
        engine_location = str(engine_location)

        run_command = ['runnpss', run_location, '-D ENG_PATH='+engine_location]
        self.options['command'] = run_command

    def setup_partials(self):
        # this external code does not provide derivatives, use finite difference
        # Note: step size should be larger than NPSS solver tolerance
        self.declare_partials(of='*', wrt='*', method='fd', step=1e-6)

    def compute(self, inputs, outputs):
        Alt_DES = inputs['Alt_DES']
        MN_DES = inputs['MN_DES']
        W_DES = inputs['W_DES']
        vec_size = self.options['vec_size']

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
    """
    Group that contains NPSSExternalCodeComp and component to calculate negative fuel flow rate
    """

    def initialize(self):
        self.options.declare('vec_size', default=72, types=int,
                             desc='number of points in NPSS model deck. Will need to be updated if size of deck changes')

    def setup(self):
        vec_size = self.options['vec_size']
        self.add_subsystem('DESIGN', NPSSExternalCodeComp(vec_size=vec_size),
                           promotes_inputs=[('W_DES', Aircraft.Engine.DESIGN_MASS_FLOW)],
                           promotes_outputs=[('Fn_SLS', Aircraft.Engine.SCALED_SLS_THRUST), ('thrust_training_data', 'Fn_train'),
                                             ('thrustmax_training_data', 'Fn_max_train'), ('Wf_training_data', 'Wf_td')])

        self.add_subsystem('negative_fuel_rate',
                           om.ExecComp('y=-x',
                                       x={'val': np.ones(vec_size), 'units': 'lbm/s'},
                                       y={'val': np.ones(vec_size), 'units': 'lbm/s'},
                                       has_diag_partials=True,),
                           promotes_inputs=[('x', 'Wf_td')],
                           promotes_outputs=[('y', 'Wf_inv_train')])

    def configure(self):
        self.set_input_defaults(Aircraft.Engine.DESIGN_MASS_FLOW, 240.0, units='lbm/s')
        self.set_input_defaults('DESIGN.Alt_DES', 0.0, units='ft')
        self.set_input_defaults('DESIGN.MN_DES', 0.0)
        super().configure()
