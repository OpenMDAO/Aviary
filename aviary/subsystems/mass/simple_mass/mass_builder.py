from aviary.interface.utils.markdown_utils import write_markdown_variable_table
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.subsystems.mass.mass_builder import MassBuilderBase
from aviary.subsystems.mass.simple_mass.mass_premission import MassPremission
from aviary.variable_info.variables import Aircraft

"""

Define subsystem builder for Aviary core mass.

Classes
--------------------------------------------------------------------------------------------------

MassBuilderBase: the interface for a mass subsystem builder. **Not sure how necessary this is for
                 my work right now, but wanted to include it as a just in case. I basically copied
                 it over from the mass_builder.py under the mass subsystems folder in Aviary github.

StructureMassBuilder: the interface for Aviary's core mass builder -- in this case, 
                 the core mass builder will work for wing and fuselage mass calculations
                 will be updated as more mass calculations are added

"""

_default_name = 'mass'

class MassBuilderBase(SubsystemBuilderBase):
    """
    Base mass builder
    
    This class is basically copied line by line from the mass subsystems folder
    ** Ask Jason if this is even necessary. 
    
    """

    def __init__(self, name=None, meta_data=None):
       if name is None:
           name = _default_name
    
       super().__init__(name=name, meta_data=meta_data)
    
    def mission_inputs(self, **kwargs):
       return ['*']
    
    def mission_outputs(self, **kwargs):
       return ['*']

class StructureMassBuilder(MassBuilderBase):
    """
    Core mass subsystem builder

    Unlike the CoreMassBuilder on the github under the mass subsystems folder, 
    I am not including the __init__'s, since I don't have any FLOPS or GASP 
    dependence in my mass calculations at the moment; the math is essentially 
    hard coded from my calculations right now.

    """

    def build_pre_mission(self, aviary_inputs):
        return MassPremission # See the commented line above in the imports

    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        super().build_mission(num_nodes, aviary_inputs)

    def report(self, prob, reports_folder, **kwargs):
         """
        Generate the report for Aviary core mass

        Parameters
        ----------
        prob : AviaryProblem
            The AviaryProblem that will be used to generate the report
        reports_folder : Path
            Location of the subsystems_report folder this report will be placed in

        * This comment is copied from the mass subsystems folder * 
            
        """
         
         filename = self.name + '.md'
         filepath = reports_folder / filename

         # Ask Jason about how I should format this
         outputs = [
             Aircraft.Wing.MASS,
             Aircraft.HorizontalTail.MASS,
             Aircraft.VerticalTail.MASS,
             Aircraft.Fuselage.MASS,
             'structure_mass'
         ]

         with open(filepath, mode='w') as f:
             method = self.code_origin.value + '-derived relations' # Ask Jason about this too since I don't have FLOPS or GASP for code_origin_value
             f.write(f'# Mass estimation: {method}')
             write_markdown_variable_table(f, prob, outputs, self.meta_data)

    