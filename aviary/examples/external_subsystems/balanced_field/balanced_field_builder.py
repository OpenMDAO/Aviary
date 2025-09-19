import openmdao.api as om

import aviary.api as av
from aviary.api import Mission
from aviary.examples.external_subsystems.balanced_field.balanced_field_submodel import (
    create_balance_field_subprob,
)
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase


class BalancedFieldBuilder(SubsystemBuilderBase):

    def build_post_mission(
        self, aviary_inputs, phase_info=None, phase_mission_bus_lengths=None, **kwargs
    ):
        return create_balance_field_subprob(aviary_inputs)


if __name__ == '__main__':
    unittest.main()
