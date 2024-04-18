import numpy as np
import openmdao.api as om


def build_submodel():
    p = om.Problem()
    supmodel = om.Group()
    supmodel.add_subsystem('supComp', om.ExecComp('diameter = r * theta'),
                           promotes_inputs=['*'],
                           promotes_outputs=['*', ('diameter', 'aircraft:fuselage:diameter')])

    subprob1 = om.Problem()
    submodel1 = subprob1.model.add_subsystem('submodel1', om.Group(), promotes=['*'])

    submodel1.add_subsystem('x', om.ExecComp('x = diameter * 2 * r * theta'),
                            promotes=['*', ('diameter', 'aircraft:fuselage:diameter')])
    submodel1.add_subsystem('y', om.ExecComp('y = mass * donkey_kong'), promotes=[
                            '*', ('mass', 'dynamic:mission:mass'), ('donkey_kong', 'aircraft:engine:donkey_kong')])

    p.model.add_subsystem('supModel', supmodel, promotes_inputs=['*'],
                          promotes_outputs=['*'])

    submodel = om.SubmodelComp(problem=subprob1, inputs=[
                               '*'], outputs=['*'], do_coloring=False)
    p.model.add_subsystem('sub1', submodel,
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])

    p.model.set_input_defaults('r', 1.25)
    p.model.set_input_defaults('theta', np.pi)

    p.setup(force_alloc_complex=True)

    p.set_val('r', 1.25)
    p.set_val('theta', 0.5)
    p.set_val('dynamic:mission:mass', 2.0)
    p.set_val('aircraft:engine:donkey_kong', 3.0)
    p.set_val('aircraft:fuselage:diameter', 3.5)

    return p


class CheckForOMSubmodelFix():
    def __bool__(self):
        p = build_submodel()
        p.run_model()

        if p.get_val('y') != 2.0 * 3.0:
            return False
        else:
            return True
    __nonzero__ = __bool__
