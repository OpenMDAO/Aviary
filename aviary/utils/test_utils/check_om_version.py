import openmdao.api as om


class CheckForOMSubmodelFix():
    def __bool__(self):
        if hasattr(om.SubmodelComp, '_get_output_kwargs'):
            return True
        else:
            return False
    __nonzero__ = __bool__

    def __str__(self):
        return str(bool(self))


if __name__ == '__main__':
    print(CheckForOMSubmodelFix())
