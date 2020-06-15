# Copyright (C) 2015-2020 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.base import NonlinearTimeDependentReductionMethod


# Base class containing the interface of a projection based ROM
# for elliptic coercive problems.
def NonlinearDynamicReductionMethod(DifferentialProblemReductionMethod_DerivedClass):

    NonlinearDynamicReductionMethod_Base = NonlinearTimeDependentReductionMethod(DifferentialProblemReductionMethod_DerivedClass)

    class NonlinearDynamicReductionMethod_Class(NonlinearDynamicReductionMethod_Base):
        pass

    # return value (a class) for the decorator
    return NonlinearDynamicReductionMethod_Class
