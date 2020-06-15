# Copyright (C) 2015-2020 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.base import DifferentialProblemReductionMethod, NonlinearPODGalerkinReduction
from rbnics.utils.decorators import ReductionMethodFor
from problems import NonlinearDynamicProblem
from .nonlinear_dynamic_reduction_method import NonlinearDynamicReductionMethod

NonlinearDynamicPODGalerkinReduction_Base = NonlinearPODGalerkinReduction(
    NonlinearDynamicReductionMethod(DifferentialProblemReductionMethod))


@ReductionMethodFor(NonlinearDynamicProblem, "PODGalerkin")
class NonlinearDynamicPODGalerkinReduction(NonlinearDynamicPODGalerkinReduction_Base):
    pass
