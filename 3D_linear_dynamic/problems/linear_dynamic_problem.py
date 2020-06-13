# Copyright (C) 2015-2020 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base import LinearProblem, ParametrizedDifferentialProblem
from rbnics.backends import product, sum, transpose

LinearDynamicProblem_Base = LinearProblem(ParametrizedDifferentialProblem)


class LinearDynamicProblem(LinearDynamicProblem_Base):
    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call to parent
        LinearDynamicProblem_Base.__init__(self, V, **kwargs)

        # Form names for LinearDynamic problems
        self.terms = ["a", "f", "s"]
        self.terms_order = {"a": 2, "f": 1, "s": 1}
        self.components = ["u"]

    # Perform a truth solve
    class ProblemSolver(LinearDynamicProblem_Base.ProblemSolver):
        def matrix_eval(self):
            problem = self.problem
            return sum(product(problem.compute_theta("a"), problem.operator["a"]))

        def vector_eval(self):
            problem = self.problem
            return sum(product(problem.compute_theta("f"), problem.operator["f"]))
    # Perform a truth evaluation of the output
    def _compute_output(self):
        self._output = transpose(self._solution) * sum(product(self.compute_theta("s"), self.operator["s"]))