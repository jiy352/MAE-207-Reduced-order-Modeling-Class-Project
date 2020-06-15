# Copyright (C) 2015-2020 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
from dolfin import *
import os
from rbnics.backends import product, sum
# from rbnics.problems.base import LinearTimeDependentProblem

from rbnics.problems.base import NonlinearTimeDependentProblem
from rbnics.problems.nonlinear_elliptic import NonlinearEllipticProblem
# from rbnics.backends import product, sum

NonlinearParabolicProblem_Base = NonlinearTimeDependentProblem(NonlinearEllipticProblem)


class NonlinearDynamicProblem(NonlinearParabolicProblem_Base):

    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call to parent
        NonlinearParabolicProblem_Base.__init__(self, V, **kwargs)
        self.V = V
        # Form names for parabolic problems
        self.terms.append("m")
        self.terms_order.update({"m": 2})

    class ProblemSolver(NonlinearParabolicProblem_Base.ProblemSolver):
        def residual_eval(self, t, solution, solution_dot):
            problem = self.problem
            
            assembled_operator = dict()
            assembled_operator["m"] = sum(product(problem.compute_theta("m"), problem.operator["m"]))
            assembled_operator["a"] = sum(product(problem.compute_theta("a"), problem.operator["a"]))
            assembled_operator["c"] = sum(product(problem.compute_theta("c"), problem.operator["c"]))
            assembled_operator["f"] = sum(product(problem.compute_theta("f"), problem.operator["f"]))
            if os.path.isfile("solution_dot_prev_f.h5") == 0:
                solution_dot_diff = solution_dot - solution_dot
            else:
                # Read `f` from a file:
                solution_dot_prev = Function(self.V)
                fFile = HDF5File(MPI.comm_world,"solution_dot_prev_f.h5","r")
                fFile.read(solution_dot_prev,"/solution_dot_prev_f")
                fFile.close()
                solution_dot_diff = solution_dot - solution_dot_prev

            # Write `f` to a file:
            fFile = HDF5File(MPI.comm_world,"solution_dot_prev_f.h5","w")
            fFile.write(solution_dot,"/solution_dot_prev_f")
            fFile.close()
            
            return (assembled_operator["m"] * (solution_dot_diff) / 0.02
                    + assembled_operator["a"] * solution
                    + assembled_operator["c"]
                    - assembled_operator["f"])

        def jacobian_eval(self, t, solution, solution_dot, solution_dot_coefficient):
            problem = self.problem

            if os.path.isfile("solution_dot_coeff_prev_f.h5") == 0:
                solution_dot_coeff_diff = solution_dot_coefficient - solution_dot_coefficient
            else:
                # Read `f` from a file:
                solution_dot_coeff_prev = Function(self.V)
                fFile = HDF5File(MPI.comm_world,"solution_dot_coeff_prev_f.h5","r")
                fFile.read(solution_dot_coeff_prev,"/solution_dot_coeff_prev_f")
                fFile.close()
                solution_dot_coeff_diff = solution_dot_coefficient - solution_dot_coeff_prev

            # Write `f` to a file:
            fFile = HDF5File(MPI.comm_world,"solution_dot_coeff_prev_f.h5","w")
            fFile.write(solution_dot_coefficient,"/solution_dot_coeff_prev_f")
            fFile.close()

            assembled_operator = dict()
            assembled_operator["m"] = sum(product(problem.compute_theta("m"), problem.operator["m"]))
            assembled_operator["a"] = sum(product(problem.compute_theta("a"), problem.operator["a"]))
            assembled_operator["dc"] = sum(product(problem.compute_theta("dc"), problem.operator["dc"]))
            return (assembled_operator["m"] * solution_dot_coeff_diff
                    + assembled_operator["a"]
                    + assembled_operator["dc"])
