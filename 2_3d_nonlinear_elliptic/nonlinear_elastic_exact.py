# Copyright (C) 2015-2020 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from rbnics import *


@ExactParametrizedFunctions()
class NonlinearElliptic(NonlinearEllipticProblem):

    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        NonlinearEllipticProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        self.du = TrialFunction(V)
        self.u = self._solution
        self.u_prev = self._solution_cache
        self.v = TestFunction(V)
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
        # Store the forcing term expression
        # self.f = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", element=self.V.ufl_element())

        self.f = Constant((0.0, 0.0, -1.))
        self.b = Constant((0.0, 0.0, -1.))
        self.E = 1.0e6
        self.nu = 0.3
        self.lambda_1 = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.lambda_2 = self.E / (2.0 * (1.0 + self.nu))
        self.I = Identity(3)
        # Customize nonlinear solver parameters
        self._nonlinear_solver_parameters.update({
            "linear_solver": "mumps",
            "maximum_iterations": 50,
            "report": True
        })

    # Return custom problem name
    def name(self):
        return "NonlinearEllipticExact"

    # Return theta multiplicative terms of the affine expansion of the problem.
    @compute_theta_for_derivatives
    def compute_theta(self, term):
        # if term == "m":
        #     theta_m0 = self.epsilon
        #     theta_m1 = 1.
        #     return (theta_m0, theta_m1)
        mu = self.mu
        if term == "a":
            theta_a0 = 1.

            return (theta_a0, )
        elif term == "c":
            theta_c0 = mu[0]
            return (theta_c0,)
        elif term == "f":
            # t = self.t
            theta_f0 = mu[1]
            theta_f1 = mu[2]
            return (theta_f0, theta_f1)
        else:
            raise ValueError("Invalid term for compute_theta().")

    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    @assemble_operator_for_derivatives
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        if term == "a":
            du = self.du
            
            # a0 = inner(grad(du), grad(v)) * dx
            a0 = inner(grad(du), grad(v)) * dx * 0

            return (a0,)
        elif term == "c":
            u = self.u
            # mu = self.mu
            c0 = self.dpsi_du(u,v)*dx
            return (c0,)
        elif term == "f":
            f = self.f
            b = self.b
            ds = self.ds
            f0 = inner(f, v) * ds(3)
            f1 = inner(b,v) * dx
            return (f0,f1)
        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, Constant((0.0, 0.0, 0.0)), self.boundaries, 1)]
            return (bc0,)
        elif term == "inner_product":
            du = self.du
            x0 = inner(grad(du), grad(v)) * dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")

    def C_times(self,f):
        return 2.*self.lambda_2*f + self.lambda_1*tr(f)*self.I

    def dpsi_du(self,u,v):
        return inner(0.5*((self.I+grad(u)).T*(self.I+grad(u))-self.I), self.C_times((self.I+grad(u)).T*grad(v)))

    # def dR_du(self,u,v,w):
    #     dx = self.dx
    #     return (inner((self.I+grad(u))*grad(w), self.C_times((self.I+grad(u)).T*grad(v))) 
    #            + inner(0.5*((self.I+grad(u)).T*(self.I+grad(u))-self.I),self.C_times(grad(w)*grad(v))))*dx



# Customize the resulting reduced problem
@CustomizeReducedProblemFor(NonlinearEllipticProblem)
def CustomizeReducedNonlinearElliptic(ReducedNonlinearElliptic_Base):
    class ReducedNonlinearElliptic(ReducedNonlinearElliptic_Base):
        def __init__(self, truth_problem, **kwargs):
            ReducedNonlinearElliptic_Base.__init__(self, truth_problem, **kwargs)
            self._nonlinear_solver_parameters.update({
                "report": True,
                "line_search": "wolfe"
            })

    return ReducedNonlinearElliptic


# 1. Read the mesh for this problem
mesh = Mesh("data/elastic_block.xml")
subdomains = MeshFunction("size_t", mesh, "data/elastic_block_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/elastic_block_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1)
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# 3. Allocate an object of the NonlinearElliptic class
nonlinear_elliptic_problem = NonlinearElliptic(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(0.1, 10.0), (10., 100.), (10., 100.)]
nonlinear_elliptic_problem.set_mu_range(mu_range)

# 4. Prepare reduction with a reduced basis method
pod_galerkin_method = PODGalerkin(nonlinear_elliptic_problem)
pod_galerkin_method.set_Nmax(20)
pod_galerkin_method.set_tolerance(1e-8)

# 5. Perform the offline phase
pod_galerkin_method.initialize_training_set(50)
reduced_nonlinear_elliptic_problem = pod_galerkin_method.offline()

# 6. Perform an online solve
online_mu = (0.1, 100., 100)
reduced_nonlinear_elliptic_problem.set_mu(online_mu)
reduced_nonlinear_elliptic_problem.solve()
reduced_nonlinear_elliptic_problem.export_solution(filename="online_solution")

# 7. Perform an error analysis
pod_galerkin_method.initialize_testing_set(50)
pod_galerkin_method.error_analysis()

# 8. Perform a speedup analysis
pod_galerkin_method.speedup_analysis()
