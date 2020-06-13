# Copyright (C) 2015-2020 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from rbnics import *
from problems import *
from reduction_methods import *


class UnsteadyElasticBlock(LinearDynamicProblem):

    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        LinearDynamicProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        self.u = TrialFunction(V)
        self.v = TestFunction(V)
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
        self.d = 3

        self.f = Constant((1.0, 0.0, 0.0))
        self.E = 1.0
        self.nu = 0.3
        self.lambda_1 = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.lambda_2 = self.E / (2.0 * (1.0 + self.nu))

    # Return custom problem name
    def name(self):
        return "UnsteadyElasticBlock1POD"

    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        if term == "m":
            theta_m0 = 1.
            return (theta_m0, )
        elif term == "a":
            theta_a0 = mu[0]
            theta_a1 = 1.
            return (theta_a0, theta_a1)
        elif term == "f":
            theta_f0 = mu[1]
            return (theta_f0,)
        else:
            raise ValueError("Invalid term for compute_theta().")

    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        if term == "m":
            u = self.u
            m0 = u * v * dx
            return (m0, )
        elif term == "a":
            u = self.u
            d = self.d
            eps_u = sym(grad(u))
            sigma = self.lambda_1*div(u)*Identity(d) + 2*self.lambda_2*eps_u
            a0 = inner(sym(grad(v)), sigma) * dx
            return (a0, )
        elif term == "f":
            ds = self.ds
            f = self.f
            f0 = inner(f, v) * ds(3)
            return (f0,)
        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, Constant(0.0, 0.0, 0.0), self.boundaries, 1)]
            return (bc0,)
        elif term == "inner_product":
            u = self.u
            x0 = inner(grad(u), grad(v)) * dx
            return (x0,)
        elif term == "projection_inner_product":
            u = self.u
            x0 = u * v * dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")


# 1. Read the mesh for this problem
print("--------------------start_reading_mesh")
mesh = Mesh("data/elastic_block.xml")
subdomains = MeshFunction("size_t", mesh, "data/elastic_block_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/elastic_block_facet_region.xml")
print("end_reading mesh")

# 2. Create Finite Element space (Lagrange P1, two components)
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# 3. Allocate an object of the UnsteadyElasticBlock class
unsteady_elastic_block_problem = UnsteadyElasticBlock(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(0.1, 10.0), (-1.0, 1.0)]
unsteady_elastic_block_problem.set_mu_range(mu_range)
unsteady_elastic_block_problem.set_time_step_size(0.05)
unsteady_elastic_block_problem.set_final_time(3)

# 4. Prepare reduction with a POD-Galerkin method
pod_galerkin_method = PODGalerkin(unsteady_elastic_block_problem)
pod_galerkin_method.set_Nmax(20, nested_POD=4)
pod_galerkin_method.set_tolerance(1e-8, nested_POD=1e-4)

# 5. Perform the offline phase
pod_galerkin_method.initialize_training_set(100)
reduced_unsteady_elastic_block_problem = pod_galerkin_method.offline()

# 6. Perform an online solve
online_mu = (8.0, -1.0)
reduced_unsteady_elastic_block_problem.set_mu(online_mu)
reduced_unsteady_elastic_block_problem.solve()
reduced_unsteady_elastic_block_problem.export_solution(filename="online_solution")

# 7. Perform an error analysis
pod_galerkin_method.initialize_testing_set(10)
pod_galerkin_method.error_analysis()

# 8. Perform a speedup analysis
pod_galerkin_method.speedup_analysis()
