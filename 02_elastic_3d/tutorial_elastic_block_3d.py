# Copyright (C) 2015-2020 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from rbnics import *


class ElasticBlock(EllipticCoerciveProblem):

    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        EllipticCoerciveProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        print("assert subdomain and boundaries")
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        self.u = TrialFunction(V)
        self.v = TestFunction(V)
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
        # ...
        self.f = Constant((0.0, 0.0, -1.))
        self.b = Constant((0.0, 0.0, -1.))
        self.E = 1.0
        self.nu = 0.3
        self.lambda_1 = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.lambda_2 = self.E / (2.0 * (1.0 + self.nu))
        print("load elastic coefficients")


    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        print("compute_theta--------------------")

        mu = self.mu
        print(mu)
        print(len(mu))

        print(len(self.mu_range))
        print('term:',term)
        if term == "a":
            print('if a loop')
            theta_a0 = mu[0]

            print('if a-----------------------------')
            return (theta_a0, )
        elif term == "f":
            theta_f0 = mu[1]
            theta_f1 = mu[2]
            print('if f-----------------------------')

            return (theta_f0,theta_f1)
        else:
            print('else--------------------')
            # return (0.,)
            raise ValueError("Invalid term for compute_theta().")

        print("--------------------compute_theta")


    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        print("assemble_operator--------------------")

        v = self.v
        dx = self.dx
        if term == "a":
            u = self.u
            a0 = self.elasticity(u, v) * dx
            return (a0, )
        elif term == "f":
            ds = self.ds
            dx = self.dx
            f = self.f
            b = self.b

            f0 = inner(f, v) * ds(3)
            f1 = inner(b, v) * dx
            return (f0,f1,)
        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, Constant((0.0, 0.0, 0.0)), self.boundaries, 1)]
            return (bc0,)
        elif term == "inner_product":
            u = self.u
            x0 = inner(u, v) * dx + inner(grad(u), grad(v)) * dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")
        
        print("--------------------assemble_operator")


    # Auxiliary function to compute the elasticity bilinear form
    def elasticity(self, u, v):
        print("assemble_operator--------------------")
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        print("--------------------elasticity")
        return 2.0 * lambda_2 * inner(sym(grad(u)), sym(grad(v))) + lambda_1 * tr(sym(grad(u))) * tr(sym(grad(v)))


# 1. Read the mesh for this problem
print("--------------------start_reading_mesh")
mesh = Mesh("data/elastic_block.xml")
subdomains = MeshFunction("size_t", mesh, "data/elastic_block_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/elastic_block_facet_region.xml")
print("end_reading mesh")

# 2. Create Finite Element space (Lagrange P1, two components)
V = VectorFunctionSpace(mesh, "Lagrange", 1)
print("--------------------create_finite_element_space")

# 3. Allocate an object of the ElasticBlock class
print("-----Allocate an object of the ElasticBlock class----------")

elastic_block_problem = ElasticBlock(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(1., 10), (-1e-4, 1.e-3), (-1e-4, 1.e-3)]
print("-----3333333333333333333333333333-----before_setting_mu----------")
elastic_block_problem.set_mu_range(mu_range)
print("---333333333333333333333333333333333333-----------------set_mu_range")

# 4. Prepare reduction with a POD-Galerkin method

print("--444444444444444444444-----------------Prepare reduction with a POD-Galerkin method")

pod_galerkin_method = PODGalerkin(elastic_block_problem)
pod_galerkin_method.set_Nmax(200)
# pod_galerkin_method.set_Nmin(10)

pod_galerkin_method.set_tolerance(2e-4)

# 5. Perform the offline phase
print("--555555555555555-----------------Perform the offline phase")
pod_galerkin_method.initialize_training_set(100)
print("--555555555555555.5555555555555----------------initialize_training_set")

reduced_elastic_block_problem = pod_galerkin_method.offline()
print("--555555555555555.5555555555555----------------pod_galerkin_method")
# 6. Perform an online solve
print("--66666666666666666-----------------Perform an online solves")

online_mu = (1., -1.0e-3, -1.0e-3)
# reduced_elastic_block_problem.set_mu(online_mu)

reduced_elastic_block_problem.set_mu(online_mu)
reduced_solution = reduced_elastic_block_problem.solve()
# plot(reduced_solution, reduced_problem=reduced_elastic_block_problem)

# reduced_elastic_block_problem.solve()
reduced_elastic_block_problem.export_solution(filename="online_solution")

# # 7. Perform an error analysis
pod_galerkin_method.initialize_testing_set(100)
pod_galerkin_method.error_analysis()

# # 8. Perform a speedup analysis
pod_galerkin_method.initialize_testing_set(100)
pod_galerkin_method.speedup_analysis()
