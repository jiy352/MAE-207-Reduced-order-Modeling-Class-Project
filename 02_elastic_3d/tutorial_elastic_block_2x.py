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
        self.f = Constant((1.0, 0.0))
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
            theta_a1 = mu[1]
            theta_a2 = mu[2]
            theta_a3 = mu[3]
            theta_a4 = mu[4]
            theta_a5 = mu[5]
            theta_a6 = mu[6]
            theta_a7 = mu[7]
            theta_a8 = mu[8]
            theta_a9 = mu[9]
            theta_a10 = mu[10]
            theta_a11 = mu[11]
            theta_a12 = mu[12]
            theta_a13 = mu[13]
            theta_a14 = mu[14]
            theta_a15 = mu[15]
            theta_a16 = mu[16]
            theta_a17 = 1.
            print('if a-----------------------------')
            return (theta_a0, theta_a1, theta_a2, theta_a3, theta_a4, theta_a5, theta_a6, theta_a7, theta_a8,
                    theta_a9, theta_a10, theta_a11, theta_a12, theta_a13, theta_a14, theta_a15, theta_a16, theta_a17)

        elif term == "f":
            theta_f0 = mu[17]
            theta_f1 = mu[18]
            theta_f2 = mu[19]
            theta_f3 = mu[20]
            theta_f4 = mu[21]
            theta_f5 = mu[22]            
            print('if f-----------------------------')

            return (theta_f0, theta_f1, theta_f2, theta_f3, theta_f4, theta_f5)
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
            a0 = self.elasticity(u, v) * dx(1)
            a1 = self.elasticity(u, v) * dx(2)
            a2 = self.elasticity(u, v) * dx(3)
            a3 = self.elasticity(u, v) * dx(4)
            a4 = self.elasticity(u, v) * dx(5)
            a5 = self.elasticity(u, v) * dx(6)
            a6 = self.elasticity(u, v) * dx(7)
            a7 = self.elasticity(u, v) * dx(8)
            a8 = self.elasticity(u, v) * dx(9)

            a9 = self.elasticity(u, v) * dx(10)
            a10 = self.elasticity(u, v) * dx(11)
            a11 = self.elasticity(u, v) * dx(12)
            a12 = self.elasticity(u, v) * dx(13)
            a13 = self.elasticity(u, v) * dx(14)
            a14 = self.elasticity(u, v) * dx(15)
            a15 = self.elasticity(u, v) * dx(16)
            a16 = self.elasticity(u, v) * dx(17)
            a17 = self.elasticity(u, v) * dx(18)
            return (a0, a1, a2, a3, a4, a5, a6, a7, a8,
                    a9, a10, a11, a12, a13, a14, a15, a16, a17)
            
        elif term == "f":
            ds = self.ds
            f = self.f
            f0 = inner(f, v) * ds(2)
            f1 = inner(f, v) * ds(3)
            f2 = inner(f, v) * ds(4)
            f3 = inner(f, v) * ds(5)
            f4 = inner(f, v) * ds(6)
            f5 = inner(f, v) * ds(7)
            return (f0, f1, f2, f3, f4, f5)

        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, Constant((0.0, 0.0)), self.boundaries, 9)]
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
mu_range = [
    (1.0, 100.0),
    (1.0, 100.0),
    (1.0, 100.0),
    (1.0, 100.0),
    (1.0, 100.0),
    (1.0, 100.0),
    (1.0, 100.0),
    (1.0, 100.0),
    (1.0, 100.0),
    (1.0, 100.0),
    (1.0, 100.0),
    (1.0, 100.0),
    (1.0, 100.0),
    (1.0, 100.0),
    (1.0, 100.0),
    (1.0, 100.0),
    (1.0, 100.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0)
]
print("-----3333333333333333333333333333-----before_setting_mu----------")
elastic_block_problem.set_mu_range(mu_range)
print("---333333333333333333333333333333333333-----------------set_mu_range")

# 4. Prepare reduction with a POD-Galerkin method

print("--444444444444444444444-----------------Prepare reduction with a POD-Galerkin method")

pod_galerkin_method = PODGalerkin(elastic_block_problem)
pod_galerkin_method.set_Nmax(20)
pod_galerkin_method.set_tolerance(2e-4)

# 5. Perform the offline phase
print("--555555555555555-----------------Perform the offline phase")
pod_galerkin_method.initialize_training_set(100)
print("--555555555555555.5555555555555----------------initialize_training_set")

reduced_elastic_block_problem = pod_galerkin_method.offline()
print("--555555555555555.5555555555555----------------pod_galerkin_method")
# 6. Perform an online solve
print("--66666666666666666-----------------Perform an online solves")

online_mu = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,1.0, 1.0, -1.0, -1.0, -1.0, -1.0)
reduced_elastic_block_problem.set_mu(online_mu)
reduced_elastic_block_problem.solve()
reduced_elastic_block_problem.export_solution(filename="online_solution")

# 7. Perform an error analysis
pod_galerkin_method.initialize_testing_set(100)
pod_galerkin_method.error_analysis()

# 8. Perform a speedup analysis
pod_galerkin_method.initialize_testing_set(100)
pod_galerkin_method.speedup_analysis()
