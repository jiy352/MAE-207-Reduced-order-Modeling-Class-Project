# Copyright (C) 2015-2020 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from mshr import *

# Create mesh
x = 0.5
y = 1.
z = 0.2
domain = Box(Point(0., 0., 0.), Point(x, y, z))
# for i in range(3):
#     for j in range(3):
#         # print(i, j, i + j * 3 + 1)
#         # print( i + j * 3 + 1)

#         domain.set_subdomain(i + j * 3 + 1, Rectangle(Point(i / 3.*0.5, j / 3.), Point((i + 1) / 3.*0.5, (j + 1) / 3.)))
#         print(i / 3.*0.5, j / 3., (i + 1) / 3.*0.5, (j + 1) / 3.)
#         # print((i + 1) / 3.*0.5, (j + 1) / 3.))

# domain.set_subdomain(1, Box(Point(0., 0., 0.), Point(x, y/2, z/2)))
# domain.set_subdomain(2, Box(Point(0., y/2, 0), Point(x, y, z/2)))
# domain.set_subdomain(3, Box(Point(0., 0., z/2), Point(x, y/2, z)))
# domain.set_subdomain(4, Box(Point(0., 0., z/2), Point(x, y, z)))

mesh = generate_mesh(domain, 32)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, 3, mesh.domains())


# Create boundaries
class Left(SubDomain):
    def __init__(self, x_min, x_max, z_min, z_max):
        SubDomain.__init__(self)
        self.x_min = x_min
        self.x_max = x_max
        self.z_min = z_min
        self.z_max = z_max

    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - 0.) < DOLFIN_EPS and x[0] >= self.x_min and x[0] <= self.x_max and x[2] >= self.z_min and x[2] <= self.z_max

class Front(SubDomain):
    def __init__(self, y_min, y_max, z_min, z_max):
        SubDomain.__init__(self)
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max

    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 0.5) < DOLFIN_EPS and x[1] >= self.y_min and x[1] <= self.y_max and x[2] >= self.z_min and x[2] <= self.z_max



class Right(SubDomain):
    def __init__(self, x_min, x_max, z_min, z_max):
        SubDomain.__init__(self)
        self.x_min = x_min
        self.x_max = x_max
        self.z_min = z_min
        self.z_max = z_max

    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - 1.) < DOLFIN_EPS and x[0] >= self.x_min and x[0] <= self.x_max and x[2] >= self.z_min and x[2] <= self.z_max

class Back(SubDomain):
    def __init__(self, y_min, y_max, z_min, z_max):
        SubDomain.__init__(self)
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max

    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 0.) < DOLFIN_EPS and x[1] >= self.y_min and x[1] <= self.y_max and x[2] >= self.z_min and x[2] <= self.z_max




class Top(SubDomain):
    def __init__(self, x_min, x_max, y_min, y_max):
        SubDomain.__init__(self)
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def inside(self, x, on_boundary):
        return on_boundary and abs(x[2] - 0.2) < DOLFIN_EPS and x[0] >= self.x_min and x[0] <= self.x_max and x[1] >= self.y_min and x[1] <= self.y_max

class Bottom(SubDomain):
    def __init__(self, x_min, x_max, y_min, y_max):
        SubDomain.__init__(self)
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def inside(self, x, on_boundary):
        return on_boundary and abs(x[2] - 0.) < DOLFIN_EPS and x[0] >= self.x_min and x[0] <= self.x_max and x[1] >= self.y_min and x[1] <= self.y_max


boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)

left = Left(0., x, 0., z) #
left.mark(boundaries, 1)

front = Front(0., y, 0., z) #
front.mark(boundaries, 2)

right = Right(0., x, 0., z) #
right.mark(boundaries, 3)

back = Back(0., y, 0., z) #
back.mark(boundaries, 4)

top = Top(0., x, 0., y) #
top.mark(boundaries, 5)

bottom = Bottom(0., x, 0., y) 
bottom.mark(boundaries, 6)

print(' top y',top.x_min, top.x_max)
print(' right y',right.z_min,right.z_max)

print(' bottom y',bottom.x_min, bottom.x_max)


# Save
File("elastic_block.xml") << mesh
File("elastic_block_physical_region.xml") << subdomains
File("elastic_block_facet_region.xml") << boundaries
XDMFFile("elastic_block.xdmf").write(mesh)
XDMFFile("elastic_block_physical_region.xdmf").write(subdomains)
XDMFFile("elastic_block_facet_region.xdmf").write(boundaries)

