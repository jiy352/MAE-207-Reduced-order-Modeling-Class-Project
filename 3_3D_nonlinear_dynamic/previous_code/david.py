from dolfin import *
# export PYTHONPATH=/home/jyan/Downloads/Software/tIGAr-master/:$PYTHONPATH
# This script does not involve any IGA, but uses the generic implementation of
# generalized-alpha from tIGAr, which leads to more compact code than
# the imlementation in the demo.  (This functionality should probably be
# moved to a separate library, since it's unrealted to IGA.)
from tIGAr.timeIntegration import *

# Define mesh
mesh = BoxMesh(Point(0., 0., 0.), Point(1., 0.1, 0.04), 60, 10, 5)
d = mesh.geometry().dim()

# Sub domain for clamp at left end
def left(x, on_boundary):
    return near(x[0], 0.) and on_boundary

# Sub domain for rotation at right end
def right(x, on_boundary):
    return near(x[0], 1.) and on_boundary

# Elastic parameters
E  = 1000.0
nu = 0.3
mu    = Constant(E / (2.0*(1.0 + nu)))
lmbda = Constant(E*nu / ((1.0 + nu)*(1.0 - 2.0*nu)))

# Mass density in reference configuration
rho0 = Constant(1.0)

# Time-stepping parameters
T       = 4.0
Nsteps  = 50
dt = Constant(T/Nsteps)

p0 = 1.
cutoff_Tc = T/5
# Define the loading as an expression depending on t
p = Expression(("0", "t <= tc ? p0*t/tc : 0", "0"), t=0, tc=cutoff_Tc, p0=p0, degree=0)

# Define function space for displacement, velocity and acceleration
V = VectorFunctionSpace(mesh, "CG", 1)

# Displacement
u = Function(V)
# Displacement test function
v = TestFunction(V)

# Displacement and time derivatives from previous time level
u_old = Function(V)
udot_old = Function(V)
uddot_old = Function(V)

# Create a time integrator for the displacement.
#
# Notes:
#
# - The tIGAr implementation of generalized-alpha follows a different
#   convention from Chung and Hulbert for how alpha_f and alpha_m are used
#   in partitions of unity; tIGAr's implementation follows the convention used
#   throughout papers by Bazilevs and collaborators.  See, e.g., Section 4.4
#   of this report:  https://www.oden.utexas.edu/media/reports/2008/0816.pdf
#
# - Unlike the demo, where all the generalized-alpha parameters are assigned
#   directly, this implementation uses the family of generalized-alpha
#   methods parameterized by rho_infty (in the range 0 to 1), which is the
#   spectral radius at infinite time step.  1 is the implicit midpoint rule
#   (no numerical damping) and 0 has maximal numerical dissipation (while
#   remaining 2nd-order accurate).  

rho_infty = Constant(0.5)
timeInt = GeneralizedAlphaIntegrator(rho_infty,dt,u,
                                     (u_old, udot_old,
                                      uddot_old))
u_alpha = timeInt.x_alpha()
udot_alpha = timeInt.xdot_alpha()
uddot_alpha = timeInt.xddot_alpha()

# Create mesh function over the cell facets
boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_subdomains.set_all(0)
force_boundary = AutoSubDomain(right)
force_boundary.mark(boundary_subdomains, 3)

# Define measure for boundary condition integral
dss = ds(subdomain_data=boundary_subdomains)

# Set up boundary condition at left end
zero = Constant((0.0, 0.0, 0.0))
bc = DirichletBC(V, zero, left)

# Hyperelastic energy density; replace this to change material model.
def psi(E):
    S = 2.0*mu*E + lmbda*tr(E)*Identity(d)
    return 0.5*inner(S,E)

# Large-strain kinematics:
I = Identity(d)
F = grad(u_alpha) + I
C = F.T*F
E = 0.5*(C-I)

# Define residual of variational problem:
#
# Notes:
#
# - The reason for the factor of (1/alpha_f) is because the derivative should be
#   w.r.t. u_alpha, but the derivative function can only take derivatives
#   w.r.t. Functions, so we instead take the derivative w.r.t. u.  Since
#   u_alpha = (1-alpha_f)*u_old + alpha_f*u, this introduces a factor of
#   alpha_f that must be canceled.
#
# - This does not include the Rayleigh damping terms from the demo (which
#   were set to zero anyway).  Mass damping is easy to add, but the
#   "correct" way to implement stiffness damping in the nonlinear setting
#   is unclear.  

du_dua = Constant(1.0/timeInt.ALPHA_F)
res = inner(rho0*uddot_alpha,v)*dx \
      + du_dua*derivative(psi(E)*dx,u,v) \
      - inner(p,v)*dss(3)
Dres = derivative(res,u)

# Time stepping loop, with output of displacement at each time step:
u.rename("u","u")
uFile = File("u.pvd")
for i in range(0,Nsteps):
    if(mpirank == 0):
        print("------- Time step "+str(i+1)
              +" , t = "+str(timeInt.t)+" -------")
        
    # Set time parameter in external loading.  (See note above about
    # different convention for interpreting alpha_f.)
    p.t = timeInt.t-(1.0-float(timeInt.ALPHA_F))*float(dt)

    # Because the LHS depends on displacement in the nolinear case, we can't
    # be clever here and re-use an LU factorization, as is done in the demo.
    # Thus, we simply use the standard nonlinear solve function:
    solve(res==0,u,J=Dres,bcs=[bc,])
    # u_tip[i+1] = u(1., 0.05, 0.)[1]

    # Write output and update time integrator:
    uFile << u
    timeInt.advance()
