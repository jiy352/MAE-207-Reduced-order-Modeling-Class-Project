# MAE-207-Reduced-order-Modeling-Class-Project
## Reduced-order modeling for fast computational analysis of hyperelastic materials

## Installation
- FeniCs from https://fenicsproject.org/
- RBniCs from https://github.com/mathLab/RBniCS
- tIGAr from https://github.com/david-kamensky/tIGAr
Note: Ubuntu users might have problem with updating the PETSC to the newest vertion. Please find the solution in tIGAr framework.

## There are mainly three problems that we consider to solve
- 3D static elastic problem with stiffness, body, traction forces changing in the POD snapshots (finished)

- 3D hyperelastic static problem with stiffness, body, traction forces changing in the POD snapshots (finished)
  
- 3D dynamic hyperelastic problem with stiffness, body, traction forces changing in the POD snapshots (in progress)
  . Note: this is a user-defined hyperbolic problem in RBniCs, which could be genralized to any 3D dynamic elastic problem     given the imported mesh (tentatively trying the meshio for unstructured geometry)
