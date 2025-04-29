"""
This file was built to solve numerically a classical PDE, 2D wave equation. The equation corresponds to 

$\dfrac{\partial}{\partial x} \left( \dfrac{\partial c^2 U}{\partial x} \right) + \dfrac{\partial}{\partia
l y} \left( \dfrac{\partial c^2 U}{\partial y} \right) = \dfrac{\partial^2 U}{\partial t^2}$

where
 - U represent the signal
 - x represent the position
 - t represent the time
 - c represent the velocity of the wave (depends on space parameters)

The numerical scheme is based on finite difference method. This program is also providing several boundary conditions. More particularly the Neumann, Dirichlet and Mur boundary conditions.
Copyright - © SACHA BINDER - 2021
