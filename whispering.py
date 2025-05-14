"""
This file was built to solve numerically a classical PDE, 2D wave equation. The equation corresponds to 

$\dfrac{\partial}{\partial x} \left( \dfrac{\partial c^2 U}{\partial x} \right) + \dfrac{\partial}{\partial y} \left( \dfrac{\partial c^2 U}{\partial y} \right) = \dfrac{\partial^2 U}{\partial t^2}$

where
 - U represent the signal
 - x represent the position
 - t represent the time
 - c represent the velocity of the wave (depends on space parameters)

The numerical scheme is based on finite difference method. This program is also providing several boundary conditions. More particularly the Neumann, Dirichlet and Mur boundary conditions.
Copyright - © SACHA BINDER - 2021
"""

############## MODULES IMPORTATION ###############
# Importing necessary libraries for numerical computation and visualization

import numpy as np              # NumPy for mathematical operations and array manipulations
import matplotlib.pyplot as plt  # Matplotlib for visualization of results
import viz_tools                # Self-developed module that groups animation functions for visualizing results

# Function defining the initial condition of the wave at time t = 0
def I(x, y):
    """
    Two space variables depending function 
    that represent the wave form at t = 0
    The function models an initial wave profile as a sum of Gaussian bumps
    centered at specific points in the 2D space.
    """
    return -0.2*np.exp(-((x-3)**2/0.01 + (y-3)**2/0.01)) - 0.2*np.exp(-((x-3)**2/0.01 + (y-2)**2/0.01)) + 0.2*np.exp(-((x-2)**2/0.01 + (y-3)**2/0.01)) + 0.2*np.exp(-((x-2)**2/0.01 + (y-2)**2/0.01))

# Function defining the initial vertical speed of the wave at time t = 0
def V(x, y):
    """
    Initial vertical speed of the wave
    The initial vertical speed of the wave is set to zero.
    """
    return 0
  
############## SET-UP THE PROBLEM ###############

# Function defining the velocity field as a spatial scalar function
def celer(x, y):
    """
    Constant velocity field depending on the position (x, y).
    The function defines the velocity of the wave as a piecewise constant value.
    Inside a circular region of radius 1 centered at (2.5, 2.5), the velocity is set to 0.66,
    while outside this region, the velocity is set to 1.
    """
    if np.sqrt((x-2.5)**2 + (y-2.5)**2) < 1:
        return 0.66
    else:
        return 1

# Flag to control the execution of the processing loop
loop_exec = 1   # Processing loop execution flag

# Boundary condition selection: 
# 1: Dirichlet, 2: Neumann, 3: Mur
bound_cond = 3  # Here, the boundary condition is set to Mur

# Validation to ensure that the chosen boundary condition is valid
if bound_cond not in [1, 2, 3]:
    loop_exec = 0  # Set the flag to 0 to stop the loop execution
    print("Please choose a correct boundary condition")

# Spatial mesh setup for the x-direction
L_x = 5  # Range of the domain in the x-direction [m]
dx = 0.05  # Infinitesimal step size in the x-direction
N_x = int(L_x / dx)  # Number of points in the x-direction
X = np.linspace(0, L_x, N_x + 1)  # Array representing the spatial grid in the x-direction

# Spatial mesh setup for the y-direction
L_y = 5  # Range of the domain in the y-direction [m]
dy = 0.05  # Infinitesimal step size in the y-direction
N_y = int(L_y / dy)  # Number of points in the y-direction
Y = np.linspace(0, L_y, N_y + 1)  # Array representing the spatial grid in the y-direction

# Temporal mesh setup with Courant-Friedrichs-Lewy condition (CFL < 1)
L_t = 10  # Duration of the simulation [s]
dt = 0.1 * min(dx, dy)  # Infinitesimal time step based on the CFL condition
N_t = int(L_t / dt)  # Number of time steps
T = np.linspace(0, L_t, N_t + 1)  # Array representing the time grid

# Initialize the velocity field array (finite elements) for computation
c = np.zeros((N_x + 1, N_y + 1), float)  # Create a grid to hold the velocity values
for i in range(0, N_x + 1):
    for j in range(0, N_y + 1):
        c[i, j] = celer(X[i], Y[j])  # Assign the velocity value at each point (i, j)

############## CALCULATION CONSTANTS ###############
# Pre-calculated constants for the finite difference scheme
Cx2 = (dt / dx) ** 2  # Constant for the x-direction finite difference
Cy2 = (dt / dy) ** 2  # Constant for the y-direction finite difference

# CFL condition for the boundary points in the y-direction (left and right boundaries)
CFL_1 = dt / dy * c[:, 0]  # CFL factor at the left boundary (x = 0)
CFL_2 = dt / dy * c[:, N_y]  # CFL factor at the right boundary (x = L_x)

# CFL condition for the boundary points in the x-direction (top and bottom boundaries)
CFL_3 = dt / dx * c[0, :]  # CFL factor at the bottom boundary (y = 0)
CFL_4 = dt / dx * c[N_x, :]  # CFL factor at the top boundary (y = L_y)

############## PROCESSING LOOP ###############

if loop_exec:
    # Initialize the solution array U to store the results for each time step
    # U has dimensions (N_x+1, N_y+1, N_t+1), representing the grid and time steps
    U = np.zeros((N_x+1, N_y+1, N_t+1), float)

    # Initialize arrays to store the solution at previous, current, and next time steps
    # u_nm1: Solution at time step n-1
    # u_n: Solution at time step n
    # u_np1: Solution at time step n+1
    u_nm1 = np.zeros((N_x+1, N_y+1), float)
    u_n = np.zeros((N_x+1, N_y+1), float)
    u_np1 = np.zeros((N_x+1, N_y+1), float)

    # Initialize V_init and q arrays, which hold initial velocity and coefficient values
    V_init = np.zeros((N_x+1, N_y+1), float)
    q = np.zeros((N_x+1, N_y+1), float)

    # Initial condition at t = 0
    # Populate q with squared values of the wave speed coefficients (c)
    for i in range(0, N_x+1):
        for j in range(0, N_y+1):
            q[i, j] = c[i, j] ** 2

    # Initialize u_n with the initial wave profile from the function I(X, Y)
    for i in range(0, N_x+1):
        for j in range(0, N_y+1):
            u_n[i, j] = I(X[i], Y[j])

    # Set the initial velocity at each point (V_init) using the velocity function V(X, Y)
    for i in range(0, N_x+1):
        for j in range(0, N_y+1):
            V_init[i, j] = V(X[i], Y[j])

    # Store the initial wave profile u_n at the first time step (n=0)
    U[:, :, 0] = u_n.copy()
    
    # Initial condition at t = 1 (without boundary conditions)
    # Calculate the solution at time step n+1 based on the previous and current time steps
    # The update formula incorporates the velocity field (V_init) and wave speed coefficients (q)
    
    u_np1[1:N_x, 1:N_y] = 2*u_n[1:N_x, 1:N_y] - (u_n[1:N_x, 1:N_y] - 2*dt*V_init[1:N_x, 1:N_y]) \
        + Cx2 * (  0.5 * (q[1:N_x, 1:N_y] + q[2:N_x + 1, 1:N_y]) * (u_n[2:N_x + 1, 1:N_y] - u_n[1:N_x, 1:N_y]) \
                   - 0.5 * (q[0:N_x - 1, 1:N_y] + q[1:N_x, 1:N_y]) * (u_n[1:N_x, 1:N_y] - u_n[0:N_x - 1, 1:N_y]) ) \
        + Cy2 * (  0.5 * (q[1:N_x, 1:N_y] + q[1:N_x, 2:N_y + 1]) * (u_n[1:N_x, 2:N_y + 1] - u_n[1:N_x, 1:N_y]) \
                   - 0.5 * (q[1:N_x, 0:N_y - 1] + q[1:N_x, 1:N_y]) * (u_n[1:N_x, 1:N_y] - u_n[1:N_x, 0:N_y - 1]) )
    
    # Boundary conditions: apply Dirichlet boundary conditions (u = 0 on boundaries)
    if bound_cond == 1:
        # Dirichlet boundary conditions
        # Set the solution at the boundaries to zero (no wave displacement at boundaries)
        u_np1[0, :] = 0     # Bottom boundary (i = 0)
        u_np1[-1, :] = 0    # Top boundary (i = N_x)
        u_np1[:, 0] = 0     # Left boundary (j = 0)
        u_np1[:, -1] = 0    # Right boundary (j = N_y)

    elif bound_cond == 2:
        # Neumann boundary condition (derivative of solution is zero at the boundaries)
        
        # Bottom-left corner (i = 0, j = 0)
        u_np1[i,j] = 2*u_n[i,j] - (u_n[i,j] - 2*dt*V_init[i,j]) \
                     + Cx2*(q[i,j] + q[i+1,j])*(u_n[i+1,j] - u_n[i,j]) \
                     + Cy2*(q[i,j] + q[i,j+1])*(u_n[i,j+1] - u_n[i,j])
    
        # Top-left corner (i = 0, j = N_y)
        u_np1[i,j] = 2*u_n[i,j] - (u_n[i,j] - 2*dt*V_init[i,j]) \
                     + Cx2*(q[i,j] + q[i+1,j])*(u_n[i+1,j] - u_n[i,j]) \
                     + Cy2*(q[i,j] + q[i,j-1])*(u_n[i,j-1] - u_n[i,j])
    
        # Bottom-right corner (i = N_x, j = 0)
        u_np1[i,j] = 2*u_n[i,j] - (u_n[i,j] - 2*dt*V_init[i,j]) \
                     + Cx2*(q[i,j] + q[i-1,j])*(u_n[i-1,j] - u_n[i,j]) \
                     + Cy2*(q[i,j] + q[i,j+1])*(u_n[i,j+1] - u_n[i,j])
    
        # Top-right corner (i = N_x, j = N_y)
        u_np1[i,j] = 2*u_n[i,j] - (u_n[i,j] - 2*dt*V_init[i,j]) \
                     + Cx2*(q[i,j] + q[i-1,j])*(u_n[i-1,j] - u_n[i,j]) \
                     + Cy2*(q[i,j] + q[i,j-1])*(u_n[i,j-1] - u_n[i,j])
    
        # Left boundary (i = 0, for j = 1 to N_y - 1)
        u_np1[i, 1:N_y - 1] = 2*u_n[i, 1:N_y - 1] - (u_n[i, 1:N_y - 1] - 2*dt*V_init[i, 1:N_y - 1]) \
                              + Cx2*(q[i, 1:N_y - 1] + q[i+1, 1:N_y - 1])*(u_n[i+1, 1:N_y - 1] - u_n[i, 1:N_y - 1]) \
                              + Cy2*( 0.5*(q[i, 1:N_y - 1] + q[i, 2:N_y])*(u_n[i, 2:N_y] - u_n[i, 1:N_y - 1]) \
                                     - 0.5*(q[i, 0:N_y - 2] + q[i, 1:N_y - 1])*(u_n[i, 1:N_y - 1] - u_n[i, 0:N_y - 2]) )
    
        # Bottom boundary (j = 0, for i = 1 to N_x - 1)
        u_np1[1:N_x - 1, j] = 2*u_n[1:N_x - 1, j] - (u_n[1:N_x - 1, j] - 2*dt*V_init[1:N_x - 1, j]) \
                              + Cx2*( 0.5*(q[1:N_x - 1, j] + q[2:N_x, j])*(u_n[2:N_x, j] - u_n[1:N_x - 1, j]) \
                                     - 0.5*(q[0:N_x - 2, j] + q[1:N_x - 1, j])*(u_n[1:N_x - 1, j] - u_n[0:N_x - 2, j]) ) \
                              + Cy2*(q[1:N_x - 1, j] + q[1:N_x - 1, j + 1])*(u_n[1:N_x - 1, j + 1] - u_n[1:N_x - 1, j])
    
        # Right boundary (i = N_x, for j = 1 to N_y - 1)
        u_np1[i, 1:N_y - 1] = 2*u_n[i, 1:N_y - 1] - (u_n[i, 1:N_y - 1] - 2*dt*V_init[i, 1:N_y - 1]) \
                              + Cx2*(q[i, 1:N_y - 1] + q[i-1, 1:N_y - 1])*(u_n[i-1, 1:N_y - 1] - u_n[i, 1:N_y - 1]) \
                              + Cy2*( 0.5*(q[i, 1:N_y - 1] + q[i, 2:N_y])*(u_n[i, 2:N_y] - u_n[i, 1:N_y - 1]) \
                                     - 0.5*(q[i, 0:N_y - 2] + q[i, 1:N_y - 1])*(u_n[i, 1:N_y - 1] - u_n[i, 0:N_y - 2]) )
    
        # Top boundary (j = N_y, for i = 1 to N_x - 1)
        u_np1[1:N_x - 1, j] = 2*u_n[1:N_x - 1, j] - (u_n[1:N_x - 1, j] - 2*dt*V_init[1:N_x - 1, j]) \
                              + Cx2*( 0.5*(q[1:N_x - 1, j] + q[2:N_x, j])*(u_n[2:N_x, j] - u_n[1:N_x - 1, j]) \
                                     - 0.5*(q[0:N_x - 2, j] + q[1:N_x - 1, j])*(u_n[1:N_x - 1, j] - u_n[0:N_x - 2, j]) ) \
                              + Cy2*(q[1:N_x - 1, j] + q[1:N_x - 1, j-1])*(u_n[1:N_x - 1, j-1] - u_n[1:N_x - 1, j])
               
               
    elif bound_cond == 3:
        # Neumann cond. with flux correction

        # Left boundary
        i = 0
        u_np1[i,:] = u_n[i+1,:] + (CFL_3 - 1)/(CFL_3 + 1)*(u_np1[i+1,:] - u_n[i,:])

        # Bottom boundary
        j = 0
        u_np1[:,j] = u_n[:,j+1] + (CFL_1 - 1)/(CFL_1 + 1)*(u_np1[:,j+1] - u_n[:,j])

        # Right boundary
        i = N_x
        u_np1[i,:] = u_n[i-1,:] + (CFL_4 - 1)/(CFL_4 + 1)*(u_np1[i-1,:] - u_n[i,:])

        # Top boundary
        j = N_y
        u_np1[:,j] = u_n[:,j-1] + (CFL_2 - 1)/(CFL_2 + 1)*(u_np1[:,j-1] - u_n[:,j])
    
    # Advance time step
    u_nm1 = u_n.copy()       # Update u^{n-1}
    u_n = u_np1.copy()       # Update u^{n}
    U[:,:,1] = u_n.copy()    # Store solution at t = dt
    
    # Process loop (on time mesh)
    for n in range(2, N_t):
        
        # Compute solution at time step n+1 (interior points only)
        u_np1[1:N_x,1:N_y] = 2*u_n[1:N_x,1:N_y] - u_nm1[1:N_x,1:N_y] \
            + Cx2 * (0.5 * (q[1:N_x,1:N_y] + q[2:N_x+1,1:N_y]) * (u_n[2:N_x+1,1:N_y] - u_n[1:N_x,1:N_y]) \
            - 0.5 * (q[0:N_x-1,1:N_y] + q[1:N_x,1:N_y]) * (u_n[1:N_x,1:N_y] - u_n[0:N_x-1,1:N_y])) \
            + Cy2 * (0.5 * (q[1:N_x,1:N_y] + q[1:N_x,2:N_y+1]) * (u_n[1:N_x,2:N_y+1] - u_n[1:N_x,1:N_y]) \
            - 0.5 * (q[1:N_x,0:N_y-1] + q[1:N_x,1:N_y]) * (u_n[1:N_x,1:N_y] - u_n[1:N_x,0:N_y-1]))

        # Boundary conditions
        if bound_cond == 1:
            # Dirichlet: fixed value (zero)
            u_np1[0,:] = 0
            u_np1[-1,:] = 0
            u_np1[:,0] = 0
            u_np1[:,-1] = 0            
        
        elif bound_cond == 2:
            # Neumann: zero-flux (du/dn = 0)
            
            # Corners
            i, j = 0, 0
            u_np1[i,j] = 2*u_n[i,j] - u_nm1[i,j] + Cx2*(q[i,j] + q[i+1,j])*(u_n[i+1,j] - u_n[i,j]) + Cy2*(q[i,j] + q[i,j+1])*(u_n[i,j+1] - u_n[i,j])

            i, j = 0, N_y
            u_np1[i,j] = 2*u_n[i,j] - u_nm1[i,j] + Cx2*(q[i,j] + q[i+1,j])*(u_n[i+1,j] - u_n[i,j]) + Cy2*(q[i,j] + q[i,j-1])*(u_n[i,j-1] - u_n[i,j])

            i, j = N_x, 0
            u_np1[i,j] = 2*u_n[i,j] - u_nm1[i,j] + Cx2*(q[i,j] + q[i-1,j])*(u_n[i-1,j] - u_n[i,j]) + Cy2*(q[i,j] + q[i,j+1])*(u_n[i,j+1] - u_n[i,j])

            i, j = N_x, N_y
            u_np1[i,j] = 2*u_n[i,j] - u_nm1[i,j] + Cx2*(q[i,j] + q[i-1,j])*(u_n[i-1,j] - u_n[i,j]) + Cy2*(q[i,j] + q[i,j-1])*(u_n[i,j-1] - u_n[i,j])

            # Borders
            i = 0
            u_np1[i,1:N_y-1] = 2*u_n[i,1:N_y-1] - u_nm1[i,1:N_y-1] + Cx2*(q[i,1:N_y-1] + q[i+1,1:N_y-1])*(u_n[i+1,1:N_y-1] - u_n[i,1:N_y-1]) + \
                Cy2*(0.5*(q[i,1:N_y-1] + q[i,2:N_y])*(u_n[i,2:N_y] - u_n[i,1:N_y-1]) - 0.5*(q[i,0:N_y-2] + q[i,1:N_y-1])*(u_n[i,1:N_y-1] - u_n[i,0:N_y-2]))

            j = 0
            u_np1[1:N_x-1,j] = 2*u_n[1:N_x-1,j] - u_nm1[1:N_x-1,j] + \
                Cx2*(0.5*(q[1:N_x-1,j] + q[2:N_x,j])*(u_n[2:N_x,j] - u_n[1:N_x-1,j]) - 0.5*(q[0:N_x-2,j] + q[1:N_x-1,j])*(u_n[1:N_x-1,j] - u_n[0:N_x-2,j])) + \
                Cy2*(q[1:N_x-1,j] + q[1:N_x-1,j+1])*(u_n[1:N_x-1,j+1] - u_n[1:N_x-1,j])

            i = N_x
            u_np1[i,1:N_y-1] = 2*u_n[i,1:N_y-1] - u_nm1[i,1:N_y-1] + Cx2*(q[i,1:N_y-1] + q[i-1,1:N_y-1])*(u_n[i-1,1:N_y-1] - u_n[i,1:N_y-1]) + \
                Cy2*(0.5*(q[i,1:N_y-1] + q[i,2:N_y])*(u_n[i,2:N_y] - u_n[i,1:N_y-1]) - 0.5*(q[i,0:N_y-2] + q[i,1:N_y-1])*(u_n[i,1:N_y-1] - u_n[i,0:N_y-2]))

            j = N_y
            u_np1[1:N_x-1,j] = 2*u_n[1:N_x-1,j] - u_nm1[1:N_x-1,j] + \
                Cx2*(0.5*(q[1:N_x-1,j] + q[2:N_x,j])*(u_n[2:N_x,j] - u_n[1:N_x-1,j]) - 0.5*(q[0:N_x-2,j] + q[1:N_x-1,j])*(u_n[1:N_x-1,j] - u_n[0:N_x-2,j])) + \
                Cy2*(q[1:N_x-1,j] + q[1:N_x-1,j-1])*(u_n[1:N_x-1,j-1] - u_n[1:N_x-1,j])
              
        elif bound_cond == 3:
            # Mur boundary condition (first-order absorbing)

            i = 0
            u_np1[i, :] = u_n[i+1, :] + (CFL_3 - 1) / (CFL_3 + 1) * (u_np1[i+1, :] - u_n[i, :])

            j = 0
            u_np1[:, j] = u_n[:, j+1] + (CFL_1 - 1) / (CFL_1 + 1) * (u_np1[:, j+1] - u_n[:, j])

            i = N_x
            u_np1[i, :] = u_n[i-1, :] + (CFL_4 - 1) / (CFL_4 + 1) * (u_np1[i-1, :] - u_n[i, :])

            j = N_y
            u_np1[:, j] = u_n[:, j-1] + (CFL_2 - 1) / (CFL_2 + 1) * (u_np1[:, j-1] - u_n[:, j])

        # Time stepping
        u_nm1 = u_n.copy()
        u_n = u_np1.copy()
        U[:, :, n] = u_n.copy()
        
######################### PLOT #############################

# Create an animation of the 2D field using the animate_2D function from viz_tools.
# This will animate the field evolution with time using the input arrays X, Y, and U.
# dt is the time step and 10 specifies the interval for updating the plot.
anim = viz_tools.anim_2D(X, Y, U, dt, 10)

# Display the animated plot
plt.show()

#%%
# Generate the animation
animation_output = viz_tools.anim_2D_flat(X, Y, U, dt, 2)

# 3D surface plot generation
fig = plt.figure(figsize=(8, 8), facecolor="white")
ax = fig.add_subplot(111, projection='3d')
SX, SY = np.meshgrid(X, Y)
surface_plot = ax.plot_surface(SX, SY, U[:,:,500], cmap=plt.cm.viridis)
# Set viewing angle
ax.view_init(elev = 75, azim = 15)
# Save 3D surface plot
plt.savefig('Surface_Plot.pdf')

# 2D heatmap plot
fig, ax = plt.subplots(1, 1)
plt.title("Wave Field at 0.0 s", fontname="serif", fontsize=17)
plt.xlabel("Position (x) [m]", fontname="serif", fontsize=12)
plt.ylabel("Position (y) [m]", fontname="serif", fontsize=12)
heatmap = plt.pcolor(SX, SY, U[:,:,500], cmap=plt.cm.viridis)
plt.colorbar(heatmap, orientation="vertical")

# Save 2D heatmap plot 
plt.savefig('Heatmap_Plot.pdf')