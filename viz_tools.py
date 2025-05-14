"""
File containing several visualization functions intended for use with the results from 1D/2D wave equation simulations.
"""

############## MODULE IMPORTS ###############
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

##################################################

def plot_a_frame_1D(x, y, xmin, xmax, ymin, ymax, title="My title", line_style="-"):
    """
    Plots a 1D solution with a customizable window.
    
    Parameters:
    x (np.ndarray): 1D array representing the x-axis (position).
    y (np.ndarray): 1D array representing the y-axis (signal).
    xmin (float): Minimum value of the x-axis.
    xmax (float): Maximum value of the x-axis.
    ymin (float): Minimum value of the y-axis.
    ymax (float): Maximum value of the y-axis.
    title (str): Title of the plot (default is "My title").
    line_style (str): Line style for the plot (default is "-").
    
    Returns:
    plot: The generated plot.
    """
    plt.axis([xmin, xmax, ymin, ymax])
    plt.plot(x, y, line_style, color="black")
    plt.title(title)
    plt.xlabel("x-axis [m]")
    plt.ylabel("y-axis [m]")
    plt.show()

##################################################

def plot_spatio_temp_3D(x, y, z):
    """
    Plots a 3D surface representing a function z = f(x, t), where x is spatial and y is time.
    
    Parameters:
    x (np.ndarray): 1D array representing the spatial variable.
    y (np.ndarray): 1D array representing the time variable.
    z (np.ndarray): 2D array representing the values of the function z at (x, y).
    
    Returns:
    plot: The generated 3D plot.
    """
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(r'$x \ [ m ]$', fontsize=16)
    ax.set_ylabel('Time', fontsize=16)
    ax.set_zlabel('Amplitude', fontsize=16)
    ax.view_init(elev=15, azim=120)
    
    ST, SX = np.meshgrid(y, x)
    ax.plot_surface(SX, ST, z, color='white')       
    plt.show()

##################################################

def anim_1D(x, y, time_step, frame_step, save=False, xlim=(0, 4), ylim=(-4, 4)):
    """
    Creates an animation for a 1D wave propagation using the provided data and time step.
    
    Parameters:
    x (np.ndarray): 1D array representing the x-axis (position).
    y (np.ndarray): 2D array where each column represents a snapshot of the signal at a given time.
    time_step (float): The time step used for the simulation.
    frame_step (int): The frame step to skip between frames.
    save (bool): Whether to save the animation as a video (default is False).
    xlim (tuple): Limits for the x-axis (default is (0, 4)).
    ylim (tuple): Limits for the y-axis (default is (-4, 4)).
    
    Returns:
    animation: The generated animation object.
    """
    fig = plt.figure()
    ax = plt.axes(xlim=xlim, ylim=ylim)
    line, = ax.plot([], [])
    ax.set_title("t = 0 s", fontname="serif", fontsize=16)
    ax.set_xlabel("x [m]", fontname="serif", fontsize=14)
    ax.set_ylabel("$u$ [m]", fontname="serif", fontsize=14)
    
    def init():
        line.set_data([], [])
        return line,
    
    def animate(i):
        line.set_data(x, y[:, frame_step * i])
        ax.set_title(f"$u(x)$ at = {np.round(i * frame_step * time_step * 0.165, 4)} fs", fontname="serif", fontsize=16)
        return line,
        
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=y.shape[1] // frame_step, interval=10, blit=True)
    
    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)
        anim.save('wave_animation.mp4', writer=writer)
    
    return anim

##################################################

def anim_2D(X, Y, L, time_step, frame_step, save=False, zlim=(-0.15, 0.15)):
    """
    Creates an animation for a 3D surface representing the function z(x, y) over time.
    
    Parameters:
    X (np.ndarray): 1D array representing the spatial variable along the x-axis.
    Y (np.ndarray): 1D array representing the spatial variable along the y-axis.
    L (np.ndarray): 3D array where each slice L[:,:,i] represents the z values at time step i.
    time_step (float): The time step used for the simulation.
    frame_step (int): The frame step to skip between frames.
    save (bool): Whether to save the animation as a video (default is False).
    zlim (tuple): Limits for the z-axis (default is (-0.15, 0.15)).
    
    Returns:
    animation: The generated animation object.
    """
    fig = plt.figure(figsize=(8, 8), facecolor="white")
    ax = fig.add_subplot(111, projection='3d')
    SX, SY = np.meshgrid(X, Y)
    surf = ax.plot_surface(SX, SY, L[:,:,0], cmap=plt.cm.viridis)
    ax.set_zlim(zlim[0], zlim[1])
    ax.set_title("t = 0 s", fontname="serif", fontsize=16)
    
    def update_surf(num):
        ax.clear()
        ax.plot_surface(SX, SY, L[:,:,frame_step * num], cmap=plt.cm.viridis)
        ax.set_xlabel("x [μm]", fontname="serif", fontsize=14)
        ax.set_ylabel("y [μm]", fontname="serif", fontsize=14)
        ax.set_zlabel("$u$ [u.a.]", fontname="serif", fontsize=16)
        ax.set_title(f"$u(x,y)$ at = {np.round(frame_step * num * time_step * 0.165, 4)} fs", fontname="serif", fontsize=16)
        ax.set_zlim(zlim[0], zlim[1])
        plt.tight_layout()
        return surf,
        
    anim = animation.FuncAnimation(fig, update_surf, frames=L.shape[2] // frame_step, interval=50, blit=False)
    
    if save:
        writer = animation.FFMpegWriter(fps=24, bitrate=10000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
        anim.save('2d_wave_animation.mp4', writer=writer)
    
    return anim

##################################################

def anim_2D_flat(X, Y, L, time_step, image_interval, save=False, myzlim=(-0.15, 0.15)):
    """
    Animates a 2D plot of a wave function z(x, y) over time. The list L contains the sequence of 2D images to display:
    L = [ [z_1(x,y)], [z_2(x,y)], ..., [z_n(x,y)] ].
    
    Parameters:
    X (np.ndarray): 1D array of x coordinates.
    Y (np.ndarray): 1D array of y coordinates.
    L (np.ndarray): 3D array of shape (Nx, Ny, Nframes), where each slice L[:,:,i] is the 2D data at time step i.
    time_step (float): Time step used for animation time progression.
    image_interval (int): Frame interval for displaying images from L.
    save (bool): If True, save the animation as an MP4 file.
    myzlim (tuple): Value range for color mapping.

    Returns:
    animation.FuncAnimation: The animated figure.
    """
    
    # Create figure and axis for the plot
    fig, ax = plt.subplots(1, 1)
    plt.title("0.0 s", fontname="serif", fontsize=17)
    plt.xlabel("x [m]", fontname="serif", fontsize=12)
    plt.ylabel("y [m]", fontname="serif", fontsize=12)

    # Set up the initial plot
    mesh = plt.pcolormesh(X, Y, L[:,:,0], vmin=myzlim[0], vmax=myzlim[1], cmap=plt.cm.viridis)
    plt.colorbar(mesh, orientation="vertical")
    
    # Update function for the animation
    def update_surf(num):
        # Update title with the corresponding time
        time = num * image_interval * time_step
        ax.set_title(f"$u(x,y)$ at t = {np.round(time, 4)} s", fontname="serif", fontsize=16)
        
        # Update the data by creating a new mesh for each frame
        mesh.set_array(L[:, :, image_interval * num].flatten())  # Update the plot with new data
        return mesh,

    # Create the animation
    anim = animation.FuncAnimation(fig, update_surf, frames=L.shape[2] // image_interval, 
                                  interval=1000 * time_step, blit=False)

    # Save the animation if specified
    if save:
        writer = animation.FFMpegWriter(fps=24, bitrate=10000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
        anim.save('wave_animation.mp4', writer=writer)
    
    return anim