#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:54:48 2024

Functions: 
    01. track_film_surfaces_and_curvature
    02. compute_scalar_tangential_velocity
    03. load_density_and_velocity_data
    04. process_density_data
    05. get_surface_velocity_of_sds
    06. convert_C_FD_to_N_MD
    07. calculate_cfl
    08. plot_contributions
    09. plot_MD_CFD 
    10. get_left_BC (CX)
    11. solve_convection_diffusion_implicit_lax_wendroff

@author: muhammadrizwanurrahman
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from plot_settings import set_plot_parameters 

from scipy.interpolate import UnivariateSpline


def track_film_surfaces_and_curvature(density_data, dx=1, threshold=0.9):
    """
    Track the top and bottom surfaces of the water layer and calculate the curvature of the top surface.

    Parameters:
    - density_data: 4D numpy array with shape (x, y, z, t), representing density at each point and time.
    - threshold: Density value to distinguish water from vapor.

    Returns:
    - top_surface, bottom_surface: Two 2D numpy arrays of shape (x, t) representing the z-coordinates
                                   of the top and bottom water surfaces, respectively.
    - curvature: 2D numpy array of shape (x, t) representing the curvature of the top surface.
    """
    # Average the density data in the y direction
    density_avg_y = np.mean(density_data, axis=1)
    
    # Prepare arrays to hold the top and bottom z-coordinates for each x and t, and the curvature
    top_surface = np.full((density_avg_y.shape[0], density_avg_y.shape[-1]), np.nan, dtype=float)
    bottom_surface = np.full_like(top_surface, np.nan, dtype=float)
    curvature = np.full_like(top_surface, np.nan, dtype=float)
    splines = []  # List to store spline objects for each time step

    
    # x-coordinates for the density profiles
    x_coords = np.arange(density_avg_y.shape[0]) * dx

    # Iterate over each time step: peak to peak
    for t in range(density_avg_y.shape[-1]):
        # Get the density profile along z at each x position for this time step
        density_profile = density_avg_y[:, :, t]
        
        # Iterate over each x position to find the top surface z-coordinate
        for x in range(density_avg_y.shape[0]):
            # Get the density profile along z at this x and time
            profile_at_x = density_profile[x, :]
            
            # Find indices where density > threshold
            density_indices = np.where(profile_at_x > threshold)[0]
            if density_indices.size > 0:
                bottom_surface[x, t] = density_indices[0]  # The first index where condition is true
                top_surface[x, t] = density_indices[-1]    # The last index where condition is true

        # Calculate the curvature for the top surface using the spline fit
        valid_indices = ~np.isnan(top_surface[:, t])
        if np.any(valid_indices):
            spline = UnivariateSpline(x_coords[valid_indices], top_surface[valid_indices, t], k=4, s=3)
            dspline_dx = spline.derivative(n=1)
            dspline_dx2 = spline.derivative(n=2)
            
            # Iterate over indices of valid x positions
            for idx in np.where(valid_indices)[0]:  # This ensures we are working with integer indices
                x = x_coords[idx]  # Get the x-coordinate for the current index
                # Calculate the curvature at each x position using the spline derivatives
                f_prime = dspline_dx(x)
                f_double_prime = dspline_dx2(x)
                curvature[idx, t] = f_double_prime / (1 + f_prime**2)**1.5
                
            splines.append(spline)  # Append the spline for this time step

                
    return top_surface, bottom_surface, curvature, splines


def compute_scalar_tangential_velocity(velocity, top_surface):
    """
    Compute the scalar tangential velocity component along a curved surface.

    Parameters:
    - velocity: A 4D numpy array of shape (x, z, t, component) containing the velocity components (vx, vy, vz).
    - top_surface: A 2D numpy array of shape (x, t) containing the z-coordinate of the surface.

    Returns:
    - scalar_tangential_velocity: A 2D numpy array of shape (x, t) with the scalar tangential velocity.
    """
    # Compute the gradient of the top surface to get the slope in the x direction
    dz_dx = np.gradient(top_surface, axis=0)
    
    # Prepare an array for the scalar tangential velocities
    scalar_tangential_velocity = np.zeros(top_surface.shape)
    scalar_normal_velocity = np.zeros(top_surface.shape)
    
    for t in range(top_surface.shape[1]):
        for x in range(top_surface.shape[0]):
            z = top_surface[x, t]
            
            # Interpolate to find velocity components at this z for each time and x
            vx = np.interp(z, np.arange(velocity.shape[1]), velocity[x, :, t, 0])
            vy = np.interp(z, np.arange(velocity.shape[1]), velocity[x, :, t, 1])
            vz = np.interp(z, np.arange(velocity.shape[1]), velocity[x, :, t, 2])

            # Construct the velocity vector and the surface tangent vector at this point
            velocity_vector = np.array([vx, vy, vz])
            surface_normal = np.array([dz_dx[x, t], 0, -1])  # Simplified surface normal vector in x-z plane, assuming y component doesn't directly affect slope
            
            # Normalize the surface normal vector
            surface_normal_normalized = surface_normal / np.linalg.norm(surface_normal)

            # Project the velocity vector onto the surface normal vector to get the component perpendicular to the surface
            velocity_normal_component = np.dot(velocity_vector, surface_normal_normalized)
            
            # Subtract the normal component from the total velocity vector to get the tangential component
            tangential_vector = velocity_vector - velocity_normal_component * surface_normal_normalized
            
            # Compute the magnitude of the tangential velocity vector
            tangential_velocity_magnitude = np.linalg.norm(tangential_vector)

            scalar_tangential_velocity [x, t] = tangential_velocity_magnitude
            scalar_normal_velocity [x,t] = velocity_normal_component

    return scalar_tangential_velocity, scalar_normal_velocity 



def load_density_and_velocity_data(filepath, surf_density_file, water_density_file, surf_velocity_file ):
    
    """Load density and velocity data for surfactant and water."""
    surfactant_density = np.load(filepath + surf_density_file)
    water_density = np.load(filepath + water_density_file)
    surfactant_velocity = np.load(filepath + surf_velocity_file)
     
    
    return surfactant_density, water_density, surfactant_velocity



def process_density_data(density_data, start_discard=5, window_length=10):
    
    """Process density data to obtain surface density and apply smoothing."""
    
    midx = density_data.shape[0] // 2 
    density = density_data[midx:, :, :, :, 0].mean(axis=1)
    density_nan = np.where(density <= 0, np.nan, density)
    c_md_with_nan_values = np.nanmean(density_nan, axis=1)
    c_md = np.nan_to_num(c_md_with_nan_values, nan=0)
    c_md_smooth = savgol_filter(c_md, window_length=window_length, polyorder=3)
    return c_md_smooth[:, start_discard:]  # Discarding initial time steps for stability

def get_surface_velocity_of_sds (surfactant_velocity):
    
    """
    does not take into account the surface curvature
    input:
        - surfactant_velocity of shape surfactant_velocity(x,y,z,t,components)
        
    returns: 
        - vsurf of shape (x,t) for half of the film from center to right end
        - vsurf_mean (t) : mean velocity at all time
    """
    midx = surfactant_velocity.shape[0]//2
    midz = surfactant_velocity.shape[2]//2
    surfactant_velocity_yx = surfactant_velocity [midx:,:,midz:,:,:].mean(axis=1)
    surfactant_velocity_yx [surfactant_velocity_yx==0] = np.nan 
    surrfactant_velocity_surflayer = np.nanmean(surfactant_velocity_yx,axis=1)
    
    
    surrfactant_velocity_surflayer_res = np.sqrt( np.sum(surrfactant_velocity_surflayer**2, axis=-1  ))
    surrfactant_velocity_surflayer_res_nmperns = surrfactant_velocity_surflayer_res * 1e5
    
    vsurf_mean = np.nanmean (surrfactant_velocity_surflayer_res_nmperns, axis=0) 
    
    surrfactant_velocity_surflayer_res_nmperns = np.nan_to_num(surrfactant_velocity_surflayer_res_nmperns, 0)
    vsurf = savgol_filter(surrfactant_velocity_surflayer_res_nmperns, window_length=10, polyorder=3)

    return vsurf , vsurf_mean


def convert_C_FD_to_N_MD (C_FD, dz):
    """convert sds concentration from gm/cm3 to number/nm2 """
    molar_mass_SDS = 288.38  # Molar mass of SDS in g/mol
    avogadros_number = 6.022e23  # molecules/mol
    cm3_to_nm3_conversion_factor = 1e21  # 1 cm^3 is 10^21 nm^3
    #nm2_to_cm2_conversion_factor = 1e14  # 1 cm^2 is 10^14 nm^2
    
    # Step 1: Convert the mass concentration (g/cm^3) to a mass concentration in a 1 nm thick layer (g/nm^3)
    # Note: This step considers the mass concentration in the specified dz thickness, so we adjust for volume conversion later
    C_FD_g_per_nm3 = C_FD / cm3_to_nm3_conversion_factor

    # Step 2: Convert mass concentration in g/nm^3 to moles/nm^3 (by dividing by molar mass)
    C_FD_moles_per_nm3 = C_FD_g_per_nm3 / molar_mass_SDS
    
    # Step 3: Convert moles/nm^3 to molecules/nm^3 (by multiplying by Avogadro's number)
    C_FD_molecules_per_nm3 = C_FD_moles_per_nm3 * avogadros_number
    
    # Step 4: Calculate the number of molecules per nm^2, considering the thickness dz
    # The molecules/nm^3 already accounts for the 3D concentration, we now distribute this over the dz thickness to get molecules/nm^2
    N_MD = C_FD_molecules_per_nm3 * dz
    
    return N_MD


def calculate_cfl(velocity, dt, dx):
    
    cfl = np.max(velocity) * dt / dx
    return cfl

def plot_contributions (contrib_avd2,contrib_diff2, contrib_curv2, dt):
    
 
    time_in_ns = np.arange(0,len(contrib_avd2[5:]))*dt
    contrib_avd_sm = savgol_filter(contrib_avd2[5:], 20,3)
    contrib_diff_sm = savgol_filter(contrib_diff2[5:], 20,3)
    contrib_curv_sm = savgol_filter(contrib_curv2[5:], 20,3)
    
    
    fig,ax = plt.subplots(figsize=(8,8))
    set_plot_parameters()
    ax.plot(time_in_ns, contrib_avd_sm , 'ko-', ms=18, mec='k', markevery=10,linewidth=5)
    ax.plot(time_in_ns, contrib_diff_sm, 'bo-', ms=18,  mec='k', markevery=10, linewidth=5,  )
    ax.plot(time_in_ns, contrib_curv_sm, 'o-', ms=18,mfc ='w',  mec='k', markevery=10, linewidth=5, color='k')
    ax.set_ylim(-0.1,1.)
    ax.set_xlim(0,6)
    
    fname = './figures/' + 'plot-' + str(t) + '.jpg'
    #plt.savefig (fname, dpi=600) 
    plt.show()


def plot_MD_CFD (locx, C_MD,C0_MD,C_FD2, C0_FD,locx_right,topsurf_smooth,markevery):
    fig,ax = plt.subplots(figsize=(10,8))
    set_plot_parameters()
    ax2 = ax.twinx() 
            
    ax.plot(locx  , C_MD[:, t]/C0_MD,'o', ms=20, markevery=markevery, mfc='silver', mec='k',mew=3,label='MD')
    cfd2 = savgol_filter(C_FD2 [:, t], window_length=10, polyorder=3)
    ax.plot(locx  , cfd2/C0_FD  , 'k-',linewidth=6, alpha=1, label='with surface velocity')
        
    ax2.plot(locx_right, topsurf_smooth[:,t] ,':', linewidth=6, ms=16,markevery= 2*markevery, mfc='None', mec='k',mew=5,label='MD',color='silver')
    ax.set_ylim(-.05, 1.2)
    ax2.set_ylim(50,80)
    ax2.set_yticks([50,55,60,65,70,75,80])
    tinns = np.round(t*0.03,3)
    ax.set_xlim(-1,50)
    plt.xlabel('x (nm)') 
    plt.title(f't: {tinns}',pad=20)
        
    fname = './figures/' + 'plot-' + str(t) + '.jpg'
    #plt.savefig (fname, dpi=600) 
    #plt.show() 
    


def get_left_BC (CX): 
    
    """ set parameters based on prior analysis on left BC """
    
    if CX == '1' :
        print(CX)
        #D, velocity = 3.5, 4 # for C1
        A,Lambda,C0 = [0.13373664, 0.10346078, 0.08554894] # for C1
        
    elif CX == '1p5' : # interpolated data
        print(CX)
         
        A,Lambda,C0 = [0.19759487, 0.08522611, 0.10520443] # for C1.5   
        
    elif CX == '2' :
        print(CX)
        #D, velocity = 3, 4.8 # for C2
        A,Lambda,C0 = [0.26145309, 0.06699145, 0.12485991] # for C2
    
    elif CX == '3' : # interpolated data
        print(CX)
        
        A,Lambda,C0 = [0.26152298, 0.06699145, 0.16247527] # for C3   
    
    elif CX =='4' :
        #D, velocity = 3, 5.8 # for C4
        A,Lambda,C0 = [0.26159287, 0.06699145, 0.20009063] # for C4
        
    C_left_boundary  =   A * np.exp(-Lambda * t_bc) + C0
    
    return C_left_boundary
    


def solve_convection_diffusion_implicit_lax_wendroff(C_left, C_right, initial_condition, dS, dt, allvelocity, num_steps, dz, Deff, Unormal, kappa):
    """
    Solve the convection-diffusion equation using an implicit method with the Lax-Wendroff scheme for convection.
    - C_left, C_right: Boundary conditions.
    - initial_condition: Initial concentration profile.
    - dS: 2D Spatial discretization array for every time step: dS^2 = dX^2 + dH^2 
    - dt: Time step.
    - allvelocity: 2D array of velocities at each spatial step and each time step.
    - num_steps: Number of time steps to simulate.
    - dz: Spatial resolution.
    - Deff: Diffusion coefficient.
    """
    
    Nx = len(initial_condition)
    C_FD = np.zeros((Nx, num_steps))
    C_FD[:, 0] = initial_condition    
    
    cum_adv = np.zeros(num_steps)  # Cumulative advective contributions
    cum_diff = np.zeros(num_steps)  # Cumulative diffusive contributions
    cum_x = np.zeros(num_steps)  # Cumulative contributions from extra terms
    
    testN = np.zeros((Nx, num_steps))
    
    for t in range(1, num_steps):

        A = np.zeros((Nx, Nx)) 
        b = np.zeros(Nx)
        
        dx = dS #np.diff(locx[:,t])
        dx = np.abs(dx) 
        dx_plus = np.append(dx, dx[-1])  # Boundary adjustment
        dx_minus = np.append(dx[0], dx)  # Boundary adjustment

        for i in range(1, Nx-1):
            velocity = allvelocity[i,t-1] if isinstance(allvelocity, np.ndarray) else allvelocity
            reaction_kappa = kappa[i,t-1] if isinstance(kappa, np.ndarray) else kappa
            reaction_v = Unormal[i,t-1] if isinstance(Unormal, np.ndarray) else Unormal
            
            N_MD = convert_C_FD_to_N_MD(C_FD[i, t-1], dz)
            if Deff is None:
                D_eff = 3.22 * np.exp(-1.31 * N_MD) if N_MD > 0 else 0
                #print(f'\n Deff is: {D_eff}')
            else:
                D_eff = Deff
                #print(f'\n Deff is: {D_eff}')
            testN [i,t] = N_MD

            # Lax-Wendroff scheme for convection
            if velocity > 0:
                A[i, i-1] += -velocity * (1 + velocity * dt / dx_minus[i]) * dt / 2 / dx_minus[i]
                A[i, i] += 1 + velocity**2 * dt**2 / dx_minus[i]**2
                A[i, i+1] += velocity * (1 - velocity * dt / dx_plus[i]) * dt / 2 / dx_plus[i]
            else:
                A[i, i-1] += -velocity * (1 - velocity * dt / dx_minus[i]) * dt / 2 / dx_minus[i]
                A[i, i] += 1 + velocity**2 * dt**2 / dx_minus[i]**2
                A[i, i+1] += velocity * (1 + velocity * dt / dx_plus[i]) * dt / 2 / dx_plus[i]

            # Central difference for diffusion
            if D_eff > 0:
                A[i, i-1] += -D_eff * dt / dx_minus[i-1]**2
                A[i, i] += 2 * D_eff * dt / dx_minus[i]**2
                A[i, i+1] += -D_eff * dt / dx_plus[i]**2
                
            A[i, i] += dt * reaction_kappa * reaction_v    
            b[i] = C_FD[i, t-1]
            
            """ CONTRIBUTIONS OF EACH MECHANISM """
            # Convection contributions
            adv_coef = velocity * dt / dx_minus[i] if velocity >= 0 else velocity * dt / dx_plus[i]
            if ~np.isnan(adv_coef) and ~np.isinf(adv_coef):
                cum_adv[t] += np.abs(adv_coef ) * C_FD[i, t-1]

            # Diffusion contributions
            diff_coef = D_eff * dt / (dx_minus[i]**2)
            if ~np.isnan(diff_coef) and ~np.isinf(diff_coef):
                cum_diff[t] +=   diff_coef * C_FD[i, t-1]  

            # Extra term contributions
            x_coef = reaction_kappa * reaction_v * dt
            if ~np.isnan(x_coef) and ~np.isinf(x_coef):
                cum_x[t] += np.abs(x_coef) * C_FD[i, t-1]
                   
        # Apply boundary conditions
        A[0, 0] = 1
        #A[0,:] = 0 # ERS
        #A[-1,:] = 0 # ERS
        A[-1, -1] = 1
        b[0] = C_left[t]
        b[-1] = C_right[t]

        C_new = np.linalg.solve(A, b)
        
        # Apply conservation correction to maintain total mass
        total_surfactant =  np.sum(C_new)
        correction_factor = np.sum(initial_condition) / total_surfactant if total_surfactant else 1
        C_FD[:, t] = C_new * correction_factor
        
        # Normalize cumulative contributions for the timestep
        total_contrib = cum_adv[t] + cum_diff[t] + cum_x[t]
        
        if total_contrib > 0:  # Avoid division by zero
            cum_adv[t] /= total_contrib
            cum_diff[t] /= total_contrib
            cum_x[t] /= total_contrib
        

    return C_FD, cum_adv, cum_diff, cum_x, testN


""" USER INPUTS """ 
# Load data files
CX = '1'
en = '-en01'
 
## MORE DATA FILE ON PORTABLE 
filepath = './sample_data_files/' 
surf_density_file = 'Csurf-1-en01-surfactant-density-gmpercm3.npy'
water_density_file = 'Csurf-1-en01-water-density-gmpercm3.npy'
surf_velocity_file = 'Csurf-1-en01-surfactant-velocity-aaperfs.npy'


""" VARIABLES OF SIMULATIONS """
Lx, Ly, Lz = 1000, 200, 500  # Domain length in AA
Nx, Ny, Nz = 100, 20, 200     # Number of grids in each direction
Nevery, Nrepeat = 2000, 3
delt = 5e-15

""" DEFINED AND DERIVED VARIABLES FOR ANALYSIS """ 
dz = (Lz / Nz) * 0.1 # in nm 
dx = (Lx / Nx) * 0.1 # in nm 
Nfreq = Nevery * Nrepeat
time_conv_factor = Nfreq * delt * 1e9 # in ns 
dt = time_conv_factor 
window_length = 15 # for smoothing 


""" PROCESS DENSITY DATA """   
surfactant_density, water_density, surfactant_velocity = load_density_and_velocity_data(filepath, surf_density_file, 
                                                                                        water_density_file, surf_velocity_file, 
                                                                                        )

# smooth MD density data from center to Right end of the film 
start_discard = 1
C_MD = process_density_data(surfactant_density, start_discard, window_length=window_length) # gm/cm3
initial_surfactant_concentration = C_MD[0,0] #C_MD[0:3,0].mean() # mean of central 5 nm at RHS
print(f'Initial concentration at center is: {initial_surfactant_concentration}') 
 
""" BOUNDARY AND INITIAL CONDITIONS """

C_left_boundary = C_MD[0, :] 
C_right_boundary = C_MD[-1, :] 
initial_condition = C_MD[:, 0] 

# total density required for surface detection  
total_density = surfactant_density + water_density 
total_density = total_density [:,:,:,:,0 ] 
midx = total_density.shape[0]//2 
 
""" PROCESS VELOCITY DATA """  
allvelocity, velocity_mean = get_surface_velocity_of_sds (surfactant_velocity) 
top_surface, bottom_surface, top_curvature, splines = track_film_surfaces_and_curvature (total_density)

velocity_yavg = surfactant_velocity.mean(axis=1)
velocity_yavg_nmperns = velocity_yavg [midx:,:,:,:] * 1e5 # nm/ns
top_surface_right = top_surface[midx:,:]

velocity_along_surface, velocity_normal2_surface = compute_scalar_tangential_velocity(velocity_yavg_nmperns, top_surface_right)
velocity_along_surface_sm = savgol_filter(velocity_along_surface,window_length,3)
velocity_normal2_surface_sm = savgol_filter(velocity_normal2_surface,window_length,3)

# consider cente-to-rightend
top_surface_right = top_surface [midx:,:]
locx_right = np.arange(0,len(top_surface_right))*dx
bottom_surface_right = bottom_surface [midx:,:] 
top_curvature_right = top_curvature [midx:,:] 

##################################################################### 
"""                  SOLVE TRANSPORT EQUATION                     """ 
##################################################################### 
num_steps = C_MD.shape[1]  # Number of time steps 
locx = np.arange(C_MD.shape[0]) * dx  

# get the spatial discretization along the surface  
dH_unpadded = np.diff(top_curvature_right.copy(),axis=0) 
dH = np.pad(dH_unpadded, ((1,0),(0,0)), mode ='reflect')
dX_unpadded = np.diff(locx)
dX = np.pad(dX_unpadded, (1,0), mode='reflect')

dS = np.zeros_like(dH)
for tstep in range (dH.shape[1]):
    for i in np.arange (0,dX.shape[0]-1):
        dS[i,tstep] =  ( dX[i]**2 + (dH[i,tstep])**2 ) **0.5

dS [:,0] = dS [:,1]  # setting the second timestep equal to first time step (diff reduces one step)
dS [-1,:] = dS[-2,:]  # setting last spatial grid equal to the secon-last grid 

plt.subplots(figsize=(8,8))
#plt.plot(locx ,dH[:,10],'k-')
plt.plot(locx ,dS[:,10],'b-')
plt.plot(locx ,dX[:],'k--')
plt.ylim(0,5)


cfl = calculate_cfl(velocity_mean, 6000*5e-15 , 1e-9)

# should be less than 1 for numerical stability
if cfl>=1:
    print('CFL>1')
else:      
    print(f'CFL:{cfl} < 1. Ok for stability ')
          
    t_bc = np.arange(0,C_MD.shape[1])
    C_left_boundary  =   get_left_BC (CX) 
    U = velocity_along_surface_sm 
    V = velocity_normal2_surface_sm
     
    D = None # if None, then an expression is used to compute local diffusion 
    kappa =   savgol_filter(top_curvature_right, window_length=window_length, polyorder=3, axis=0)
    topsurf_smooth = savgol_filter(top_surface_right[:,:], window_length=10, polyorder=3, axis=0)

    # Solve transport equation
    C_FD2, contrib_avd2, contrib_diff2, contrib_curv2,testD = solve_convection_diffusion_implicit_lax_wendroff (
                                             C_left_boundary, C_right_boundary, initial_condition, 
                                             dS, dt, U, 
                                             num_steps, dz, D, V, kappa)
 
    C_FD = C_FD2.copy() 
    markevery = 2
    C0_MD =  C_MD [0,0]
    C0_FD =  C_FD [0,0]
    residual = C_MD - C_FD2
    err = residual  /2
    print(f'max residual is: {np.max(residual)}')
     
    for t in np.arange(0, 170, 30):
        
        plot_MD_CFD (locx, C_MD,C0_MD,C_FD2, C0_FD,locx_right,topsurf_smooth,markevery)


""" Contributions of each of the terms in the transport equation """
# uncomment to plot contributions
# plot_contributions (contrib_avd2,contrib_diff2, contrib_curv2, dt )
