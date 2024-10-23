import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fsolve, minimize_scalar, minimize


plt.rcParams.update({
    "text.usetex": True,         # Enable LaTeX
    "font.family": "serif",      # Use serif fonts in LaTeX
    "text.latex.preamble": r"\usepackage{amsmath}"  # Load amsmath package for math symbols
})

# Initialize the bilinear element class
class BilinearElement2D:
    def __init__(self, x_coords, y_coords):
        assert len(x_coords) == 4 and len(y_coords) == 4, "There must be exactly 4 corner points."
        self.x_coords = np.array(x_coords)
        self.y_coords = np.array(y_coords)

    def shape_functions(self, chi, omega):
        N1 = 0.25 * (1 - chi) * (1 - omega)
        N2 = 0.25 * (1 + chi) * (1 - omega)
        N3 = 0.25 * (1 + chi) * (1 + omega)
        N4 = 0.25 * (1 - chi) * (1 + omega)
        return np.array([N1, N2, N3, N4])

    def map_to_physical(self, chi, omega):
        N = self.shape_functions(chi, omega)
        x = np.dot(N, self.x_coords)
        y = np.dot(N, self.y_coords)
        return x, y

    def jacobian(self, chi, omega):
        N_chi = np.array([
            -0.25 * (1 - omega),  0.25 * (1 - omega),
             0.25 * (1 + omega), -0.25 * (1 + omega)
        ])
        N_omega = np.array([
            -0.25 * (1 - chi), -0.25 * (1 + chi),
             0.25 * (1 + chi),  0.25 * (1 - chi)
        ])
        J11 = np.dot(N_chi, self.x_coords)
        J12 = np.dot(N_chi, self.y_coords)
        J21 = np.dot(N_omega, self.x_coords)
        J22 = np.dot(N_omega, self.y_coords)
        J = np.array([[J11, J12], [J21, J22]])
        if np.abs(np.linalg.det(J)) < 1e-8:
            raise ValueError("Jacobian is near singular. Check quadrilateral shape or try different initial guess.")
        
        return J

    def map_to_reference(self, x, y):

        # Use a numerical approach to find (chi, omega) that corresponds to the given (x_scaled, y_scaled)

        def objective(params):
            chi, omega = params
            mapped_x, mapped_y = self.map_to_physical(chi, omega)
            return (mapped_x - x) ** 2 + (mapped_y - y) ** 2

        # Start the optimization at the center of the reference square
        result = minimize(objective, [0.0, 0.0], bounds=[(-1.01, 1.01), (-1.01, 1.01)])
        if result.success:
            return result.x # Returns (chi, omega)
        else:
            print("Warning, Could not find a suitable reference", x,y)
            return [np.NaN, np.NaN]
            #raise ValueError("Could not find a suitable reference point.")

    def is_point_inside(self, x_points, y_points):
        # Convert input points to arrays
        x_points = np.array(x_points)
        y_points = np.array(y_points)

        # Prepare arrays for output
        inside = np.zeros(x_points.shape, dtype=bool)

        # Define the number of vertices (4 for quadrilateral)
        n = 4
        
        # Loop through each edge of the quadrilateral
        p1x, p1y = self.x_coords[0], self.y_coords[0]
        for i in range(n + 1):
            p2x, p2y = self.x_coords[i % n], self.y_coords[i % n]
            
            # Conditions to determine intersections
            y_condition = (y_points > np.minimum(p1y, p2y)) & (y_points <= np.maximum(p1y, p2y))
            x_condition = x_points <= np.maximum(p1x, p2x)
            valid_indices = y_condition & x_condition
            
            # Calculate intersection x-coordinates where conditions are met
            if p1y != p2y:
                xinters = (y_points[valid_indices] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            else:
                xinters = np.full(y_points[valid_indices].shape, p1x)
            
            # Update the inside array using vectorized operations
            inside[valid_indices] ^= (p1x == p2x) | (x_points[valid_indices] <= xinters)
            
            # Move to the next vertex
            p1x, p1y = p2x, p2y
        
        return inside

    def is_convex_quadrilateral(self):

        x = self.x_coords
        y = self.y_coords

        # Define vectors for each side
        v1 = (x[1] - x[0], y[1] - y[0])
        v2 = (x[2] - x[1], y[2] - y[1])
        v3 = (x[3] - x[2], y[3] - y[2])
        v4 = (x[0] - x[3], y[0] - y[3])
        
        # Calculate cross products for each consecutive pair of vectors
        def cross_product(v, w):
            return v[0] * w[1] - v[1] * w[0]
        
        z1 = cross_product(v1, v2)
        z2 = cross_product(v2, v3)
        z3 = cross_product(v3, v4)
        z4 = cross_product(v4, v1)
        
        # Check if all cross products have the same sign
        if (z1 > 0 and z2 > 0 and z3 > 0 and z4 > 0) or (z1 < 0 and z2 < 0 and z3 < 0 and z4 < 0):
            return True  # Convex
        else:
            return False  # Concave or self-intersecting

def xi(x):
    return spline(x)

def dxi(x):
    return spline.derivative()(x)

def norm(x):
    return -1./dxi(x)

def a_xi(x):
    return np.sqrt(1+ dxi(x)**2)


#Equation for line normal to surface
#x0 point on surface
def l(x, x0):
    return xi(x0)+(x0-x)/dxi(x0)

def inv_l(y, x0):
    return x0-dxi(x0)*(y-xi(x0))

# Find intersection of the normal line at x0 with the shifted curve xi(x) + shift
def find_intersection(x0, shift):
    # Define function whose root gives the intersection
    def func(x):
        return l(x, x0) - (xi(x) + shift)
    
    # Use fsolve to find the root (i.e., intersection point)
    x_intersection = fsolve(func, x0)[0]
    return x_intersection, xi(x_intersection) + shift


def parse_xyz(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    blocks = []
    i = 0
    while i < len(lines):
        num_atoms = int(lines[i].strip())  # Read number of atoms
        i += 1
        second_line = lines[i].strip().split()
        i += 1  # Skip the lattice and origin line

        # Parse each block of atom data
        block_atoms = []
        for j in range(num_atoms):
            parts = lines[i].split()
            atom_id = int(parts[0])
            x, y, z = map(float, parts[1:4])
            species = int(parts[4])  # Convert species to int
            molecule = int(parts[5])  # Convert molecule ID to int
            block_atoms.append([atom_id, x, y, z, species, molecule])
            i += 1

        blocks.append(block_atoms)

    # Convert the blocks list into a 3D numpy array (blocks, atoms, atom_data)
    blocks_array = np.array(blocks, dtype=float)
    return blocks_array


def get_max_y_in_bins(r, num_bins=10, xmin=False, xmax=False):

    # Extract x and y positions from the atom block
    x_positions = r[:, 0]  # x positions
    y_positions = r[:, 1]  # y positions

    # Create equally spaced bins in the x range
    if not xmin:
        xmin = np.min(x_positions)
    if not xmax:
        xmax = np.max(x_positions)
    bins = np.linspace(xmin, xmax, num_bins + 1)

    # Digitize the x positions into bins
    bin_indices = np.digitize(x_positions, bins)

    max_y_indices = []
    
    # Loop through each bin and find the index of the max y value in that bin
    for i in range(1, num_bins + 1):
        # Get the indices of atoms in the current bin
        indices_in_bin = np.where(bin_indices == i)[0]
        
        if len(indices_in_bin) > 0:
            # Find the index of the maximum y value in this bin
            max_y_index_in_bin = indices_in_bin[np.argmax(y_positions[indices_in_bin])]
        else:
            continue
            #max_y_index_in_bin = None  # If no atoms in the bin, assign None
        
        max_y_indices.append(max_y_index_in_bin)

    # Convert to numpy array for easier manipulation
    return max_y_indices, bins


def set_plot_parameters():
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams['font.size'] = 26
    plt.rcParams['axes.labelsize'] = 26
    plt.rcParams['font.family'] = 'Times New Roman' #'sans-serif'
    plt.rcParams['axes.linewidth'] = 2  # set the value globally
    plt.rcParams['lines.linewidth'] = 2  # set the value globally
    plt.rcParams['xtick.major.size'] = 8  # Major tick size for x-axis
    plt.rcParams['xtick.minor.size'] = 5  # Minor tick size for x-axis
    plt.rcParams['ytick.major.size'] = 8  # Major tick size for y-axis
    plt.rcParams['ytick.minor.size'] = 5  # Minor tick size for y-axis
    # Increase the width of major and minor ticks on both axes
    plt.rcParams['xtick.major.width'] = 2  # Major tick width for x-axis
    plt.rcParams['xtick.minor.width'] = 1  # Minor tick width for x-axis
    plt.rcParams['ytick.major.width'] = 2  # Major tick width for y-axis
    plt.rcParams['ytick.minor.width'] = 1  # Minor tick width for y-axis
    # Increase the size of major and minor tick labels on both axes
    plt.rcParams['xtick.major.pad'] = 8  # Distance of major tick labels from the axis for x-axis
    plt.rcParams['xtick.minor.pad'] = 8  # Distance of minor tick labels from the axis for x-axis
    plt.rcParams['ytick.major.pad'] = 8  # Distance of major tick labels from the axis for y-axis
    plt.rcParams['ytick.minor.pad'] = 8  # Distance of minor tick labels from the axis for y-axis
    
plot_mols = False #True
plot_trajectory =  False
sumJ = True
dy = 40 
Nbins = 15 #Should be odd
testbin = int(Nbins/2)
eps = 1e-8
mi = 1.  #Mass of the head group molecule

# Read and parse the XYZ data from the file
filename = "filtered-freq-1.xyz"
#filename = "filtered-head-n-tail.xyz"
data = parse_xyz(filename)
Nsteps = data.shape[0]
N = data.shape[1]

#Set domain
x_min = data[...,1].min()
x_max = data[...,1].max()
x_mid = 0.5*(x_max + x_min)
Lx = x_max - x_min
dx = Lx/Nbins
binedges = np.linspace(x_min, x_max, Nbins+1)

#Order based on molecular number (assumes same molecules in all cases)
for t in range(Nsteps-1):
    #Then assign temp varible, sort and resave in data
    tdata = data[t, :, :]
    data[t, :, :] = tdata[tdata[:, 0].argsort()]

    #After first timestep, check molecules in this
    #step are same as last one
    if t>0:
        for i in range(N):
            assert data[t, i,0] == data[t-1,i,0]

#Pad out to have two more columns
data = np.c_[data, np.zeros(list(data.shape[:2])+[2])]

#Loop over all data
r = np.zeros([data.shape[1],2])


if plot_mols:
    fig, ax = plt.subplots(2,1,figsize=(5,6))
    set_plot_parameters()
    fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([.85, 0.15, 0.05, 0.7])
    plt.ion()
    plt.show()
    ft = True

Jsum_t = []; binsum_t = []; dSi_t = []; splines = []
k=0 

for t in range(Nsteps):

    #Get molecules data
    r[:,0] = data[t,:,1] #x part
    r[:,1] = data[t,:,3] #z is the normal, not y but 2D analysis used here

    #================================
    #Fit spline surface
    #================================
    #Get a bunch of points maximum in local bin to fit spline to
    indices, bin_edges = get_max_y_in_bins(r, 20)#, xmin=400, xmax=600)

    # Surface points
    x_surface = r[indices,0] #x_liquid[indices]
    y_surface = r[indices,1] #y_liquid[indices]

    #Use low weight to give only an approximate fitting
    spline = UnivariateSpline(x_surface, y_surface, w=0.1*np.ones(y_surface.shape[0]), s = 0.7, k=2)

    # Generate smooth x values for plotting the spline
    x_smooth = np.linspace(x_min, x_max, 30)
    y_spline = spline(x_smooth)

    if plot_mols:
        # Plot the interface
        ax[0].plot(x_smooth, y_spline, color='orange', linewidth=3, label='Fitted Spline to Surface Molecules')
        ax[0].plot(x_smooth, y_spline-dy/2., color='orange', linewidth=3, label='Fitted Spline to Surface Molecules')
        ax[0].plot(x_smooth, y_spline+dy/2., color='orange', linewidth=3, label='Fitted Spline to Surface Molecules')


    #Add to array to plot at end
    #splines.append(y_spline)

    #================================
    #Create an element per volume
    #================================
    elements = []; Jelements = []
    for bin_indx in range(Nbins):
        bi = int(bin_indx)

        x_cell = x_min + (bin_indx+0.5)*dx
        x0_neg = x_cell-dx/2.
        x0_pos = x_cell+dx/2.
        x = np.linspace(x0_neg,x0_pos,10)

        # Intersection for positive dx/2
        x_inter_pos_top, y_inter_pos_top = find_intersection(x0_pos, dy/2)
        x_inter_pos_bottom, y_inter_pos_bottom = find_intersection(x0_pos, -dy/2)

        # Intersection for negative dx/2
        x_inter_neg_top, y_inter_neg_top = find_intersection(x0_neg, dy/2)
        x_inter_neg_bottom, y_inter_neg_bottom = find_intersection(x0_neg, -dy/2)

        if plot_mols:

            ax[0].plot(x_inter_pos_top, y_inter_pos_top, 'go', label='Top right corner')
            ax[0].plot(x_inter_pos_bottom, y_inter_pos_bottom, 'go', label='Bottom right corner')
            ax[0].plot(x_inter_neg_top, y_inter_neg_top, 'go', label='Top left corner')
            ax[0].plot(x_inter_neg_bottom, y_inter_neg_bottom, 'go', label='Bottom left corner')

            x_ = np.linspace(x_inter_pos_bottom, x_inter_pos_top, 2)
            ax[0].plot(x_, l(x_, x0_pos), 'b-')
            x_ = np.linspace(x_inter_neg_bottom, x_inter_neg_top, 2)
            ax[0].plot(x_, l(x_, x0_neg), 'b-')
            
        # Get coordinates of element can construct element object
        #Use standard FE order 
        #Node 1: Bottom-left (x1,y1)(x1​,y1​)
        #Node 2: Bottom-right (x2,y2)(x2​,y2​)
        #Node 3: Top-right (x3,y3)(x3​,y3​)
        #Node 4: Top-left (x4,y4)(x4​,y4​)
        x_corners = np.array([x_inter_neg_bottom, x_inter_pos_bottom, x_inter_pos_top, x_inter_neg_top])
        y_corners = np.array([y_inter_neg_bottom, y_inter_pos_bottom, y_inter_pos_top, y_inter_neg_top])

        element = BilinearElement2D(x_corners, y_corners)
        elements.append(element)

        #Plot boxes around elements
        if plot_mols:
            ax[0].plot(list(x_corners) + [x_corners[0]], 
                       list(y_corners) + [y_corners[0]], 'k-')

            #Map to chi omega space to check gives squares
            nshift = Nbins-1-2*bi
            chis = []; omegas = []
            for xp, yp in zip(x_corners, y_corners):
                chi, omega = element.map_to_reference(xp, yp)
                chis.append(chi); omegas.append(omega)
            ax[1].plot(np.array(chis)-nshift, omegas, 'k-o')

            if bin_indx == 0:
                ax[1].plot([-Nbins,Nbins], [-1,-1], 'k-')

    #Get molecules in boxes
    mask = np.abs(r[:,1] - spline(r[:,0])) < dy/2. 
    indices = np.where(mask)[0]
    point_on_surface = np.column_stack((r[indices,0], r[indices,1]))
    num_vapor = point_on_surface.shape[0]

    #Loop over all molecules in the boxes
    binsum = np.zeros(Nbins)
    Jsum = np.zeros(Nbins)
    dSi = np.zeros((Nbins,2))
    for indx in indices:
        rp = data[t, indx,[1,3]]
        if plot_trajectory:
            rp_pdt = data[t+1, indx,[1,3]]
            ax[0].plot([rp[0], rp_pdt[0]], [rp[1], rp_pdt[1]], 'r-', alpha=0.2)
            ax[0].plot(rp_pdt[0], rp_pdt[1], 'kx')

        #Binning to see which volume molecules are in
        bi = int(np.floor((rp[0]-x_min-eps)/dx))

        #Then check they are actually between tangent lines, correct bi if not
        if elements[bi].is_point_inside(rp[0], rp[1]):
            pass
        elif elements[(bi-1)%Nbins].is_point_inside(rp[0], rp[1]):
            bi = (bi-1)%Nbins
        elif elements[(bi+1)%Nbins].is_point_inside(rp[0], rp[1]):
            bi = (bi+1)%Nbins
        else:
            print("Molecule ", indx, rp[0], rp[1] ,"  has been absorbed into bulk")

        element = elements[bi]
        chi, omega = element.map_to_reference(rp[0], rp[1])
        J = element.jacobian(chi, omega)
        Ji = np.linalg.det(J)
        if t > 0:
            Ji_mdt = data[t-1, indx, 7]

        #If we add up the mass times Jacobian (sumJ=True) 
        #or just mass (sumJ=False)
        if sumJ:
            moladd = mi * Ji
            if t > 0:
                flux = Ji_mdt
        else:
            moladd = mi 
            flux = mi
            
        #Store tally for each CV
        binsum[bi] += moladd
        if (t>0):
            Jsum[bi] += mi * (Ji - Ji_mdt)
        else:
            Jsum[bi] = 0.

        #Code here plots original and mapped molecule
        if plot_mols and elements[testbin].is_point_inside(rp[0], rp[1]):
            ax[0].plot(rp[0], rp[1], 'ro', alpha=0.6, ms=4,mec='k') #'k.', alpha=0.6)
            ax[1].plot(chi, omega, 'ro', alpha=0.6, ms=4,mec='k')
            
            
            ax[0].set_ylim([0,100])  
            ax[0].set_xlim([275,700]) 
            ax[0].set_xticks([300,400,500,600,700])
            ax[0].set_yticks([0,50,100])
            
            ax[1].set_xlim([-16,16]) 
            ax[1].set_ylim([-1.5,1.5])  
            ax[1].set_xticks([-15,-10,-5,0,5,10,15])  
 
            
        #Overwrite species (identical and 1) with bin index
        data [t, indx, 6] = bi

        #Overwrite molecule (not used) with Ji
        data [t, indx, 7] = Ji

        #dSi if index changes must be surface crossing
        if t > 0:
            binchange = data[t, indx, 6]-data[t-1, indx, 6]
            if np.abs(binchange) != 0:
                bim1 = int(data[t-1, indx, 6])
                #Crossing bottom surface
                if binchange > 0:
                    dSi[bi,0] += flux
                    dSi[bim1,0] -= flux
                #Crossing top surface
                elif binchange < 0:
                    dSi[bi,1] += flux
                    dSi[bim1,1] -= flux
                else:
                    raise IOError("Error")

    #Check that binning operation adds up to total number
    #print(np.sum(binsum), num_vapor, Jsum)
    binsum_t.append(binsum)
    Jsum_t.append(Jsum)
    dSi_t.append(dSi)

    if plot_mols:
        #Plot molecules, coloured by Jacobian
        cm = ax[0].scatter(r[:, 0], r[:, 1], c=data[t, :, 7], alpha=0.6, label='Liquid Molecules')
        #cbar_ax.cla() 
        #fig.colorbar(cm, cax=cbar_ax)
        cm.set_clim(248, 254)  # Set the color range between 200 and 300
        # Labels and Title
        #plt.axis("equal") # Normals don't look normal on scaled axes
        #ax[0].set_ylim([20, 100])
        
         
        # Save the figure for each time step
        figname = f'./figs/surface_plot_{t:04d}.jpg'
        #plt.savefig(figname,dpi=300) 
        
        plt.show()  
        
        plt.pause(0.0001)
        [a.cla() for a in ax]
        
 
###############
# Per atom values
#Get time derivatives of molecular Jacobian
dJidt = data[:,:,7]
#Change in box is a molecular surface crossing
dSi = np.diff(data[:,:,6],axis=0)

###############
# Box values
#Change in total mass in box
dNdt = np.diff(np.array(binsum_t),axis=0)
#Change in summed Jacobian in box
dJdt = np.array(Jsum_t)

#Plot for a given box - this logic is not correct yet
#d/dt mi Ji CV_i  - mi vi Ji dSi + mi dJi/dt CV_i = 0



plt.ioff
#"""
fig, ax = plt.subplots(1,1)
box = testbin
ax.plot(dNdt[:,box], '-r', label=r"$\frac{d}{dt} \displaystyle\sum_{i=1}^N m_i J_i \vartheta_i$")
dSi_t = np.array(dSi_t)
ax.plot(dSi_t[1:,box,0]+dSi_t[1:,box,1], 'go', alpha=1, ms=9,mec='k', label=r"$\displaystyle\sum_{i=1}^N m_i J_i \boldsymbol{\dot{s}}_i d\textbf{S}_i$")
ax.plot(10*dJdt[1:,box], 'b--', label=r"$10 \times \displaystyle\sum_{i=1}^N m_i \dot{J}_i \vartheta_i$", zorder=10)
ax.plot(-dNdt[:,box] + dSi_t[1:,box,0]+dSi_t[1:,box,1] + dJdt[1:,box], 'k-', label="Sum")
#plt.legend()
plt.xlim(0,50)
plt.ylim(-2000,1000)
figname = './figs/surface_plot.jpg'
plt.savefig(figname,dpi=300) 
plt.show()
#"""

#in_box = data[:,:,4]==box

#dJdt/Jsum - dSi + dJidt/Jsum = 0





