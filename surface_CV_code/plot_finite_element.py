import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from plot_settings import set_plot_parameters

# Initialize the bilinear element class (same as before)
class BilinearElement2D:
    def __init__(self, x_coords, y_coords):
        assert len(x_coords) == 4 and len(y_coords) == 4, "There must be exactly 4 corner points."
        self.x_coords = np.array(x_coords)
        self.y_coords = np.array(y_coords)

    def shape_functions(self, xi, eta):
        N1 = 0.25 * (1 - xi) * (1 - eta)
        N2 = 0.25 * (1 + xi) * (1 - eta)
        N3 = 0.25 * (1 + xi) * (1 + eta)
        N4 = 0.25 * (1 - xi) * (1 + eta)
        return np.array([N1, N2, N3, N4])

    def map_to_physical(self, xi, eta):
        N = self.shape_functions(xi, eta)
        x = np.dot(N, self.x_coords)
        y = np.dot(N, self.y_coords)
        return x, y

    def jacobian(self, xi, eta):
        N_xi = np.array([
            -0.25 * (1 - eta),  0.25 * (1 - eta),
             0.25 * (1 + eta), -0.25 * (1 + eta)
        ])
        N_eta = np.array([
            -0.25 * (1 - xi), -0.25 * (1 + xi),
             0.25 * (1 + xi),  0.25 * (1 - xi)
        ])
        J11 = np.dot(N_xi, self.x_coords)
        J12 = np.dot(N_xi, self.y_coords)
        J21 = np.dot(N_eta, self.x_coords)
        J22 = np.dot(N_eta, self.y_coords)
        jacobian = np.array([[J11, J12], [J21, J22]])
        if np.abs(np.linalg.det(jacobian)) < 1e-8:
            raise ValueError("Jacobian is near singular. Check quadrilateral shape or try different initial guess.")
        
        return jacobian

#    def map_to_reference(self, x, y, tol=1e-6, max_iter=1000, plot_coverge=True):
#        xi, eta = 0.0, 0.0  # Initial guess
#        if plot_coverge:
#            fig, ax = plt.subplots(1,1); deltas=[]
#        for _ in range(max_iter):
#            x_map, y_map = self.map_to_physical(xi, eta)
#            res_x = x_map - x
#            res_y = y_map - y
#            res = np.array([res_x, res_y])
#            if np.linalg.norm(res) < tol:
#                break
#            J = self.jacobian(xi, eta)
#            delta = np.linalg.solve(J, -res)
#            if plot_coverge:
#                deltas.append(np.abs(delta))

#            xi += delta[0]
#            eta += delta[1]
#        
#        if plot_coverge:
#            ax.plot(np.array(deltas)[:,0], 'ro-')
#            ax.plot(np.array(deltas)[:,1], 'bo-')
#            plt.yscale("log")
#            plt.show()
#        return xi, eta


    def map_to_reference(self, x, y):

        # Use a numerical approach to find (xi, eta) that corresponds to the given (x_scaled, y_scaled)

        def objective(params):
            xi, eta = params
            mapped_x, mapped_y = self.map_to_physical(xi, eta)
            return (mapped_x - x) ** 2 + (mapped_y - y) ** 2

        # Start the optimization at the center of the reference square
        result = minimize(objective, [0.0, 0.0], bounds=[(-1.01, 1.01), (-1.01, 1.01)])
        if result.success:
            return result.x  # Returns (xi, eta)
        else:
            print("Warning, Could not find a suitable reference", x,y)
            return [np.NaN, np.NaN]
            #raise ValueError("Could not find a suitable reference point.")

#    def is_point_inside(self, x, y):
#        # Using Ray Casting Algorithm to check if the point is inside the quadrilateral
#        n = 4  # Number of vertices (quadrilateral has 4)
#        inside = False
#        p1x, p1y = self.x_coords[0], self.y_coords[0]
#        for i in range(n + 1):
#            p2x, p2y = self.x_coords[i % n], self.y_coords[i % n]
#            if y > min(p1y, p2y):
#                if y <= max(p1y, p2y):
#                    if x <= max(p1x, p2x):
#                        if p1y != p2y:
#                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
#                        if p1x == p2x or x <= xinters:
#                            inside = not inside
#            p1x, p1y = p2x, p2y
#        return inside

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


# Example usage
# Define the corner points of the quadrilateral in the physical domain
#x_coords = np.array([+0.1, 1.8, 1.8, -0.2])
#y_coords = np.array([0.0, 0.0, 1.0, 0.7])

x_coords = np.array([298.73979983, 302.43430025, 426.53619708, 424.02516767])
y_coords = np.array([43.57682556, 78.18239975, 67.13639773, 32.31748518])


#y_coords = np.array([78.18239975, 43.57682556, 32.31748518, 67.13639773])

# Create the bilinear element
element = BilinearElement2D(x_coords, y_coords)
is_convex = element.is_convex_quadrilateral()
print("Is the quadrilateral convex?", is_convex)

# Map the physical coordinates back to reference space
reference_coords = [element.map_to_reference(x, y) for x, y in zip(x_coords, y_coords)]
xi, eta = zip(*reference_coords)
xi = list(xi); eta = list(eta)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
set_plot_parameters()

N = 20
chip = np.linspace(-1.5,1.5,N)
omegap = np.linspace(-1.5,1.5,N)
Chi, Omega = np.meshgrid(chip, omegap)
detJ = np.zeros([N,N])
for i in range(N):
    for j in range(N):
        J = element.jacobian(Chi[i,j], Omega[i,j])
        detJ[i,j] = np.abs(np.linalg.det(J))

cm = ax.pcolormesh(Chi, Omega, detJ)
ax.plot(xi+[xi[0]], eta+[eta[0]], 'ko-')
plt.colorbar(cm)
plt.show()

# Create the plots 
fig, axs = plt.subplots(1, 2, figsize=(10, 5)) 
set_plot_parameters() 

# Plot the quadrilateral in the physical space 
axs[0].plot(list(x_coords[:]) + [x_coords[0]],  
            list(y_coords[:]) + [y_coords[0]],  
            'o-', lw=4, label='Physical Element') 
axs[0].set_xlim(250,490)
axs[0].set_ylim(25,90)

axs[0].set_xticks([250,300,350,400,450])

for n in range(4):
    axs[0].text(x_coords[n], y_coords[n], str(n)) 

# Plot the quadrilateral in the reference space (unit square) 
axs[1].plot(xi+[xi[0]], eta+[eta[0]], 'o-') 


xp = np.linspace(np.min(x_coords)*0.9,np.max(x_coords)*1.1,N)
yp = np.linspace(np.min(y_coords)*0.9,np.max(y_coords)*1.1,N)
X, Y = np.meshgrid(xp, yp)
mask = element.is_point_inside(X, Y)
axs[0].scatter(X, Y, c=mask)

for i in range(N):
    for j in range(N):
        if element.is_point_inside(X[i,j], Y[i,j]):
            eta, xi = element.map_to_reference(X[i,j], Y[i,j])
            print(i,j, X[i,j], Y[i,j], eta, xi)
            axs[1].plot(eta, xi, 'ko')
plt.xlim(-1.25,1.25)
plt.ylim(-1.25,1.25)
plt.savefig('test.jpg',dpi=300)
plt.show()
