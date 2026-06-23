import numpy as np
from scipy.integrate import solve_ivp

# Physical Constants (SI Units)
q = 1.602176634e-19        # Proton charge (C)
m = 1.672621923e-27        # Proton mass (kg)
E_vec = np.array([0.0, 1000.0, 0.0])  # Electric field E (V/m)
B_vec = np.array([0.0, 0.0, 0.5])     # Magnetic field B (T)
v_0 = np.array([10000.0, 0.0, 10000.0]) # Initial velocity (m/s)

# Lorentz Force ODEs
# state vector: [x, y, z, vx, vy, vz]
def lorentz_ode(t, state):
    x, y, z, vx, vy, vz = state
    
    # Lorentz Force: F = q * (E + v x B)
    # v x B = [vy * Bz - vz * By, vz * Bx - vx * Bz, vx * By - vy * Bx]
    ax = (q / m) * (E_vec[0] + vy * B_vec[2] - vz * B_vec[1])
    ay = (q / m) * (E_vec[1] + vz * B_vec[0] - vx * B_vec[2])
    az = (q / m) * (E_vec[2] + vx * B_vec[1] - vy * B_vec[0])
    
    return [vx, vy, vz, ax, ay, az]

# Trajectory solver bounds (t = [0, 10 microseconds])
t_span = (0.0, 10e-6)
y0 = [0.0, 0.0, 0.0, v_0[0], v_0[1], v_0[2]]

# Solve using high precision RK45 adaptive steps
sol = solve_ivp(lorentz_ode, t_span, y0, method='RK45', rtol=1e-7, atol=1e-9)

# Fit a straight line to x(t) over the entire interval to filter out
# periodic cyclotron oscillations and extract the clean drift velocity.
t_steps = sol.t
x_steps = sol.y[0]
numerical_drift_vx, _ = np.polyfit(t_steps, x_steps, 1)

# Theoretical E x B Drift Velocity calculation
# v_d = (E x B) / B^2
E_cross_B = np.cross(E_vec, B_vec)
B_squared = np.dot(B_vec, B_vec)
theoretical_drift_vector = E_cross_B / B_squared
theoretical_drift_vx = theoretical_drift_vector[0]

# Rel difference verification
rel_diff = abs(numerical_drift_vx - theoretical_drift_vx) / theoretical_drift_vx

print(f"--- E x B Drift Verification Results ---")
print(f"Numerical Drift Velocity (Slope fit): {numerical_drift_vx:.6f} m/s")
print(f"Theoretical E x B Drift Velocity:      {theoretical_drift_vx:.6f} m/s")
print(f"Relative Difference:                   {rel_diff:.4e}")

# Assert within 1e-3 relative tolerance
assert rel_diff < 1e-3, f"Assertion failed: Relative difference {rel_diff} is >= 1e-3"
print("VERIFIED")
