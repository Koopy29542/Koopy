import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load Koopman matrix K
with open('koopy_3.pkl', 'rb') as f:
    model = pickle.load(f)
K = model['K']

# Eigendecomposition
eigenvalues, _ = np.linalg.eig(K)

# Plot
plt.figure(figsize=(6, 6))
plt.scatter(eigenvalues.real, eigenvalues.imag, color='blue', label='Eigenvalues', s=30)

# Unit circle
unit_circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--',
                         linewidth=3, label='Unit circle')
ax = plt.gca()
ax.add_artist(unit_circle)


# Axes setup
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
# plt.xlim(0.85, 1.15)
# plt.ylim(-0.15, 0.15)
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.gca().set_aspect('equal')

# Enlarged fonts
plt.xlabel('Re(λ)', fontsize=20)
plt.ylabel('Im(λ)', fontsize=20)
plt.tick_params(axis='both', labelsize=15)
plt.grid(True)
# plt.grid(False)


# Optional: add legend and title
# plt.title('Koopman Eigenvalues on Complex Plane', fontsize=16)
# plt.legend(fontsize=12)

plt.tight_layout()
plt.show()
