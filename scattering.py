"""
Wave Scattering Around a Circular Obstacle - Object-Oriented Implementation

This module implements a class-based solution to compute and visualize acoustic/electromagnetic
wave scattering around a circular disc using the Green's function method and Gauss-Legendre
quadrature for numerical integration.

Author: Firas Boustila
Date: October 2025

Physical Model:
    - Helmholtz equation: ∇²u + k²u = 0
    - Boundary condition: u = 0 on the disc boundary (rigid scatterer)
    - Solution via Green's function method (BEM - Boundary Element Method)
"""

import numpy as np
from scipy.special import hankel1
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class NumericalParameters:
    """
    Container for all numerical parameters of the simulation.
    
    Attributes:
        disc_radius (float): Radius of the scattering disc (a)
        num_boundary_points (int): Number of discretization points on the boundary
        wavenumber (float): Wave number k = 2π/λ (related to frequency)
        quad_order (int): Order of Gauss-Legendre quadrature (2-7)
        num_fourier_modes (int): Number of Fourier modes for incident field
        grid_nx (int): Number of points in x-direction for observation grid
        grid_ny (int): Number of points in y-direction for observation grid
    """
    disc_radius: float
    num_boundary_points: int
    wavenumber: float
    quad_order: int
    num_fourier_modes: int
    grid_nx: int
    grid_ny: int


class GaussLegendreQuadrature:
    """
    Gauss-Legendre quadrature rule for numerical integration on [-1, 1].
    
    This class provides pre-computed abscissas and weights for accurate
    polynomial approximation of integrals.
    
    The quadrature formula approximates:
        ∫_{-1}^{1} f(x) dx ≈ Σ w_i * f(x_i)
    """
    
    # Pre-computed nodes and weights for different orders
    GAUSS_TABLE: Dict[int, Tuple[list, list]] = {
        2: ([-1/np.sqrt(3), 1/np.sqrt(3)], [1, 1]),
        3: ([0, -np.sqrt(3/5), np.sqrt(3/5)], [8/9, 5/9, 5/9]),
        4: ([-0.861136, -0.339981, 0.339981, 0.861136],
            [0.347855, 0.652145, 0.652145, 0.347855]),
        5: ([0, -0.538469, 0.538469, -0.90618, 0.90618],
            [0.568889, 0.478629, 0.478629, 0.236927, 0.236927]),
        6: ([0.661209, -0.661209, -0.238619, 0.238619, -0.93247, 0.93247],
            [0.360762, 0.360762, 0.467914, 0.467914, 0.171324, 0.171324]),
        7: ([0, 0.405845, -0.405845, 0.741531, -0.741531, 0.949108, -0.949108],
            [0.417959, 0.381830, 0.381830, 0.279705, 0.279705, 0.129485, 0.129485])
    }
    
    @staticmethod
    def integrate_segment(func, p1: np.ndarray, p2: np.ndarray, order: int) -> complex:
        """
        Compute the integral of a function over a line segment using Gauss-Legendre quadrature.
        
        The integration transforms from the standard interval [-1, 1] to the physical segment [P1, P2].
        
        Formula: ∫[P1,P2] f(x) dx = (L/2) * Σ wi * f(ξi(xi))
        where L = |P2 - P1|, xi ∈ [-1,1], and ξi maps xi to the segment.
        
        Args:
            func: Function to integrate (takes np.ndarray point as input)
            p1: Starting point of segment
            p2: Ending point of segment
            order: Quadrature order (must be in GAUSS_TABLE)
            
        Returns:
            Complex value of the integral
        """
        x_hat, weights = GaussLegendreQuadrature.GAUSS_TABLE[order]
        p1, p2 = np.array(p1), np.array(p2)
        
        # Segment length
        segment_length = np.linalg.norm(p2 - p1)
        
        # Transform integration points from [-1,1] to [P1, P2]
        # ξ(x) = ((1-x)/2)*P1 + ((1+x)/2)*P2
        physical_points = [((1 - xi) / 2) * p1 + ((1 + xi) / 2) * p2 for xi in x_hat]
        
        # Compute weighted sum
        integral = sum(w * func(pt) for pt, w in zip(physical_points, weights))
        
        # Apply Jacobian for interval transformation
        return integral * segment_length / 2


class GreenFunction:
    """
    2D Green's function for the Helmholtz equation.
    
    The fundamental solution to ∇²u + k²u = 0 in 2D is:
    G(x, y, k) = (i/4) * H₀⁽¹⁾(k|x-y|)
    
    where H₀⁽¹⁾ is the Hankel function of the first kind.
    
    Attributes:
        k (float): Wave number
    """
    
    def __init__(self, wavenumber: float):
        """
        Initialize the Green's function.
        
        Args:
            wavenumber: Wave number k = ω/c (frequency parameter)
        """
        self.k = wavenumber
    
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> complex:
        """
        Evaluate the Green's function at a pair of points.
        
        Args:
            x: Observation point
            y: Source point
            
        Returns:
            G(x, y) = (i/4) * H₀⁽¹⁾(k*r) where r = |x - y|
        """
        distance = np.linalg.norm(np.array(x) - np.array(y))
        return 1j / 4 * hankel1(0, self.k * distance)


class CircularScatterer:
    """
    Represents a circular disc obstacle for wave scattering.
    
    This class handles:
    - Boundary discretization into line segments
    - Boundary node positions
    - Mesh generation on the disc boundary
    """
    
    def __init__(self, radius: float, num_points: int):
        """
        Initialize the circular scatterer.
        
        Args:
            radius: Radius of the circular disc (a)
            num_points: Number of boundary points (determines mesh refinement)
        """
        self.radius = radius
        self.num_points = num_points
        self.boundary_nodes = self._generate_boundary_nodes()
    
    def _generate_boundary_nodes(self) -> np.ndarray:
        """
        Generate uniformly distributed nodes on the circular boundary.
        
        Uses parametric representation: x = a*cos(θ), y = a*sin(θ)
        with θ ∈ [0, 2π).
        
        Returns:
            Array of shape (num_points, 2) containing boundary node coordinates
        """
        theta = np.linspace(0, 2 * np.pi, self.num_points, endpoint=False)
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        return np.column_stack((x, y))
    
    def get_segment_midpoints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute midpoints of boundary segments and their angles.
        
        Returns:
            Tuple of (midpoint_array, angle_array)
            - midpoint_array: coordinates of segment midpoints
            - angle_array: angular position of each midpoint
        """
        x_mid = (self.boundary_nodes[:, 0] + np.roll(self.boundary_nodes[:, 0], -1)) / 2
        y_mid = (self.boundary_nodes[:, 1] + np.roll(self.boundary_nodes[:, 1], -1)) / 2
        theta_mid = np.arctan2(y_mid, x_mid)
        theta_mid = (theta_mid + 2 * np.pi) % (2 * np.pi)
        return np.column_stack((x_mid, y_mid)), theta_mid


class IncidentField:
    """
    Incident wave represented by Fourier-Hankel series expansion.
    
    This class computes the incident field on the boundary of the circular
    scatterer using Hankel function coefficients.
    
    Theory:
        The incident plane wave u_i = e^{ikx} can be expanded in:
        u_i(r,θ) = Σ_{n} i^n J_n(kr) e^{inθ}
        
        At the boundary r=a, the field values are extracted.
    """
    
    def __init__(self, radius: float, wavenumber: float, num_modes: int):
        """
        Initialize the incident field.
        
        Args:
            radius: Disc radius
            wavenumber: Wave number k
            num_modes: Number of Fourier modes (±N)
        """
        self.radius = radius
        self.k = wavenumber
        self.num_modes = num_modes
    
    def compute_boundary_pressure(self, theta_mid: np.ndarray) -> np.ndarray:
        """
        Compute incident pressure on the boundary at midpoint angles.
        
        Algorithm:
        1. For each Fourier mode n from -N to N:
            - Compute Hankel function coefficient: H_n(ka)
            - Compute mode amplitude: (-i)^n / H_n(ka)
            - Add contribution: coef * e^{i*n*θ}
        2. Multiply by normalization factor: (2i)/(π*a)
        
        Args:
            theta_mid: Angular positions of segment midpoints
            
        Returns:
            Complex array of pressure values on the boundary
        """
        num_points = len(theta_mid)
        pressure = np.zeros(num_points, dtype=complex)
        
        # Sum over Fourier modes
        for n in range(-self.num_modes, self.num_modes + 1):
            # Hankel function of order n at ka
            hn = hankel1(n, self.k * self.radius)
            
            # Coefficient for this mode
            coefficient = (-1j) ** n / hn
            
            # Add mode contribution
            pressure += coefficient * np.exp(1j * n * theta_mid)
        
        # Apply normalization
        pressure *= 2j / (np.pi * self.radius)
        
        return pressure


class ScatteringSimulator:
    """
    Main simulator class for wave scattering computation.
    
    Orchestrates:
    1. Geometry setup (circular scatterer)
    2. Boundary discretization
    3. Incident field calculation
    4. Scattered field computation on observation grid
    5. Visualization
    """
    
    def __init__(self, params: NumericalParameters):
        """
        Initialize the scattering simulator.
        
        Args:
            params: NumericalParameters object with all simulation settings
        """
        self.params = params
        self.scatterer = CircularScatterer(params.disc_radius, params.num_boundary_points)
        self.green = GreenFunction(params.wavenumber)
        self.incident_field = IncidentField(
            params.disc_radius,
            params.wavenumber,
            params.num_fourier_modes
        )
        
        # Will be computed
        self.boundary_pressure = None
        self.observation_grid = None
        self.scattered_field = None
    
    def _setup_observation_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create observation grid in the physical domain.
        
        Grid extends from [-3a, 6a] in x and [-3a, 3a] in y to capture
        scattering pattern both upstream and downstream.
        
        Returns:
            Tuple of (X_meshgrid, Y_meshgrid)
        """
        a = self.params.disc_radius
        x = np.linspace(-3 * a, 6 * a, self.params.grid_nx)
        y = np.linspace(-3 * a, 3 * a, self.params.grid_ny)
        return np.meshgrid(x, y)
    
    def compute_incident_field(self) -> None:
        """
        Compute incident wave pressure on the boundary.
        
        This solves the boundary value problem by computing pressure
        at segment midpoints using the Fourier-Hankel expansion.
        """
        _, theta_mid = self.scatterer.get_segment_midpoints()
        self.boundary_pressure = self.incident_field.compute_boundary_pressure(theta_mid)
    
    def compute_scattered_field(self) -> None:
        """
        Compute the scattered field at all observation points.
        
        Algorithm (Boundary Element Method):
        For each observation point x_obs:
            1. Integrate Green's function along each boundary segment
            2. Weight by boundary pressure
            3. Sum contributions from all segments
            
        Mathematical formula:
        u(x_obs) = Σ_{j=1}^{n_segments} [∫_{segment_j} G(x_obs, y) dy] * p_j
        
        where G is the Green's function and p_j is the pressure on segment j.
        """
        if self.boundary_pressure is None:
            self.compute_incident_field()
        
        # Setup observation grid
        X, Y = self._setup_observation_grid()
        self.observation_grid = (X, Y)
        
        # Initialize field array
        U = np.zeros_like(X, dtype=complex)
        total_points = X.shape[0] * X.shape[1]
        
        # Progress bar for monitoring computation
        with tqdm(total=total_points, desc="Computing scattered field", unit="pts") as pbar:
            for ix in range(X.shape[0]):
                for iy in range(X.shape[1]):
                    x_obs = np.array([X[ix, iy], Y[ix, iy]])
                    
                    # Integrate Green's function over all boundary segments
                    segment_integrals = np.zeros(self.params.num_boundary_points, dtype=complex)
                    
                    boundary_nodes = self.scatterer.boundary_nodes
                    for j in range(self.params.num_boundary_points):
                        p1 = boundary_nodes[j]
                        p2 = boundary_nodes[(j + 1) % self.params.num_boundary_points]
                        
                        # Integrate Green's function on this segment
                        segment_integrals[j] = GaussLegendreQuadrature.integrate_segment(
                            lambda y: self.green.evaluate(x_obs, y),
                            p1, p2,
                            self.params.quad_order
                        )
                    
                    # Sum contributions weighted by boundary pressure
                    U[ix, iy] = np.dot(segment_integrals, self.boundary_pressure)
                    pbar.update(1)
        
        self.scattered_field = U
    
    def mask_interior(self) -> np.ndarray:
        """
        Mask the interior of the disc (set to NaN for visualization).
        
        Returns:
            Masked field array where interior points are NaN
        """
        if self.scattered_field is None:
            raise ValueError("Scattered field not computed. Call compute_scattered_field() first.")
        
        X, Y = self.observation_grid
        R = np.sqrt(X**2 + Y**2)
        return np.where(R > self.params.disc_radius, self.scattered_field, np.nan)
    
    def plot_2d_field(self, save_path: str = None, title: str = "Scattered Field |u|") -> None:
        """
        Create a 2D pseudocolor plot of the scattered field magnitude.
        
        Features:
        - High-quality 'cividis' colormap (perceptually uniform)
        - Boundary circle overlay in light yellow
        - Properly scaled aspect ratio
        
        Args:
            save_path: Path to save figure (if None, only display)
            title: Plot title
        """
        if self.scattered_field is None:
            raise ValueError("Scattered field not computed. Call compute_scattered_field() first.")
        
        X, Y = self.observation_grid
        U_masked = self.mask_interior()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot field magnitude
        pcm = ax.pcolormesh(X, Y, np.abs(U_masked), shading='gouraud', cmap='cividis')
        cbar = fig.colorbar(pcm, ax=ax, label='|u| (Field Magnitude)')
        
        # Add boundary circle
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(
            self.params.disc_radius * np.cos(theta),
            self.params.disc_radius * np.sin(theta),
            color='lightyellow',
            linewidth=3,
            label='Scatterer Boundary'
        )
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 2D plot saved: {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_3d_field(self, save_path: str = None, title: str = "Scattered Field 3D") -> None:
        """
        Create a 3D surface plot of the scattered field.
        
        Features:
        - Wireframe surface for clarity
        - Normalized colormap for better visualization
        - Subsampled mesh for performance
        
        Args:
            save_path: Path to save figure (if None, only display)
            title: Plot title
        """
        if self.scattered_field is None:
            raise ValueError("Scattered field not computed. Call compute_scattered_field() first.")
        
        X, Y = self.observation_grid
        U_masked = self.mask_interior()
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Subsample for better visualization (every 4th point)
        step = 4
        X_sub, Y_sub = X[::step, ::step], Y[::step, ::step]
        Z_sub = np.abs(U_masked[::step, ::step])
        
        # Plot surface
        surf = ax.plot_surface(X_sub, Y_sub, Z_sub, cmap='viridis', 
                              edgecolor='none', alpha=0.8)
        
        fig.colorbar(surf, ax=ax, label='|u| (Field Magnitude)', shrink=0.5)
        
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_zlabel('|u|', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 3D plot saved: {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_phase_field(self, save_path: str = None, title: str = "Scattered Field Phase") -> None:
        """
        Create a 2D plot of the scattered field phase (angle in complex plane).
        
        Args:
            save_path: Path to save figure (if None, only display)
            title: Plot title
        """
        if self.scattered_field is None:
            raise ValueError("Scattered field not computed. Call compute_scattered_field() first.")
        
        X, Y = self.observation_grid
        U_masked = self.mask_interior()
        phase = np.angle(U_masked)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        pcm = ax.pcolormesh(X, Y, phase, shading='gouraud', cmap='hsv')
        cbar = fig.colorbar(pcm, ax=ax, label='Phase (radians)')
        
        # Add boundary circle
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(
            self.params.disc_radius * np.cos(theta),
            self.params.disc_radius * np.sin(theta),
            color='white',
            linewidth=3,
            label='Scatterer Boundary'
        )
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.legend(loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Phase plot saved: {save_path}")
        
        plt.tight_layout()
        plt.show()


def main():
    """
    Main execution function demonstrating the full wave scattering simulation.
    
    Simulation Parameters:
    - Disc radius: a = 1.0 (normalized units)
    - Wavenumber: k = 10.0 (corresponds to wavelength λ ≈ 0.628)
    - Boundary discretization: 64 points
    - Fourier modes: N = ±10 (21 modes total)
    - Observation grid: 400×400 points
    - Integration order: 7-point Gauss-Legendre quadrature
    """
    
    print("\n" + "="*70)
    print(" WAVE SCATTERING AROUND CIRCULAR OBSTACLE ")
    print("="*70)
    
    # Define simulation parameters
    params = NumericalParameters(
        disc_radius=1.0,
        num_boundary_points=64,
        wavenumber=10.0,
        quad_order=7,
        num_fourier_modes=10,
        grid_nx=400,
        grid_ny=400
    )
    
    print(f"\n📋 Simulation Parameters:")
    print(f"   • Disc radius: a = {params.disc_radius}")
    print(f"   • Wavenumber: k = {params.wavenumber}")
    print(f"   • Wavelength: λ = {2*np.pi/params.wavenumber:.4f}")
    print(f"   • Boundary points: {params.num_boundary_points}")
    print(f"   • Fourier modes: ±{params.num_fourier_modes}")
    print(f"   • Grid size: {params.grid_nx} × {params.grid_ny}")
    print(f"   • Quadrature order: {params.quad_order}")
    
    # Initialize simulator
    simulator = ScatteringSimulator(params)
    
    print("\n⚙️  Computing scattered field...")
    simulator.compute_scattered_field()
    
    print("\n📊 Field Statistics:")
    field_abs = np.abs(simulator.scattered_field)
    print(f"   • Max magnitude: {np.nanmax(field_abs):.6f}")
    print(f"   • Min magnitude: {np.nanmin(field_abs):.6f}")
    print(f"   • Mean magnitude: {np.nanmean(field_abs):.6f}")
    
    print("\n📈 Generating visualizations...")
    
    # Generate all plots
    simulator.plot_2d_field(
        save_path='scattered_field_2d.png',
        title=f'Scattered Field |u| - k={params.wavenumber}, a={params.disc_radius}'
    )
    
    simulator.plot_3d_field(
        save_path='scattered_field_3d.png',
        title=f'Scattered Field 3D Surface - k={params.wavenumber}'
    )
    
    simulator.plot_phase_field(
        save_path='scattered_field_phase.png',
        title=f'Scattered Field Phase - k={params.wavenumber}'
    )
    
    print("\n" + "="*70)
    print(" ✅ SIMULATION COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
