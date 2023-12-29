import numpy as np
from take_home_exam import compute_eigenvals_gauss_seidel_error_propagation_matrix, f_rhs, analytical_solution, create_discretized_helmholtz_matrix

import numpy as np
import matplotlib.pyplot as plt


def helmholtz_direct_solver(size, h, c):
    """
    Solve the Helmholtz equation using finite differences.

    Parameters:
    - size: Number of grid points (excluding boundary).
    - h: Step size = 1/N

    Returns:
    - u: Discretized solution.
    - x: Grid points.
    """
    # Set up the Helmholtz matrix
    x = np.linspace(0, 1, size+1)
    x = x[1:-1]
    A = create_discretized_helmholtz_matrix(size=size, c=c)

    if A.shape[0] != x.size:
        raise ValueError("Sizes do not match. A:", A.shape, " x:", x.size)

    # Discretized Helmholtz equation: A*u = f
    f = f_rhs(c=c, x=x, h=h)

    # Solve for u
    u = np.linalg.solve(A / h**2, f)

    return u, x


def calculate_rmse(approximate, exact):
    """Calculate the root mean square error (RMSE) between approximate and exact solutions."""
    return np.sqrt(np.mean((approximate - exact)**2))


def plot_all():
    # Parameters
    grid_sizes = [10, 100]  # Example grid sizes
    plt.figure(figsize=(12, 8))
    c = 10

    for grid_size in grid_sizes:
        h = 1/grid_size
        x = np.linspace(0, 1, grid_size+1)
        x = x[1:-1]

        # Solve the Helmholtz equation
        u, _ = helmholtz_direct_solver(grid_size, h, c=c)

        A = create_discretized_helmholtz_matrix(size=grid_size, c=c)
        rhs = f_rhs(c, x, h)
        u_gs, rel_errors = gauss_seidel_solver(A / h**2, rhs)

        # Compute the analytical solution
        u_exact = analytical_solution(x)

        # Compute and print the RMSE
        rmse = calculate_rmse(u, u_exact)
        rmse_gs = calculate_rmse(u, u_gs)
        print(f'RMSE for h = {h}: {rmse:.2E}')
        # Plot the numerical and analytical solutions
        plt.plot(x, u, label=f'Direct (h = {h}), rmse: {rmse:.2E}')
        plt.plot(x, u_gs, label=f'GS (h = {h}), rmse: {rmse_gs:.2E}')

        # eigenvalues and spectral radius
        _ = compute_eigenvals_gauss_seidel_error_propagation_matrix(A)

    # Only plot the last exact solution
    plt.plot(x, u_exact, label='Analytical', linestyle='dashed')
    # Finalize the plot
    plt.title('Helmholtz Equation: Numerical vs Analytical Solutions')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_direct_solver():
    # Parameters
    grid_sizes = [10, 100, 1000]  # Example grid sizes
    plt.figure(figsize=(12, 8))
    c = 0.1

    for grid_size in grid_sizes:
        h = 1/grid_size

        # Solve the Helmholtz equation
        u, x = helmholtz_direct_solver(grid_size, h, c=c)

        # Compute the analytical solution
        u_exact = analytical_solution(x)

        # Compute and print the RMSE
        rmse = calculate_rmse(u, u_exact)
        print(f'RMSE for h = {h}: {rmse:.2E}')
        # Plot the numerical and analytical solutions
        plt.plot(x, u, label=f'Numerical (h = {h}), rmse: {rmse:.2E}')

    # Only plot the last exact solution
    plt.plot(x, u_exact, label='Analytical', linestyle='dashed')
    # Finalize the plot
    plt.title('Helmholtz Equation: Numerical vs Analytical Solutions')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_all()
