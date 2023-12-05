import numpy as np
from take_home_exam import f_rhs, analytical_solution, create_discretized_helmholtz_matrix

import numpy as np
import matplotlib.pyplot as plt


def helmholtz_solver(size, h, c):
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
    f = f_rhs(c=0.1, x=x, h=h)

    # Solve for u
    u = np.linalg.solve(A / h**2, f)

    return u, x


def calculate_rmse(approximate, exact):
    """Calculate the root mean square error (RMSE) between approximate and exact solutions."""
    return np.sqrt(np.mean((approximate - exact)**2))


if __name__ == "__main__":
    # Parameters
    grid_sizes = [10, 100, 1000]  # Example grid sizes
    plt.figure(figsize=(12, 8))

    for grid_size in grid_sizes:
        h = 1/grid_size

        # Solve the Helmholtz equation
        u, x = helmholtz_solver(grid_size, h, c=0.1)

        # Compute the analytical solution
        u_exact = analytical_solution(x)

        # Plot the numerical and analytical solutions
        plt.plot(x, u, label=f'Numerical (h = {h})')
        plt.plot(x, u_exact, label='Analytical', linestyle='dashed')

        # Compute and print the RMSE
        rmse = calculate_rmse(u, u_exact)
        print(f'RMSE for h = {h}: {rmse:.6f}')

    # Finalize the plot
    plt.title('Helmholtz Equation: Numerical vs Analytical Solutions')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid(True)
    plt.show()
