import numpy as np
from take_home_exam import compute_symmetric_ssor_preconditioner
from helmholtz_problem import f_rhs, analytical_solution, create_discretized_helmholtz_matrix
import helmholtz_solvers
import direct_solution

import numpy as np
import matplotlib.pyplot as plt


def calculate_rmse(approximate, exact):
    """Calculate the root mean square error (RMSE) between approximate and exact solutions."""
    return np.sqrt(np.mean((approximate - exact)**2))


def plot_comparison_plot(x, u_exact, u_solver):
    plt.figure(figsize=(12, 8))
    # Only plot the last exact solution
    plt.plot(x, u_solver, label=f'Solver')
    plt.plot(x, u_exact, label='Analytical', linestyle='dashed')
    # Finalize the plot
    plt.title('Helmholtz Equation: Numerical vs Analytical Solutions')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid(True)


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
        u, _ = direct_solution.helmholtz_direct_solver(grid_size, h, c=c)

        A = create_discretized_helmholtz_matrix(size=grid_size, c=c)
        rhs = f_rhs(c, x, h)
        u_gs, rel_errors, convergence_flag = helmholtz_solvers.gauss_seidel_solver(
            A / h**2, rhs)

        M_sgs_inv = compute_symmetric_ssor_preconditioner(A/h**2, 1.0)
        # M_sgs_inv = np.eye(A.shape[0])
        ritz_values = []
        u_sgs_cg, convergence_flag_cg = helmholtz_solvers.preconditioned_conjugate_gradient_with_ritz(
            A/h**2, rhs, M_inv=M_sgs_inv, ritz_values=ritz_values)

        # Compute the analytical solution
        u_exact = analytical_solution(x)

        # Compute and print the RMSE
        rmse = calculate_rmse(u, u_exact)
        rmse_gs = calculate_rmse(u, u_gs)
        rmse_sgs_cg = calculate_rmse(u, u_sgs_cg)
        print(f'RMSE for h = {h}: {rmse:.2E}')
        # Plot the numerical and analytical solutions
        plt.plot(x, u, label=f'Direct (h = {h}), rmse: {rmse:.2E}')
        plt.plot(x, u_gs, label=f'GS (h = {h}), rmse: {rmse_gs:.2E}')
        plt.plot(
            x, u_sgs_cg, label=f'Prec. CG (h = {h}), rmse: {rmse_sgs_cg:.2E}')

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
