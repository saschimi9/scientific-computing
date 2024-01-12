import numpy as np
from helmholtz_problem import f_rhs, analytical_solution, create_discretized_helmholtz_matrix
import helmholtz_solvers
from timeit import default_timer as timer
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
    grid_sizes = [100]  # Example grid sizes
    plt.figure(figsize=(12, 8))
    c = 60

    for grid_size in grid_sizes:
        h = 1/grid_size
        x = np.linspace(0, 1, grid_size+1)
        x = x[1:-1]
        A_h = create_discretized_helmholtz_matrix(size=grid_size, c=c)/h**2
        rhs = f_rhs(c, x, h)

        # Solve the Helmholtz equation
        start_time = timer()
        u = np.linalg.solve(A_h, rhs)
        end_time = timer()
        print(f'time spent: {end_time-start_time:.2g}')

        u_gs, convergence_flag = helmholtz_solvers.gauss_seidel_matrix(
            A_h, rhs)
        # u_ssor, convergence_flag_ssor = helmholtz_solvers.ssor_solver(
        #     A_h, rhs, omega=1.7)

        # M_sgs_inv = helmholtz_solvers.compute_symmetric_ssor_M_inv(
        #     A_h, 1.0)
        # u_cg_sgs, convergence_flag_cg = helmholtz_solvers.preconditioned_conjugate_gradient(
        #     A_h, rhs, M_inv=M_sgs_inv)
        # u_cgc_dir, convergence_flag_cgc_dir = helmholtz_solvers.two_grid_method_point_wise(
        #     A_h, rhs, num_presmoothing_iter=5, num_postsmoothing_iter=5, internal_solver='direct')
        u_cgc, convergence_flag_cgc = helmholtz_solvers.coarse_grid_correction(
            A_h, rhs)
        u_tgm, convergence_flag_tgm = helmholtz_solvers.two_grid_method_matrix(
            A_h, rhs)

        # Compute the analytical solution
        u_exact = analytical_solution(x)

        # Compute and print the RMSE
        rmse = calculate_rmse(u, u_exact)
        rmse_gs = calculate_rmse(u, u_gs)
        # rmse_cg_sgs = calculate_rmse(u, u_cg_sgs)
        # rmse_ssor = calculate_rmse(u, u_ssor)
        rmse_cgc = calculate_rmse(u, u_cgc)
        rmse_tgm = calculate_rmse(u, u_tgm)
        # print(f'RMSE for h = {h}: {rmse:.2E}')
        # Plot the numerical and analytical solutions
        plt.plot(x, u, label=f'Direct (h = {h}), rmse: {rmse:.2E}')
        plt.plot(x, u_gs, label=f'GS (h = {h}), rmse: {rmse_gs:.2E}')
        # plt.plot(x, u_ssor, label=f'SSOR (h = {h}), rmse: {rmse_ssor:.2E}')
        # plt.plot(
        #     x, u_cg_sgs, label=f'Prec. CG (h = {h}), rmse: {rmse_cg_sgs:.2E}')
        plt.plot(
            x, u_cgc, label=f'CGC (h = {h}), rmse: {rmse_cgc:.2E}')
        plt.plot(
            x, u_tgm, label=f'TGM dir. (h = {h}), rmse: {rmse_tgm:.2E}')

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
