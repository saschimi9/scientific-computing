
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import helmholtz_solvers
import scipy.linalg as sp_la
from helmholtz_problem import create_discretized_helmholtz_matrix, analytical_solution, f_rhs


def plot_solutions(x, u_exact, *label_and_solutions):
    plt.figure(figsize=(12, 8))
    for label, u_sol in label_and_solutions:
        plt.plot(x, u_sol, label=label)
    plt.plot(x, u_exact, label='Analytical', linestyle='dashed')
    # Finalize the plot
    plt.title('Helmholtz Equation: Numerical vs Analytical Solutions')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_rmse(approximate, exact):
    """Calculate the root mean square error (RMSE) between approximate and exact solutions."""
    return np.sqrt(np.mean((approximate - exact)**2))


def create_coarsening_matrix(size):
    """
    Create a coarsening matrix for a given finer grid size size.

    Parameters:
    - size: Size of the finer grid.

    Returns:
    - A numpy matrix representing the coarsening matrix.
    """
    if size % 2 != 0:
        raise ValueError("The size of the finer grid 'size' should be even.")
    size_fine = size-1
    size_coarse = size_fine // 2

    coarsening_matrix = np.zeros((size_coarse, size_fine))

    # Fill the diagonal with ones
    coarsening_matrix[np.arange(size_coarse),
                      np.arange(0, size_fine - 2, 2)] = 1

    # Fill the upper diagonal with twos
    coarsening_matrix[np.arange(size_coarse),
                      np.arange(1, size_fine - 1, 2)] = 2

    # Fill the second upper diagonal with ones
    coarsening_matrix[np.arange(size_coarse),
                      np.arange(2, size_fine, 2)] = 1

    return coarsening_matrix/4


def create_prolongation_matrix(size):
    """
    Uses `create_coarsening_matrix` and returns the transpose of the matrix
    multiplied by 2
    """
    return 2 * np.transpose(create_coarsening_matrix(size))


def is_symmetric(A):
    return np.allclose(A, np.transpose(A), rtol=1e-5, atol=1e-5)


def compute_rate_of_convergence(residuals):
    n = len(residuals)
    vals = np.zeros(n-2)
    for i in range(2, n):
        val = np.linalg.norm(residuals[i] - residuals[i-1], ord=2) / \
            np.linalg.norm(residuals[i-1] - residuals[i-2], ord=2)
        vals[i-2] = val
    rate_of_convergence = np.average(vals[n-n//2:])
    return rate_of_convergence


def compute_condition_number(A):
    if is_symmetric(A):
        eig_vals = np.linalg.eigvals(A)
        condition_number_A = np.max(np.abs(eig_vals))/np.min(np.abs(eig_vals))
    else:
        raise TypeError(
            "Input A is not symmetric and thus inv(M_ssor)*A is not symmetric")
    return condition_number_A


def compute_effective_condition_number(A):
    if is_symmetric(A):
        eig_vals = np.linalg.eigvals(A)
        eig_vals = np.array(
            [eig_val for eig_val in eig_vals if np.abs(eig_val) > 1e-12])
        condition_number_A = np.max(np.abs(eig_vals))/np.min(np.abs(eig_vals))
    else:
        raise TypeError(
            "Input A is not symmetric and thus inv(M_ssor)*A is not symmetric")
    return condition_number_A


def create_gauss_seidel_error_propagation_matrix(A_h, backwards=False):
    M_gs_inv = helmholtz_solvers.compute_gauss_seidel_M_inverse(
        A_h, backwards=backwards)
    B_gs = np.eye(A_h.shape[0]) - np.matmul(M_gs_inv, A_h)
    return B_gs


def compute_eigenvals_gauss_seidel_error_propagation_matrix(A_h):
    B_gs = create_gauss_seidel_error_propagation_matrix(A_h)
    eigvals_B_gs = np.linalg.eigvals(B_gs)
    spectral_radius = np.max(np.abs(eigvals_B_gs))

    return eigvals_B_gs, spectral_radius


def create_coarse_grid_correction_operator(A_h):
    I_toFine = create_prolongation_matrix(A_h.shape[0]+1)
    I_toCoarse = create_coarsening_matrix(A_h.shape[0]+1)
    A_H = I_toCoarse @ A_h @ I_toFine
    A_H_inv = np.linalg.inv(A_H)

    M_cgc_inv = I_toFine @ A_H_inv @ I_toCoarse
    return M_cgc_inv


def create_coarse_grid_correction_error_propagation_matrix(A_h):
    M_cgc_inv = create_coarse_grid_correction_operator(A_h)
    B_cgc = np.eye(A_h.shape[0]) - M_cgc_inv @ A_h

    return B_cgc


def compute_series_of_ritz_values(sys_mat, residuals):
    """Restructure the Ritz values provided by a Prec. CG method
    into a list of Ritz values that should converge to the eigen
    values of the system matrix if the solver converged.

    Args:
        sys_mat (np.ndarray): NxN system matrix
        residuals (list of np.ndarray): List of residual vectors

    Returns:
        list of lists: i-th entry contains i elements of a sequence of the
        same ritz value
    """
    n_row = len(residuals[0])
    n_col = len(residuals)
    r_ks = np.zeros((n_row, n_col))
    T_k_matrices = []
    ritz_values = []

    for i, residual in enumerate(residuals):
        r_k = np.array(residual)
        r_ks[:, i] = r_k/np.linalg.norm(r_k, ord=2)
        R_k = r_ks[:, 0:i+1]
        T_k_matrices.append(R_k.T @ sys_mat @ R_k)  # R_k^T @ M^{-1} @ A @ R_k

    assert T_k_matrices[0].shape == (1, 1)  # "should be k x N x N x k == 1

    for T_k_matrix in T_k_matrices:
        ritz_values.append(np.linalg.eigvals(T_k_matrix))

    series_of_ritz_values = []
    for ritz_values_ith_iter in ritz_values:
        series_of_ritz_values.append([])
        for j, ritz_value in enumerate(ritz_values_ith_iter):
            series_of_ritz_values[j].append(ritz_value)
    return series_of_ritz_values


def experiments_exercise_2_3():
    figsize = (15, 8)
    c_values = [0.01, 0.1, 1, 10, 100, 1000]
    grid_sizes = [10, 100, 1000]
    colors = cm.viridis(np.linspace(0, 1, len(c_values)*len(grid_sizes)))
    spectral_rads = []
    plt.figure(figsize=figsize)
    for i, c_ in enumerate(c_values):
        for j, size in enumerate(grid_sizes):
            h = 1/size
            A = create_discretized_helmholtz_matrix(size, c_)/h**2
            eigvals, spectral_rad = compute_eigenvals_gauss_seidel_error_propagation_matrix(
                A)
            spectral_rads.append((c_, size, spectral_rad))
            # Plot the numerical and analytical solutions
            plt.scatter(np.real(eigvals), np.imag(eigvals),
                        label=f'h = {1/size:.1E}, c = {c_:.1E})',
                        color=colors[i*len(grid_sizes)+j],
                        s=1)

    plt.xlabel('Real part')
    plt.ylabel('Imag part')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig("figures/plot_ex_2_3_eigenvalues_01.pdf")
    plt.savefig("figures/plot_ex_2_3_eigenvalues_01.svg")

    plt.figure(figsize=figsize)
    c_values = [1]
    # c_values = [0.01, 0.1, 10]
    grid_sizes = [10, 100, 1000]
    colors = cm.rainbow(np.linspace(0, 1, len(c_values)*len(grid_sizes)))
    for i, c_ in enumerate(c_values):
        for j, size in enumerate(grid_sizes):
            h = 1/size
            A = create_discretized_helmholtz_matrix(size, c_)/h**2
            eigvals, spectral_rad = compute_eigenvals_gauss_seidel_error_propagation_matrix(
                A)
            spectral_rads.append((c_, size, spectral_rad))
            # Plot the numerical and analytical solutions
            plt.scatter(np.real(eigvals), np.imag(eigvals),
                        label=f'h = {1/size:.1E}, c = {c_:.1E})',
                        color=colors[i*len(grid_sizes)+j],
                        s=1)

    plt.xlabel('Real part')
    plt.ylabel('Imag part')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig("figures/plot_ex_2_3_eigenvalues_02.pdf")
    plt.savefig("figures/plot_ex_2_3_eigenvalues_02.svg")

    plt.figure(figsize=figsize)
    c_values = [0.1, 1, 10, 100, 1000]
    # c_values = [0.01, 0.1, 10]
    grid_sizes = [1000]
    colors = cm.rainbow(np.linspace(0, 1, len(c_values)*len(grid_sizes)))
    for i, c_ in enumerate(c_values):
        for j, size in enumerate(grid_sizes):
            h = 1/size
            A = create_discretized_helmholtz_matrix(size, c_)/h**2
            eigvals, spectral_rad = compute_eigenvals_gauss_seidel_error_propagation_matrix(
                A)
            spectral_rads.append((c_, size, spectral_rad))
            # Plot the numerical and analytical solutions
            plt.scatter(np.real(eigvals), np.imag(eigvals),
                        label=f'h = {1/size:.1E}, c = {c_:.1E})',
                        color=colors[i*len(grid_sizes)+j],
                        s=1)

    plt.xlabel('Real part')
    plt.ylabel('Imag part')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig("figures/plot_ex_2_3_eigenvalues_03.pdf")
    plt.savefig("figures/plot_ex_2_3_eigenvalues_03.svg")

    with open("results_ex_2_3.txt", 'w') as f:
        for (c_, size, spectral_rad) in spectral_rads:
            line = f"(c,h)=({c_:1E}, {1/size:1E}): rho(B_GS)={spectral_rad})\n"
            print(line)
            f.write(line)


def experiments_exercise_4():
    c_values = [0.01, 0.1, 1, 10]
    grid_sizes = [10, 100, 1000]

    with open("results_ex_4.txt", 'w') as f:
        for c_ in c_values:
            for grid_size in grid_sizes:
                rel_errors = []
                h = 1/grid_size
                x = np.linspace(0, 1, grid_size+1)
                x = x[1:-1]
                h = 1/grid_size
                A = create_discretized_helmholtz_matrix(grid_size, c_)/h**2
                _, spectral_radius = compute_eigenvals_gauss_seidel_error_propagation_matrix(
                    A)
                rhs = f_rhs(c_, x, h)
                u_sol, rel_errors, convergence_flag = helmholtz_solvers.gauss_seidel_solver(
                    A, rhs, tol=1e-6)
                if convergence_flag:
                    lines = [f"(c, h) = ({c_}, {h})\n",
                             f"rate of convergence: {rel_errors[len(rel_errors)-5]:.3E}\n",
                             f"spectral rad.: {spectral_radius:.3E}\n",
                             f"min rate: {min(rel_errors):.3E}\n",
                             f"avg. rate: {np.average(np.array(rel_errors)[2:-2]):.3E}\n"]
                    print(*lines)
                    f.writelines(lines)
                    f.write('\n')
                else:
                    print(f"Convergence failed for (c, h) = ({c_}, h={h})\n")
                    f.write(f"Convergence failed for (c, h) = ({c_}, h={h})\n")


def experiments_exercise_5():
    # compute and tabulate condition number of symmetric gauss-seidel
    # preconditioned matrix inv(M_SGS)*A
    c_values = [0.01, 0.1, 1, 10]
    grid_sizes = [10, 100, 1000]

    with open("results_ex_5.txt", 'w') as f:
        for c in c_values:
            for grid_size in grid_sizes:
                h = 1/grid_size
                A = create_discretized_helmholtz_matrix(
                    size=grid_size, c=c)/h**2
                condition_number_A, = compute_condition_number(
                    A)
                M_ssor_inv = helmholtz_solvers.compute_symmetric_ssor_preconditioner(
                    A, 1.0)
                prec_operator = M_ssor_inv * A
                condition_number_prec_operator = np.linalg.cond(
                    prec_operator)
                line = f"(c, h) = ({c}, {h}) K_2(A): {condition_number_A}, K_2(M_SGS_i A): {condition_number_prec_operator}\n"
                f.write(line)


def experiments_exercise_6():
    c_values = [0.01, 0.1, 1, 10, 100, 1000]
    # c_values = [0.1]
    grid_sizes = [1000]

    fig = plt.figure(figsize=(16, 8))

    for c in c_values:
        for grid_size in grid_sizes:
            residuals_uncond = []
            residuals = []
            ritz_values = []
            h = 1/grid_size
            x = np.linspace(0, 1, grid_size+1)
            x = x[1:-1]
            A_h = create_discretized_helmholtz_matrix(
                size=grid_size, c=c) / h**2
            rhs = f_rhs(c, x, h)

            M_sgs_inv = helmholtz_solvers.compute_symmetric_ssor_preconditioner(
                A_h, omega=1.0)

            # _, _ = helmholtz_solvers.preconditioned_conjugate_gradient_with_ritz(
            #     A, rhs, residuals=residuals_uncond)
            u_sol, convergence_flag = helmholtz_solvers.preconditioned_conjugate_gradient_with_ritz(
                A_h, rhs=rhs, M_inv=M_sgs_inv, tol=1e-10, residuals=residuals)

            # Check solution
            u_exact = analytical_solution(x)
            assert np.isclose(calculate_rmse(
                u_sol, u_exact), 0, atol=1e-7)

            cond_A = compute_condition_number(A_h)
            cond_prec = np.linalg.cond(M_sgs_inv @ A_h)
            print(cond_A, cond_prec)

            # plt.plot(
            #     [np.linalg.norm(residual, ord=2)
            #      for residual in residuals_uncond],
            #     label=f"CG: $(c,h)=({c}, {h}), \kappa_2={cond_prec:.2E}$")
            plt.plot(
                [np.linalg.norm(residual, ord=2) for residual in residuals],
                label=f"PCG: $(c,h)=({c}, {h}), \kappa_2={cond_prec:.2E}$")

    plt.xlabel('# iterations')
    plt.ylabel('2-norm of residual')
    plt.legend()
    plt.semilogy()
    plt.grid(True)
    plt.show()
    fig.savefig("figures/plot_ex_6_convergence.pdf")
    fig.savefig("figures/plot_ex_6_convergence.svg")


def experiments_exercise_7():
    c = 1000
    grid_size = 1000

    residuals = []
    h = 1/grid_size
    x = np.linspace(0, 1, grid_size)
    x = x[1:-1]
    A_h = create_discretized_helmholtz_matrix(size=grid_size, c=c)/h**2
    rhs = f_rhs(c, x, h)

    M_sgs_inv = helmholtz_solvers.compute_symmetric_ssor_preconditioner(
        A_h, omega=1.0)

    prec_operator = np.matmul(M_sgs_inv, A_h)

    u_sol, convergence_flag = helmholtz_solvers.preconditioned_conjugate_gradient_with_ritz(
        A_h, rhs=rhs, tol=1e-10, M_inv=M_sgs_inv, residuals=residuals)
    assert convergence_flag  # "Problem did not converge"
    print("CG done")

    series_of_ritz_values = compute_series_of_ritz_values(
        M_sgs_inv @ A_h, residuals)
    prec_operator_eigenvalues = np.linalg.eigvals(prec_operator)
    print("Compute Ritz values done")

    # Check solution
    u_exact = analytical_solution(x)
    assert np.isclose(calculate_rmse(
        u_sol, u_exact), 0, atol=1e-7)

    fig = plt.figure(figsize=(16, 8))
    plt.scatter(range(len(
        series_of_ritz_values[0])),
        np.abs(prec_operator_eigenvalues[:len(
            series_of_ritz_values[0])]),
        label="$\mathrm{eig}(M_{\mathrm{SGS}}^{-1} A)$",
        s=1)
    plt.scatter([len(
        series_of_ritz_values[0])], prec_operator_eigenvalues[-1], label="last $\mathrm{eig}(M_{\mathrm{SGS}}^{-1} A)$",
        s=5)

    for i, series_of_ritz_value in enumerate(series_of_ritz_values):
        plt.plot(np.abs(series_of_ritz_value))

    plt.xlabel('# iterations')
    plt.ylabel('real part')
    # plt.semilogy()
    plt.legend()
    plt.show()
    fig.savefig("figures/plot_ex_7_ritz_values.pdf")
    fig.savefig("figures/plot_ex_7_ritz_values.svg")


def experiments_exercise_8_9():
    c_values = [0.1, 1, 10, 100, 1000]
    grid_sizes = [100, 1000]
    colors = cm.viridis(np.linspace(0, 1, len(c_values)*len(grid_sizes)))

    figsize = (15, 8)
    fig, axs = plt.subplots(figsize=figsize, ncols=1, nrows=2)
    spectral_rads = []
    for i, c_ in enumerate(c_values):
        for j, grid_size in enumerate(grid_sizes):
            h = 1/grid_size
            x = np.linspace(0, 1, grid_size)
            x = x[1:-1]
            A_h = create_discretized_helmholtz_matrix(grid_size, c_) / h**2
            B_cgc = create_coarse_grid_correction_error_propagation_matrix(A_h)
            eigvals_B_cgc = np.linalg.eigvals(B_cgc)
            spectral_radius = np.max(np.abs(eigvals_B_cgc))
            spectral_rads.append((c_, grid_size, spectral_radius))
            # Plot the numerical and analytical solutions
            eigvals_B_cgc_0 = eigvals_B_cgc[np.real(eigvals_B_cgc) < 0.01]
            axs[0].scatter(np.real(eigvals_B_cgc_0), np.imag(eigvals_B_cgc_0),
                           label=f'h = {1/grid_size:.1E}, c = {c_:.1E})',
                           color=colors[i*len(grid_sizes)+j],
                           s=2)
            axs[0].legend()
            axs[0].set_ylabel('Imag part')
            axs[0].set_xlabel('Real part')

            eigvals_B_cgc_1 = eigvals_B_cgc[np.real(eigvals_B_cgc) > 0.01]
            axs[1].scatter(np.real(eigvals_B_cgc_1), np.imag(eigvals_B_cgc_1),
                           label=f'h = {1/grid_size:.1E}, c = {c_:.1E})',
                           color=colors[i*len(grid_sizes)+j],
                           s=2)
            axs[1].set_xlabel('Real part')

    fig.show()
    fig.savefig("figures/plot_ex_8_B_CGC_eigenvalues.pdf")
    fig.savefig("figures/plot_ex_8_B_CGC_eigenvalues.svg")

    with open("results_ex_8_9.txt", 'w') as f:
        for (c_, grid_size, spectral_rad) in spectral_rads:
            line = f"(c,h)=({c_:1E}, {1/grid_size:1E}): rho(B_GS)={spectral_rad})\n"
            print(line)
            f.write(line)


def experiments_exercise_10():
    c_values = [0.1, 1, 10, 100, 1000]

    grid_sizes = [100, 1000]

    with open("results_ex_10.txt", 'w') as f:
        for i, c_ in enumerate(c_values):
            for j, grid_size in enumerate(grid_sizes):
                residuals = []
                h = 1/grid_size
                x = np.linspace(0, 1, grid_size+1)
                x = x[1:-1]
                A_h = create_discretized_helmholtz_matrix(
                    grid_size, c_) / h**2
                rhs = f_rhs(c_, x, h)

                u_sol_cgc, convergence_flag_cgc = helmholtz_solvers.coarse_grid_correction(
                    A_h, rhs_h=rhs, max_iterations=100, tol=1e-10, internal_solver='direct', residuals=residuals)

                # Check solution
                u_exact = analytical_solution(x)
                try:
                    assert np.isclose(calculate_rmse(
                        u_sol_cgc, u_exact), 0, atol=1e-6)
                except:
                    print(f"rmse: {calculate_rmse(u_sol_cgc, u_exact):.2E}")

                rate_of_convergence = compute_rate_of_convergence(residuals)
                B_cgc = create_coarse_grid_correction_error_propagation_matrix(
                    A_h)
                eigvals_B_cgc = np.linalg.eigvals(B_cgc)
                spectral_radius = np.max(np.abs(eigvals_B_cgc))

                if convergence_flag_cgc:
                    lines = [f"(c, h) = ({c_}, {h})\n",
                             f"rate of convergence: {rate_of_convergence:.3E}\n",
                             f"spectral rad.: {spectral_radius:.3E}\n"]
                    f.writelines(lines)
                    f.write('\n')


def experiments_exercise_11():
    c_values = [0.1, 1, 10, 100, 1000]

    grid_sizes = [10, 100, 1000]
    # colors = cm.viridis(np.linspace(0, 1, len(c_values)*len(grid_sizes)))

    fig = plt.figure()
    for i, c_ in enumerate(c_values):
        for j, grid_size in enumerate(grid_sizes):
            residuals = []
            h = 1/grid_size
            x = np.linspace(0, 1, grid_size+1)
            x = x[1:-1]
            A_h = create_discretized_helmholtz_matrix(
                grid_size, c_) / h**2
            rhs = f_rhs(c_, x, h)

            M_cgc_inv = create_coarse_grid_correction_operator(A_h)
            projection = np.eye(A_h.shape[0]) - A_h @ M_cgc_inv

            P_A_h = projection @ A_h
            P_rhs_h = projection @ rhs
            M_sgs_inv = helmholtz_solvers.compute_symmetric_ssor_preconditioner(
                P_A_h, omega=1.0)

            u_sol_cg, convergence_flag_cg = helmholtz_solvers.preconditioned_conjugate_gradient_with_ritz(
                A=P_A_h, rhs=P_rhs_h, max_iterations=1000, tol=1e-12, residuals=residuals)

            # Check solution
            u_exact = analytical_solution(x)
            print('rmse: ', calculate_rmse(u_sol_cg, u_exact))

            cond_A = compute_effective_condition_number(P_A_h)
            cond_prec = np.linalg.cond(P_A_h)
            print(cond_A, cond_prec)

            plt.plot(
                [np.linalg.norm(residual, ord=2) for residual in residuals],
                label=f"$(c,h)=({c_}, {h}), \kappa_{{eff,2}}={cond_A:.2E}$")
        plot_solutions(x, u_exact, *[("Projected CG", u_sol_cg)])

    plt.xlabel('# iterations')
    plt.ylabel('2-norm of residual')
    plt.legend()
    plt.semilogy()
    plt.grid(True)
    plt.show()
    fig.savefig("figures/plot_ex_11_convergence.pdf")
    fig.savefig("figures/plot_ex_11_convergence.svg")


def experiments_exercise_12():
    figsize = (15, 8)
    c_values = [0.01, 0.1, 1, 10, 100]
    grid_sizes = [100, 1000]
    colors = cm.viridis(np.linspace(0, 1, len(c_values)*len(grid_sizes)))
    spectral_rads = []
    fig, axs = plt.subplots(figsize=figsize, ncols=3, nrows=1)
    for i, c_ in enumerate(c_values):
        for j, grid_size in enumerate(grid_sizes):
            h = 1/grid_size
            x = np.linspace(0, 1, grid_size+1)
            x = x[1:-1]
            A_h = create_discretized_helmholtz_matrix(
                grid_size, c_) / h**2
            rhs = f_rhs(c_, x, h)

            B_gs = create_gauss_seidel_error_propagation_matrix(A_h)
            B_cgc = create_coarse_grid_correction_error_propagation_matrix(A_h)
            B_gs_backwards = create_gauss_seidel_error_propagation_matrix(
                A_h, backwards=True)
            B_tgm = B_gs_backwards @ B_cgc @ B_gs

            eigvals_B_tgm = np.linalg.eigvals(B_tgm)
            spectral_radius = np.max(np.abs(eigvals_B_tgm))
            spectral_rads.append((c_, grid_size, spectral_radius))

            # Plot the numerical and analytical solutions
            eigvals_B_tgm_0 = eigvals_B_tgm[np.real(eigvals_B_tgm) < 1e-12]
            axs[0].scatter(np.real(eigvals_B_tgm_0), np.imag(eigvals_B_tgm_0),
                           label=f'h = {1/grid_size:.1E}, c = {c_:.1E})',
                           color=colors[i*len(grid_sizes)+j],
                           s=3)
            axs[0].loglog()
            axs[0].set_ylabel('Imag part')
            axs[0].set_xlabel('Real part')

            eigvals_B_tgm_1 = eigvals_B_tgm[np.logical_and(np.real(eigvals_B_tgm) > 1e-12,
                                            np.real(eigvals_B_tgm) < 0.1)]
            axs[1].scatter(np.real(eigvals_B_tgm_1), np.imag(eigvals_B_tgm_1),
                           label=f'h = {1/grid_size:.1E}, c = {c_:.1E})',
                           color=colors[i*len(grid_sizes)+j],
                           s=3)
            axs[1].legend()
            axs[1].set_xlabel('Real part')

            eigvals_B_tgm_2 = eigvals_B_tgm[np.real(eigvals_B_tgm) > 0.1]
            axs[2].scatter(np.real(eigvals_B_tgm_2), np.imag(eigvals_B_tgm_2),
                           label=f'h = {1/grid_size:.1E}, c = {c_:.1E})',
                           color=colors[i*len(grid_sizes)+j],
                           s=3)
            axs[2].set_xlabel('Real part')

    fig.show()
    fig.savefig("figures/plot_ex_12_eigenvalues.pdf")
    fig.savefig("figures/plot_ex_12_eigenvalues.svg")

    with open("results_ex_12.txt", 'w') as f:
        for (c_, size, spectral_rad) in spectral_rads:
            line = f"(c,h)=({c_:1E}, {1/size:1E}): rho(B_GS)={spectral_rad})\n"
            print(line)
            f.write(line)


if __name__ == "__main__":
    # Example usage:
    # A = create_discretized_helmholtz_matrix(5, 0.1)

    # A = np.diag(np.linspace(1, 5, 10))  # Symmetric positive definite matrix
    # b = np.ones(10)  # Right-hand side vector

    # solution, ritz_values = conjugate_gradient_with_ritz(A, b)

    # Print the solution and Ritz values
    # print("Solution:", solution)
    # print("Ritz Values:", ritz_values)

    # Example usage:
    # h = 8
    # coarsening_matrix = create_coarsening_matrix(h)
    # print("Coarsening Matrix:")
    # print(4*coarsening_matrix)
    # prolongation_matrix = create_prolongation_matrix(h)
    # print("Prolongation Matrix:")
    # print(2*prolongation_matrix)

    # Exercise 02, 03
    # experiments_exercise_2_3()
    # Exercise 04
    # experiments_exercise_4()
    # Exercise 05
    # experiments_exercise_5()
    # Exercise 06
    # experiments_exercise_6()
    # Exercise 07
    # experiments_exercise_7()

    # Exercise 08, 09
    # experiments_exercise_8_9()
    # Exercise 10
    # experiments_exercise_10()
    # Exercise 11
    # experiments_exercise_11()
    # Exercise 12
    experiments_exercise_12()
