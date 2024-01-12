
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import helmholtz_solvers
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


def compute_effective_condition_number(M, u_sol, u_0):
    if is_symmetric(M):
        # Reader effective condition number does not work
        # lhs = u_sol - u_0
        # eig_vals, eig_vecs = np.linalg.eig(M)
        # gammas = np.zeros(eig_vals.shape)
        # for i, eig_vec in enumerate(eig_vecs):
        #     gammas[i] = np.dot(eig_vec, lhs)
        # eig_vals_nonzero = eig_vals[np.where(gammas > 1e-14)]

        # alpha = np.min(eig_vals_nonzero)
        # beta = np.max(eig_vals_nonzero)
        # condition_number_A = beta/alpha
        eig_vals = np.linalg.eigvals(M)
        min_eigval = np.min(np.abs(eig_vals[eig_vals > 1e-8]))
        max_eigval = np.max(np.abs(eig_vals[eig_vals > 1e-8]))
        condition_number_M = max_eigval/min_eigval
    else:
        raise TypeError(
            "Input A is not symmetric and thus inv(M_ssor)*A is not symmetric")
    return condition_number_M


def compute_eigenvals_gauss_seidel_error_propagation_matrix(A_h):
    B_gs = helmholtz_solvers.create_gauss_seidel_error_propagation_matrix(A_h)
    eigvals_B_gs = np.linalg.eigvals(B_gs)
    spectral_radius = np.max(np.abs(eigvals_B_gs))

    return eigvals_B_gs, spectral_radius


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

    ritz_value_min = []
    ritz_value_max = []
    for T_k_matrix in T_k_matrices:
        eig_vals = np.linalg.eigvals(T_k_matrix)
        ritz_values.append(eig_vals)
        ritz_value_min.append(min(eig_vals))
        ritz_value_max.append(max(eig_vals))

    series_of_ritz_values = []
    for ritz_values_ith_iter in ritz_values:
        series_of_ritz_values.append([])
        for j, ritz_value in enumerate(ritz_values_ith_iter):
            series_of_ritz_values[j].append(ritz_value)
    return series_of_ritz_values, ritz_value_max, ritz_value_min


def experiments_exercise_1():
    grid_sizes = [10, 100, 1000, 10000]  # Example grid sizes
    plt.figure(figsize=(15, 8))
    c = 1

    solutions = []
    labels = []
    for grid_size in grid_sizes:
        h = 1/grid_size
        x = np.linspace(0, 1, grid_size+1)
        x = x[1:-1]
        A_h = create_discretized_helmholtz_matrix(size=grid_size, c=c)/h**2
        rhs = f_rhs(c, x, h)

        # Solve the Helmholtz equation
        u = np.linalg.solve(A_h, rhs)
        # Compute the analytical solution
        u_exact = analytical_solution(x)

        # rmse = calculate_rmse(u, u_exact)
        solutions.append(np.abs(u - u_exact))
        # Plot the numerical and analytical solutions
        labels.append(f'h = {h}')

    plt.violinplot(solutions, showmeans=False,
                   showmedians=True)
    ax = plt.gca()
    ax.set_xticks([y + 1 for y in range(len(solutions))],
                  labels=labels)
    # Only plot the last exact solution
    plt.xlabel('$x$')
    plt.ylabel('$|u(x) - u^h(x)|$')
    plt.legend()
    plt.semilogy()
    plt.savefig("figures/plot_ex_1_pointwise_error_01.pdf")
    plt.savefig("figures/plot_ex_1_pointwise_error_01.svg")


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
    plt.legend(prop={'size': 6})
    plt.savefig("figures/plot_ex_2_3_eigenvalues_01.pdf")
    plt.savefig("figures/plot_ex_2_3_eigenvalues_01.svg")

    plt.figure(figsize=figsize)
    c_values = [1]
    # c_values = [0.01, 0.1, 10]
    grid_sizes = [10000, 1000, 100, 10]
    colors = cm.viridis(np.linspace(0, 1, len(c_values)*len(grid_sizes)))
    for i, c_ in enumerate(c_values):
        for j, size in enumerate(grid_sizes):
            h = 1/size
            A = create_discretized_helmholtz_matrix(size, c_)/h**2
            eigvals, spectral_rad = compute_eigenvals_gauss_seidel_error_propagation_matrix(
                A)
            # Plot the numerical and analytical solutions
            plt.scatter(np.real(eigvals), np.imag(eigvals),
                        label=f'h = {1/size:.1E}, c = {c_:.1E})',
                        color=colors[i*len(grid_sizes)+j],
                        s=10)

    plt.xlabel('Real part')
    plt.ylabel('Imag part')
    plt.legend()
    plt.savefig("figures/plot_ex_2_3_eigenvalues_02.pdf")
    plt.savefig("figures/plot_ex_2_3_eigenvalues_02.svg")

    plt.figure(figsize=figsize)
    c_values = [0.1, 1, 10, 100, 1000]
    # c_values = [0.01, 0.1, 10]
    grid_sizes = [1000]
    colors = cm.viridis(np.linspace(0, 1, len(c_values)*len(grid_sizes)))
    for i, c_ in enumerate(c_values):
        for j, size in enumerate(grid_sizes):
            h = 1/size
            A = create_discretized_helmholtz_matrix(size, c_)/h**2
            eigvals, spectral_rad = compute_eigenvals_gauss_seidel_error_propagation_matrix(
                A)
            # Plot the numerical and analytical solutions
            plt.scatter(np.real(eigvals), np.imag(eigvals),
                        label=f'h = {1/size:.1E}, c = {c_:.1E})',
                        color=colors[i*len(grid_sizes)+j],
                        s=10)

    plt.xlabel('Real part')
    plt.ylabel('Imag part')
    plt.legend()
    plt.savefig("figures/plot_ex_2_3_eigenvalues_03.pdf")
    plt.savefig("figures/plot_ex_2_3_eigenvalues_03.svg")

    with open("results_ex_2_3.txt", 'w') as f:
        for (c_, size, spectral_rad) in spectral_rads:
            line = f"(c,h)=({c_:1E}, {1/size:1E}): rho(B_GS)={spectral_rad})\n"
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
                A_h = create_discretized_helmholtz_matrix(grid_size, c_)/h**2
                _, spectral_radius = compute_eigenvals_gauss_seidel_error_propagation_matrix(
                    A_h)
                rhs = f_rhs(c_, x, h)
                u_sol, rel_errors, convergence_flag = helmholtz_solvers.gauss_seidel_solver(
                    A_h, rhs, tol=1e-6, max_iterations=10000)
                if convergence_flag:
                    lines = [f"(c, h) = ({c_}, {h})\n",
                             f"rate of convergence: {rel_errors[len(rel_errors)-5]:.3E}\n",
                             f"spectral rad.: {spectral_radius:.3E}\n",
                             f"min rate: {min(rel_errors):.3E}\n",
                             f"avg. rate: {np.average(np.array(rel_errors)[2:-2]):.3E}\n"]
                    f.writelines(lines)
                    f.write('\n')
                else:
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
                condition_number_A = compute_condition_number(
                    A)
                M_ssor_inv = helmholtz_solvers.compute_symmetric_ssor_M_inv(
                    A, 1.0)
                prec_operator = M_ssor_inv @ A
                condition_number_prec_operator = np.linalg.cond(
                    prec_operator)
                line = f"(c, h) = ({c}, {h}) K_2(A): {condition_number_A}, K_2(M_SGS_i A): {condition_number_prec_operator}\n"
                f.write(line)


def experiments_exercise_6(prec_type, var):
    # prec_type: type of preconditioning in the CG method. Options:
    #    - explicit
    #    - residuals
    # var : variable to test multiple values of ( for plotting purposes ). Options:
    #    - c
    #    - h
    if var == 'c':
        c_values = [0.01, 0.1, 1, 10, 100, 1000]
        grid_sizes = [1000]
    elif var == 'h':
        grid_sizes = [10, 100, 1000]
        c_values = [10]
    else:
        raise ValueError(f"var={var} not supported")

    fig = plt.figure(figsize=(16, 8))

    for c in c_values:
        for grid_size in grid_sizes:
            residuals = []
            h = 1/grid_size
            x = np.linspace(0, 1, grid_size+1)
            x = x[1:-1]
            A_h = create_discretized_helmholtz_matrix(
                size=grid_size, c=c) / h**2
            rhs = f_rhs(c, x, h)

            M_sgs_inv = helmholtz_solvers.compute_symmetric_ssor_M_inv(
                A_h, omega=1.0)

            # _, _ = helmholtz_solvers.preconditioned_conjugate_gradient_with_ritz(
            #     A, rhs, residuals=residuals_uncond)
            if prec_type == 'explicit':
                u_sol, convergence_flag = helmholtz_solvers.preconditioned_conjugate_gradient_type2(
                    A_h, rhs=rhs, M_inv=M_sgs_inv, tol=1e-10, residuals=residuals)
            elif prec_type == 'residuals':
                u_sol, convergence_flag = helmholtz_solvers.preconditioned_conjugate_gradient(
                    A_h, rhs=rhs, tol=1e-10, M_inv=M_sgs_inv, residuals=residuals)
            else:
                raise ValueError(f"prec_type={prec_type} not supported")

            # Check solution
            u_exact = analytical_solution(x)
            # assert np.isclose(calculate_rmse(
            #    u_sol, u_exact), 0, atol=1e-7)

            cond_A = compute_condition_number(A_h)
            cond_prec = np.linalg.cond(M_sgs_inv @ A_h)

            plt.plot(
                [np.linalg.norm(residual, ord=2) for residual in residuals], label=f"PCG: $(c,h)=({c}, {h}), \kappa_2={cond_prec:.2E}$")
    plt.xlabel('# iterations')
    plt.ylabel('2-norm of residual')
    plt.legend(fontsize=14.5, loc='lower left')
    plt.semilogy()
    plt.grid(True)
    fig.savefig(f"figures/plot_ex_6_convergence_prec_{prec_type}_{var}.pdf")
    fig.savefig(f"figures/plot_ex_6_convergence_prec_{prec_type}_{var}.svg")


def experiments_exercise_7():
    c = 10
    grid_size = 10

    residuals = []
    h = 1/grid_size
    x = np.linspace(0, 1, grid_size+1)
    x = x[1:-1]
    A_h = create_discretized_helmholtz_matrix(size=grid_size, c=c)/h**2
    rhs = f_rhs(c, x, h)

    # split preconditioning - doesn't work
    # M1_sgs_inv, M2_sgs_inv = helmholtz_solvers.compute_symmetric_ssor_M_inv_split(
    #    A_h, omega=1.0)
    # u_sol, convergence_flag = helmholtz_solvers.preconditioned_conjugate_gradient_type3(
    #    A_h, rhs=rhs, tol=1e-10, M_inv=[M1_sgs_inv, M2_sgs_inv], residuals=residuals) # if sys not prec, M_sgs_inv=None
    # prec_operator = M1_sgs_inv @ A_h @ M2_sgs_inv

    # residual preconditioning
    M_sgs_inv = helmholtz_solvers.compute_symmetric_ssor_M_inv(A_h, omega=1.0)
    prec_operator = np.matmul(M_sgs_inv, A_h)

    u_sol, convergence_flag = helmholtz_solvers.preconditioned_conjugate_gradient(
        A_h, rhs=rhs, tol=1e-10, M_inv=M_sgs_inv, residuals=residuals)  # if sys not prec, M_sgs_inv=None
    assert convergence_flag  # "Problem did not converge"

    series_of_ritz_values, ritz_values_max, ritz_values_min = compute_series_of_ritz_values(
        prec_operator, residuals)  # if sys not preconditioned just use A_h
    prec_operator_eigenvalues = np.linalg.eigvals(
        prec_operator)  # if sys not preconditioned, just use A_h
    max_eigval = np.max(prec_operator_eigenvalues)
    min_eigval = np.min(prec_operator_eigenvalues)

    # plot the sequences of ritz values and the eigenvalues of the (preconditioned) system
    colors = cm.viridis(np.linspace(0, 1, len(series_of_ritz_values)))
    fig = plt.figure(figsize=(16, 8))
    plt.scatter(np.ones(len(prec_operator_eigenvalues)), sorted(
        prec_operator_eigenvalues), label="eigenvalues A", color='tab:orange')
    sorted_series_of_ritz_values = sorted(
        series_of_ritz_values, key=lambda x: x[-1])
    for i, (series_of_ritz_value, color) in enumerate(zip(sorted_series_of_ritz_values, colors)):
        plt.plot(np.linspace(0, 1, len(series_of_ritz_value)),
                 series_of_ritz_value, color=color)
    plt.ylabel('real part')
    plt.legend(fontsize=15)
    fig.savefig("figures/plot_ex_7c_ritz_values.svg")
    fig.savefig("figures/plot_ex_7c_ritz_values.pdf")

    # plot the max and the min eigenvalues of every T_k matrix and the eigenvalues of the (preconditioned) system
    fig = plt.figure(figsize=(16, 8))
    plt.scatter(len(ritz_values_max)-1, max_eigval,
                label="max eig A", marker='o', color='tab:orange')
    plt.scatter(len(ritz_values_max)-1, min_eigval,
                label="min eig A", marker='o', color='tab:orange')
    plt.plot(ritz_values_max, label=f"max Ritz value", marker='.',
             fillstyle='none', color='tab:cyan', markersize=5)
    plt.plot(ritz_values_min, label=f"min Ritz value", marker='.',
             fillstyle='none', color='tab:cyan', markersize=5)
    plt.xlabel('# iterations')
    plt.ylabel('real part')
    plt.legend(fontsize=15)
    fig.savefig("figures/plot_ex_7b_ritz_values.pdf")
    fig.savefig("figures/plot_ex_7b_ritz_values.svg")


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
            B_cgc = helmholtz_solvers.create_coarse_grid_correction_error_propagation_matrix(
                A_h)
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
            f.write(line)


def experiments_exercise_10():
    c_values = [0.1, 1, 10, 100, 1000]

    grid_sizes = [10, 100]

    figsize = (15, 8)
    fig = plt.figure(figsize=figsize)
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
                    A_h=A_h, rhs_h=rhs, max_iterations=10, tol=1e-10, residuals=residuals)

                # Check solution
                u_exact = analytical_solution(x)
                try:
                    assert np.isclose(calculate_rmse(
                        u_sol_cgc, u_exact), 0, atol=1e-6)
                except:
                    print("Assertion failed")

                rate_of_convergence = compute_rate_of_convergence(residuals)
                B_cgc = helmholtz_solvers.create_coarse_grid_correction_error_propagation_matrix(
                    A_h)
                eigvals_B_cgc = np.linalg.eigvals(B_cgc)
                spectral_radius = np.max(np.abs(eigvals_B_cgc))
                compute_condition_number(A_h)

                if convergence_flag_cgc:
                    lines = [f"(c, h) = ({c_}, {h})\n",
                             f"rate of convergence: {rate_of_convergence:.3E}\n",
                             f"spectral rad.: {spectral_radius:.3E}\n"]
                    f.writelines(lines)
                    f.write('\n')

            plt.plot(
                [np.linalg.norm(residual, ord=2) for residual in residuals],
                label=f"(c,h)=({c_}, {h}), "+"$\\rho(B_{CGC})=$"+f"{spectral_radius:.2E}")
    plt.xlabel('# iterations', fontsize=20)
    plt.ylabel('2-norm of residual', fontsize=20)
    plt.legend(fontsize=18)
    plt.semilogy()
    fig.savefig("figures/plot_ex_10_convergence.pdf")
    fig.savefig("figures/plot_ex_10_convergence.svg")


def experiments_exercise_11():
    c_values = [-50, -1, 0.1, 1, 10, 100, 1000]

    grid_sizes = [10, 100, 1000]
    # colors = cm.viridis(np.linspace(0, 1, len(c_values)*len(grid_sizes)))

    fig = plt.figure()
    for i, c_ in enumerate(c_values):
        for j, grid_size in enumerate(grid_sizes):
            residuals = []
            residuals1 = []
            h = 1/grid_size
            x = np.linspace(0, 1, grid_size+1)
            x = x[1:-1]
            A_h = create_discretized_helmholtz_matrix(
                grid_size, c_) / h**2
            rhs = f_rhs(c_, x, h)

            M_cgc_inv = helmholtz_solvers.create_coarse_grid_correction_M_inv(
                A_h)
            projection = np.eye(A_h.shape[0]) - A_h @ M_cgc_inv

            P_A_h = projection @ A_h
            P_rhs_h = projection @ rhs
            # M_sgs_inv = helmholtz_solvers.compute_symmetric_ssor_M_inv(
            #     P_A_h, omega=1.0)
            # A u = f -> u -> P A x = P f -> P A x = Pf = PA u
            u_sol, convergence_flag_cg = helmholtz_solvers.preconditioned_conjugate_gradient(
                A=A_h, rhs=rhs, max_iterations=1000, tol=1e-12, residuals=residuals1)
            u_sol_cg, convergence_flag_cg = helmholtz_solvers.preconditioned_conjugate_gradient(
                A=P_A_h, rhs=P_rhs_h, max_iterations=1000, tol=1e-12, residuals=residuals)

            # Check solution
            u_exact = analytical_solution(x)

            cond_P_A_h = compute_effective_condition_number(
                P_A_h, u_sol=u_sol, u_0=np.zeros(u_sol.shape))
            print("cond_prec:", np.linalg.cond(P_A_h))

            plt.plot(
                [np.linalg.norm(residual, ord=2) for residual in residuals],
                label=f"$(c,h)=({c_}, {h}), \kappa_{{eff,2}}={cond_P_A_h:.2E}$")
        # plot_solutions(x, u_exact, *[("Projected CG", u_sol_cg)])

    plt.xlabel('# iterations')
    plt.ylabel('2-norm of residual')
    plt.legend(prop={'size': 4})
    plt.semilogy()
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

            B_tgm = helmholtz_solvers.create_error_propagation_matrix_two_grid(
                A_h)

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
    # fig.savefig("figures/plot_ex_12_eigenvalues.pdf")
    # fig.savefig("figures/plot_ex_12_eigenvalues.svg")

    with open("results_ex_12.txt", 'w') as f:
        for (c_, size, spectral_rad) in spectral_rads:
            line = f"(c,h)=({c_:1E}, {1/size:1E}): rho(B_TGM)={spectral_rad})\n"
            f.write(line)


def experiments_exercise_12_4():
    c_values = [0.01, 0.1, 1, 10, 100, 1000]
    # c_values = [0.1]
    grid_sizes = [10, 100]

    fig = plt.figure(figsize=(16, 8))

    for c in c_values:
        for grid_size in grid_sizes:
            residuals = []
            h = 1/grid_size
            x = np.linspace(0, 1, grid_size+1)
            x = x[1:-1]
            A_h = create_discretized_helmholtz_matrix(
                size=grid_size, c=c) / h**2
            rhs = f_rhs(c, x, h)

            u_sol, convergence_flag = helmholtz_solvers.two_grid_method_matrix(
                A_h=A_h, rhs_h=rhs, tol=1e-10, residuals=residuals)

            # Check solution
            u_exact = analytical_solution(x)
            # assert np.isclose(calculate_rmse(
            #     u_sol, u_exact), 0, atol=1e-7)

            cond_A = compute_condition_number(A_h)
            cond_prec = np.linalg.cond(A_h)
            plt.plot(
                [np.linalg.norm(residual, ord=2) for residual in residuals],
                label=f"TGM: $(c,h)=({c}, {h}), \kappa_2={cond_prec:.2E}$")

    plt.xlabel('# iterations')
    plt.ylabel('2-norm of residual')
    plt.legend(prop={'size': 6})
    plt.semilogy()
    fig.savefig("figures/plot_ex_12_4_convergence.pdf")
    fig.savefig("figures/plot_ex_12_4_convergence.svg")


def experiments_exercise_12_4a():
    c_values = [0.01, 0.1, 1, 10, 100]
    grid_sizes = [10, 100, 1000]

    with open("results_ex_12_4a.txt", 'w') as f:
        for c_ in c_values:
            for grid_size in grid_sizes:
                residuals = []
                h = 1/grid_size
                x = np.linspace(0, 1, grid_size+1)
                x = x[1:-1]
                h = 1/grid_size
                spectral_rads = []
                A_h = create_discretized_helmholtz_matrix(grid_size, c_)/h**2
                B_tgm = helmholtz_solvers.create_error_propagation_matrix_two_grid(
                    A_h)

                eigvals_B_tgm = np.linalg.eigvals(B_tgm)
                spectral_radius = np.max(np.abs(eigvals_B_tgm))
                spectral_rads.append((c_, grid_size, spectral_radius))
                rhs = f_rhs(c_, x, h)
                M_inv_tgm = helmholtz_solvers.create_two_grid_method_M_inv(
                    A_h)
                u_sol, convergence_flag = helmholtz_solvers.iterative_solve(
                    A_h, rhs, M_inv_tgm, tol=1e-6, max_iterations=10000, residuals=residuals)
                rate_of_convergence = compute_rate_of_convergence(
                    residuals=residuals)
                if convergence_flag:
                    lines = [f"(c, h) = ({c_}, {h})\n",
                             f"rate of convergence: {rate_of_convergence:.3E}\n",
                             f"spectral rad.: {spectral_radius:.3E}\n"]
                    f.writelines(lines)
                    f.write('\n')
                else:
                    f.write(f"Convergence failed for (c, h) = ({c_}, h={h})\n")


def experiments_exercise_12_5():
    # compute and tabulate condition number of symmetric gauss-seidel
    # preconditioned matrix inv(M_SGS)*A
    c_values = [0.01, 0.1, 1, 10, 100]
    grid_sizes = [10, 100, 1000]

    with open("results_ex_12_5.txt", 'w') as f:
        for c in c_values:
            for grid_size in grid_sizes:
                h = 1/grid_size
                A_h = create_discretized_helmholtz_matrix(
                    size=grid_size, c=c)/h**2
                condition_number_A = compute_condition_number(
                    A_h)
                M_tgm_inv = helmholtz_solvers.create_two_grid_method_M_inv(
                    A_h)
                condition_number_prec_operator = np.linalg.cond(
                    M_tgm_inv @ A_h)
                line = f"(c, h) = ({c}, {h}) K_2(A): {condition_number_A}, K_2(M_SGS_i A): {condition_number_prec_operator}\n"
                f.write(line)


def experiments_exercise_12_6():
    c_values = [0.01, 0.1, 1, 10, 100, 1000]
    grid_sizes = [10, 100, 1000]

    fig = plt.figure(figsize=(16, 8))
    colors = cm.tab20b(np.linspace(0, 1, len(c_values)*len(grid_sizes)))
    for i, c in enumerate(c_values):
        for j, grid_size in enumerate(grid_sizes):
            residuals = []
            h = 1/grid_size
            x = np.linspace(0, 1, grid_size+1)
            x = x[1:-1]
            A_h = create_discretized_helmholtz_matrix(
                size=grid_size, c=c) / h**2
            rhs = f_rhs(c, x, h)

            M_tgm_inv = helmholtz_solvers.create_two_grid_method_M_inv(
                A_h)

            # _, _ = helmholtz_solvers.preconditioned_conjugate_gradient_with_ritz(
            #     A, rhs, residuals=residuals_uncond)
            u_sol, convergence_flag = helmholtz_solvers.preconditioned_conjugate_gradient(
                A_h, rhs=rhs, tol=1e-12, M_inv=M_tgm_inv, residuals=residuals)
            assert convergence_flag
            # Check solution
            u_exact = analytical_solution(x)
            # assert np.isclose(calculate_rmse(
            #    u_sol, u_exact), 0, atol=1e-7)

            cond_prec = np.linalg.cond(M_tgm_inv @ A_h)

            plt.plot(
                [np.linalg.norm(residual, ord=2) for residual in residuals], label=f"TGM PCG: $(c,h)=({c}, {h}), \kappa_2={cond_prec:.2E}$",
                color=colors[j*len(c_values)+i])
    plt.xlabel('# iterations')
    plt.ylabel('2-norm of residual')
    plt.legend(fontsize=8, loc='lower left')
    fig.savefig(f"figures/plot_ex_12_6_convergence_prec.pdf")
    fig.savefig(f"figures/plot_ex_12_6_convergence_prec.svg")


def experiments_exercise_12_7():
    c = 0.1
    grid_size = 10

    residuals = []
    h = 1/grid_size
    x = np.linspace(0, 1, grid_size+1)
    x = x[1:-1]
    A_h = create_discretized_helmholtz_matrix(size=grid_size, c=c)/h**2
    rhs = f_rhs(c, x, h)

    # residual preconditioning
    M_tgm_inv = helmholtz_solvers.create_two_grid_method_M_inv(A_h)
    prec_operator = np.matmul(M_tgm_inv, A_h)

    u_sol, convergence_flag = helmholtz_solvers.preconditioned_conjugate_gradient(
        A_h, rhs=rhs, tol=1e-10, M_inv=M_tgm_inv, residuals=residuals)  # if sys not prec, M_sgs_inv=None
    assert convergence_flag  # "Problem did not converge"

    series_of_ritz_values, ritz_values_max, ritz_values_min = compute_series_of_ritz_values(
        prec_operator, residuals)  # if sys not preconditioned just use A_h
    prec_operator_eigenvalues = np.linalg.eigvals(
        prec_operator)  # if sys not preconditioned, just use A_h
    max_eigval = np.max(prec_operator_eigenvalues)
    min_eigval = np.min(prec_operator_eigenvalues)

    # plot the sequences of ritz values and the eigenvalues of the (preconditioned) system
    colors = cm.viridis(np.linspace(0, 1, len(series_of_ritz_values)))
    fig = plt.figure(figsize=(16, 8))
    plt.scatter(np.ones(len(prec_operator_eigenvalues)), sorted(
        prec_operator_eigenvalues), label="eigenvalues A", color='tab:orange')
    sorted_series_of_ritz_values = sorted(
        series_of_ritz_values, key=lambda x: x[-1])
    for i, (series_of_ritz_value, color) in enumerate(zip(sorted_series_of_ritz_values, colors)):
        plt.plot(np.linspace(0, 1, len(series_of_ritz_value)),
                 series_of_ritz_value, color=color)
    plt.ylabel('real part')
    plt.legend(fontsize=8)
    fig.savefig("figures/plot_ex_12_7c_ritz_values.svg")
    fig.savefig("figures/plot_ex_12_7c_ritz_values.pdf")

    # plot the max and the min eigenvalues of every T_k matrix and the eigenvalues of the (preconditioned) system
    fig = plt.figure(figsize=(16, 8))
    plt.scatter(len(ritz_values_max)-1, max_eigval,
                label="max eig A", marker='o', color='tab:orange')
    plt.scatter(len(ritz_values_max)-1, min_eigval,
                label="min eig A", marker='o', color='tab:orange')
    plt.plot(ritz_values_max, label=f"max Ritz value", marker='.',
             fillstyle='none', color='tab:cyan', markersize=5)
    plt.plot(ritz_values_min, label=f"min Ritz value", marker='.',
             fillstyle='none', color='tab:cyan', markersize=5)
    plt.xlabel('# iterations')
    plt.ylabel('real part')
    plt.legend(fontsize=8)
    fig.savefig("figures/plot_ex_12_7b_ritz_values.pdf")
    fig.savefig("figures/plot_ex_12_7b_ritz_values.svg")


if __name__ == "__main__":
    # Exercise 01
    experiments_exercise_1()
    # Exercise 02, 03
    experiments_exercise_2_3()
    # Exercise 04
    experiments_exercise_4()
    # Exercise 05
    experiments_exercise_5()
    # Exercise 06
    experiments_exercise_6(var='h', prec_type='explicit')
    # Exercise 07
    experiments_exercise_7()
    # Exercise 08, 09
    experiments_exercise_8_9()
    # Exercise 10
    experiments_exercise_10()
    # Exercise 11
    experiments_exercise_11()
    # Exercise 12
    experiments_exercise_12()
    experiments_exercise_12_4a()
    experiments_exercise_12_5()
    experiments_exercise_12_6()
    experiments_exercise_12_7()
