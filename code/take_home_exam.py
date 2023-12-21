
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import helmholtz_solvers


def create_discretized_helmholtz_matrix(size=10, c=0.1):
    """Setup and return the system matrix A^h with the form tridiag[-1, 2+h^2c, -1]
    where h = 1/size and the size of the matrix is (size - 1) x (size - 1)
    """
    h = 1/size
    A = (2 + h**2 * c) * np.identity(size-1)
    for i in range(1, size-1):
        A[i][i-1] = -1
        A[i-1][i] = -1
    return A


def create_coarsening_matrix(h):
    """
    Create a coarsening matrix for a given finer grid size h.

    Parameters:
    - h: Size of the finer grid.

    Returns:
    - A numpy matrix representing the coarsening matrix.
    """
    if h % 2 != 0:
        raise ValueError("The size of the finer grid 'h' should be even.")
    size_fine = h+1
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


def create_prolongation_matrix(h):
    """
    Uses `create_coarsening_matrix` and returns the transpose of the matrix
    multiplied by 2
    """
    return 2 * np.transpose(create_coarsening_matrix(h))


def analytical_solution(x):
    """Analytical solution for the Helmholtz equation."""
    return np.exp(x)*(1-x)


def f_rhs(c, x, h):
    """Return the right-hand side of the Helmholtz-problem assuming the analytical solution given
    in `analytical_solution` given a specific value of c and the given boundary conditions given
    in `boundary_conditions`.

    Parameters:
    - c: Constant of the Helmholtz problem
    - x: Grid vector for the Helmhotz problem with elimination and Von-Neumann boundary conditions
    - h: Step size
    """
    rhs = np.exp(x)*(c + 1 + x - c*x)
    alpha, beta = boundary_conditions()
    rhs[0] += alpha/h**2
    rhs[-1] += beta/h**2
    return rhs


def boundary_conditions():
    """Return the boundary conditions alpha, beta = 1, 0."""
    alpha = 1
    beta = 0
    return alpha, beta


def is_symmetric(A):
    return np.allclose(A, np.transpose(A), rtol=1e-5, atol=1e-5)


def compute_symmetric_ssor_preconditioner(A, omega):
    """Return the preconditioning matrix M_SGS(w) for the symmetric successive over relaxation (SSOR)
    with parameter $\omega$. The matrix would need to be inverted before being applied to the system
    matrix.

    Args:
        A (np.ndarray): System matrix NxN
        omega (float): Overrelaxation parameter. Setting it to 1 leads to Symmetric Gauss-Seidel
        precondioning
    """
    if not is_symmetric(A):
        raise TypeError(
            f"System matrix is not symmetric A[0:2,0:2]: {A[0:2,0:2]}")
    D = np.diag(A)  # return 1-D array
    D_inv = np.reciprocal(D)  # element-wise inverse
    D_inv = np.diag(D_inv)
    D = np.diag(D)  # create 2-D matrix
    E = -np.tril(A, k=-1)  # get lower triangle without diagonal
    F = -np.triu(A, k=1)  # get upper triangle without diagonal

    M_ssor = 1/(omega*(2 - omega)) * (D - omega * E) @ D_inv @ (D - omega * F)
    return M_ssor


def compute_condition_number(A):
    if is_symmetric(A):
        eig_vals = np.linalg.eigvals(A)
        condition_number_A = np.max(np.abs(eig_vals))/np.min(np.abs(eig_vals))
    else:
        raise TypeError(
            "Input A is not symmetric and thus inv(M_ssor)*A is not symmetric")
    return condition_number_A


def create_gauss_seidel_error_propagation_matrix(A):
    M_gs = np.tril(A, 0)  # get lower triangle with diagonal
    F = np.triu(A, 1)  # get upper triangle without diagonal
    M_gs_inv = np.linalg.inv(M_gs)
    B_gs = np.eye(A.shape[0]) - np.matmul(M_gs_inv, A)
    eigvals_B_gs = np.linalg.eigvals(B_gs)
    spectral_radius = np.max(np.abs(eigvals_B_gs))

    return eigvals_B_gs, spectral_radius


def compute_series_of_ritz_values(ritz_values):
    """Restructure the Ritz values provided by a Prec. CG method
    into a list of Ritz values that should converge to the eigen
    values of the system matrix if the solver converged.

    Args:
        ritz_values (list of np.ndarray): List of Ritz values per iteration,
        k-th np.ndarray contains Ritz values of the k-th iteration.

    Returns:
        list of lists: i-th entry contains i elements of a sequence of the
        same ritz value
    """
    series_of_ritz_values = []
    for ritz_values_ith_iter in ritz_values:
        series_of_ritz_values.append([])
        for j, ritz_value in enumerate(ritz_values_ith_iter):
            series_of_ritz_values[j].append(ritz_value)
    return series_of_ritz_values


def experiments_exercise_2_3():
    figsize = (15, 8)
    c_values = [0.01, 0.1, 1, 10]
    grid_sizes = [10, 100, 1000]
    colors = cm.rainbow(np.linspace(0, 1, len(c_values)*len(grid_sizes)))
    spectral_rads = []
    plt.figure(figsize=figsize)
    for i, c_ in enumerate(c_values):
        for j, size in enumerate(grid_sizes):
            A = create_discretized_helmholtz_matrix(size, c_)
            eigvals, spectral_rad = create_gauss_seidel_error_propagation_matrix(
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
            A = create_discretized_helmholtz_matrix(size, c_)
            eigvals, spectral_rad = create_gauss_seidel_error_propagation_matrix(
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
            A = create_discretized_helmholtz_matrix(size, c_)
            eigvals, spectral_rad = create_gauss_seidel_error_propagation_matrix(
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
        for c in c_values:
            for grid_size in grid_sizes:
                rel_errors = []
                h = 1/grid_size
                x = np.linspace(0, 1, grid_size+1)
                x = x[1:-1]
                A = create_discretized_helmholtz_matrix(size=grid_size, c=c)
                _, spectral_rad = create_gauss_seidel_error_propagation_matrix(
                    A)
                rhs = f_rhs(c, x, h)
                u_sol, rel_errors, convergence_flag = helmholtz_solvers.gauss_seidel_solver(
                    A, rhs)
                if convergence_flag:
                    lines = [f"(c, h) = ({c}, {h})\n",
                             f"rate of convergence: {rel_errors[len(rel_errors)-5]:.3E}\n",
                             f"spectral rad.: {spectral_rad:.3E}\n",
                             f"min rate: {min(rel_errors):.3E}\n",
                             f"avg. rate: {np.average(np.array(rel_errors)[2:-2]):.3E}\n",
                             f"\n"]
                    print(*lines)
                    f.writelines(lines)
                else:
                    print(f"Convergence failed for (c, h) = ({c}, h={h})\n")
                    f.write(f"Convergence failed for (c, h) = ({c}, h={h})\n")


def experiments_exercise_5():
    # compute and tabulate condition number of symmetric gauss-seidel
    # preconditioned matrix inv(M_SGS)*A
    c_values = [0.01, 0.1, 1, 10]
    grid_sizes = [10, 100, 1000]

    with open("results_ex_5.txt", 'w') as f:
        for c in c_values:
            for grid_size in grid_sizes:
                h = 1/grid_size
                A = create_discretized_helmholtz_matrix(size=grid_size, c=c)
                condition_number_A,  = compute_condition_number(
                    A)
                M_ssor = compute_symmetric_ssor_preconditioner(A, 1.0)
                prec_operator = np.linalg.inv(M_ssor) * A
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
            A = create_discretized_helmholtz_matrix(size=grid_size, c=c)
            rhs = f_rhs(c, x, h)

            M_sgs = compute_symmetric_ssor_preconditioner(A, omega=1.0)
            M_sgs_inv = np.linalg.inv(M_sgs)

            # _, _ = helmholtz_solvers.preconditioned_conjugate_gradient_with_ritz(
            #     A, rhs, residuals=residuals_uncond)
            u_sol, convergence_flag = helmholtz_solvers.preconditioned_conjugate_gradient_with_ritz(
                A, f=rhs, M_inv=M_sgs_inv, ritz_values=ritz_values, residuals=residuals)
            # sequence_of_ritz_values = compute_series_of_ritz_values(
            #     ritz_values)
            # cond_a = compute_condition_number(A)
            cond_A = compute_condition_number(A)
            cond_prec = np.linalg.cond(M_sgs_inv @ A)
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
    # plt.show()
    # fig.savefig("figures/plot_ex_6_convergence.pdf")
    # fig.savefig("figures/plot_ex_6_convergence.svg")


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

    # A = create_discretized_helmholtz_matrix(5, 0)
    # print("A:\n", A)
    # create_gauss_seidel_error_propagation_matrix(A)

    # Exercise 02, 03
    # experiments_exercise_2_3()
    # Exercise 04
    # experiments_exercise_4()
    # Exercise 05
    # experiments_exercise_5()
    # Exercise 06
    experiments_exercise_6()
