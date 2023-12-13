
import numpy as np
import scipy.linalg as scla
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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


def conjugate_gradient_with_ritz(A, b, max_iter=100, tol=1e-8):
    """
    Conjugate Gradient algorithm with Ritz value computation for a one-dimensional problem.

    Parameters:
    - A: Symmetric positive definite matrix (1D array representing the diagonal elements).
    - b: Right-hand side vector.
    - max_iter: Maximum number of iterations.
    - tol: Tolerance for convergence.

    Returns:
    - x: Solution vector.
    - ritz_values: List of Ritz values computed at each iteration.
    """

    n = len(b)
    x = np.zeros(n)  # Initial guess
    r = b - np.diag(A) * x
    p = r.copy()
    r_norm = np.linalg.norm(r, ord=2)
    T_k = np.array(r_norm)
    ritz_values = []

    for k in range(max_iter):
        Ap = np.diag(A) * p
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap

        # Compute the Ritz value
        ritz_value = np.dot(r_new, r_new) / np.dot(r, r)
        ritz_values.append(ritz_value)

        if np.linalg.norm(r_new) < tol:
            break

        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new

    return x, ritz_values


def boundary_conditions():
    """Return the boundary conditions alpha, beta = 1, 0."""
    alpha = 1
    beta = 0
    return alpha, beta


def gauss_seidel_solver(A, rhs, tol=1e-5, num_iterations=10000):
    M_gs = np.tril(A, 0)  # get lower triangle with diagonal
    F = np.triu(A, 1)  # get upper triangle without diagonal
    u_sol = np.zeros(A.shape[1])
    u_sol_old = np.copy(u_sol)
    u_sol_min_1 = np.copy(u_sol)
    u_sol_min_2 = np.copy(u_sol) + tol  # prevent division by zero

    residual = rhs - A @ u_sol
    rhs_norm = np.linalg.norm(rhs)
    counter = 0
    rel_errors = []
    convergence_flag = False

    # choice of stopping criterion p. 82
    while np.linalg.norm(residual)/rhs_norm > tol and counter < num_iterations:
        for i in range(u_sol.shape[0]):
            u_sol[i] = rhs[i] - np.dot(A[i, 0:i], u_sol[0:i])
            # indices out of bounds correctly return empty arrays
            u_sol[i] -= np.dot(A[i, i+1:], u_sol_min_1[i+1:])
            u_sol[i] /= A[i, i]
            u_sol_old[i] = u_sol[i]

        rel_errors.append(np.abs(np.linalg.norm(u_sol - u_sol_min_1) /
                                 np.linalg.norm(u_sol_min_1 - u_sol_min_2)))
        u_sol_min_2 = np.copy(u_sol_min_1)
        u_sol_min_1 = np.copy(u_sol)
        residual = rhs - A @ u_sol
        counter += 1

    if counter >= num_iterations:
        print(
            f"GS solver did not converge after {num_iterations} iterations")
    if np.linalg.norm(residual)/rhs_norm <= tol:
        convergence_flag = True

    return u_sol, rel_errors, convergence_flag


def is_symmetric(A):
    return np.allclose(A, np.transpose(A), rtol=1e-5, atol=1e-5)


def ssor_solver(A, rhs, tol=1e-5, omega=1, num_iterations=10000):
    """Symmetric successive overrelaxation with parameter.

    Args:
        A (np.array_like): System matrix NxN
        rhs (np.array_like): Right-hand side of the problem Nx1
        tol (float, optional): Tolerance for the stopping criterion. Defaults to 1e-5.
        omega (float, optional): Relaxation parameter. Defaults to 1.
        num_iterations (int, optional): Maximum number of iterations. Defaults to 10000.
    """
    D = np.diag(A)
    D_inv = np.reciprocal(D)
    E = np.tril(A, 1)  # get lower triangle without diagonal
    F = np.triu(A, 1)  # get upper triangle without diagonal

    M_ssor = 1/(omega*(2 - omega)) * (D - omega * F) * D_inv * (D - omega * E)
    # check if M_ssor is symmetric if A is symmetric
    if is_symmetric(A):
        if not is_symmetric(M_ssor):
            raise ValueError("Some computation is wrong.")
    else:
        raise TypeError("A is not symmetric.")

    u_sol = np.zeros(A.shape[1])
    sigma = np.zeros(A.shape[1])

    residual = rhs - A @ u_sol
    rhs_norm = np.linalg.norm(rhs)
    counter = 0
    rel_errors = []
    convergence_flag = False

    while np.linalg.norm(residual)/rhs_norm > tol and counter < num_iterations:
        # Forward sweep
        for i in range(u_sol.shape[0]):
            sigma[i] = u_sol[i]
            u_sol[i] = rhs[i] - np.dot(A[i, 0:i], u_sol[0:i])
            u_sol[i] -= np.dot(A[i, i+1:], u_sol[i+1:])
            u_sol /= A[i, i]
            u_sol = omega * u_sol[i] + (1 - omega) * sigma[i]
        # Backward sweep
        for i in reversed(range(u_sol.shape[0])):
            sigma[i] = u_sol[i]
            u_sol[i] = rhs[i] - np.dot(A[i, 0:i], u_sol[0:i])
            u_sol[i] -= np.dot(A[i, i+1:], u_sol[i+1:])
            u_sol /= A[i, i]
            u_sol = omega * u_sol[i] + (1 - omega) * sigma[i]
        residual = rhs - A @ u_sol
        counter += 1

    if counter >= num_iterations:
        print(
            f"SSOR solver did not converge after {num_iterations} iterations with omega {omega}.")
    if np.linalg.norm(residual)/rhs_norm <= tol:
        convergence_flag = True

    return u_sol, rel_errors, convergence_flag


def compute_condition_number(A):
    # Exercise 5
    omega = 1
    D = np.diag(A)
    D_inv = np.reciprocal(D)
    E = np.tril(A)  # get lower triangle without diagonal
    F = np.triu(A)  # get upper triangle without diagonal

    M_ssor = 1/(omega*(2 - omega)) * (D - omega * F) * D_inv * (D - omega * E)
    prec_operator = np.linalg.inv(M_ssor) * A
    if is_symmetric(prec_operator):
        eig_vals = np.linalg.eigvals(M_ssor)
        # Condition number of symmetric matrices can be computed using absolute values
        # of the maximum and minimum eigenvalue.
        condition_number = np.max(np.abs(eig_vals))/np.min(np.abs(eig_vals))
    else:
        raise TypeError(
            "Input A or M_ssor are not symmetric and thus inv(M_ssor)*A is not symmetric")
    return condition_number


def create_gauss_seidel_error_propagation_matrix(A):
    M_gs = np.tril(A, 0)  # get lower triangle with diagonal
    F = np.triu(A, 1)  # get upper triangle without diagonal
    M_gs_inv = np.linalg.inv(M_gs)
    B_gs = np.eye(A.shape[0]) - np.matmul(M_gs_inv, A)
    eigvals_B_gs = np.linalg.eigvals(B_gs)
    spectral_radius = np.max(np.abs(eigvals_B_gs))

    return eigvals_B_gs, spectral_radius


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
                u_sol, rel_errors, convergence_flag = gauss_seidel_solver(
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
                condition_number = compute_condition_number(A)
                line = f"(c, h) = ({c}, {h}) condition num.: {condition_number}\n"
                f.write(line)


if __name__ == "__main__":
    # Example usage:
    A = create_discretized_helmholtz_matrix(5, 0.1)

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
    experiments_exercise_5()
