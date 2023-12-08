
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


# Exercise 2
def create_gauss_seidel_error_propagation_matrix(A):
    M_gs = np.tril(A, 0)  # get lower triangle with diagonal
    F = np.triu(A, 1)  # get upper triangle without diagonal
    # M_gs_inv = scla.lapack.dtrtri(M_gs)
    M_gs_inv = np.linalg.inv(M_gs)
    B_gs = np.eye(A.shape[0]) - np.matmul(M_gs_inv, A)
    eigvals_B_gs = np.linalg.eigvals(B_gs)

    # print("Eigenvalues:")
    # print(eigvals_B_gs)

    spectral_radius = np.max(np.abs(eigvals_B_gs))
    print("spectral radius:\n", spectral_radius)

    return eigvals_B_gs, spectral_radius


def experiments_exercise_2():
    c_values = [0.01, 0.1, 1, 10]
    # c_values = [0.01, 0.1, 10]
    sizes = [10, 100, 1000]
    colors = cm.rainbow(np.linspace(0, 1, len(c_values)*len(sizes)))
    spectral_rads = dict()
    for i, c_ in enumerate(c_values):
        for j, size in enumerate(sizes):
            A = create_discretized_helmholtz_matrix(size, c_)
            eigvals, spectral_rad = create_gauss_seidel_error_propagation_matrix(
                A)
            spectral_rads[(c_, size)] = spectral_rad
            # Plot the numerical and analytical solutions
            plt.scatter(np.real(eigvals), np.imag(eigvals),
                        label=f'h = {1/size:.1E}, c = {c_:.1E})',
                        color=colors[i*len(sizes)+j], facecolors='none')

    plt.xlabel('Real part')
    plt.ylabel('Imag part')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    c_values = [1]
    # c_values = [0.01, 0.1, 10]
    sizes = [10, 100, 1000]
    colors = cm.rainbow(np.linspace(0, 1, len(c_values)*len(sizes)))
    spectral_rads = dict()
    for i, c_ in enumerate(c_values):
        for j, size in enumerate(sizes):
            A = create_discretized_helmholtz_matrix(size, c_)
            eigvals, spectral_rad = create_gauss_seidel_error_propagation_matrix(
                A)
            spectral_rads[(c_, size)] = spectral_rad
            # Plot the numerical and analytical solutions
            plt.scatter(np.real(eigvals), np.imag(eigvals),
                        label=f'h = {1/size:.1E}, c = {c_:.1E})',
                        color=colors[i*len(sizes)+j], facecolors='none')

    plt.xlabel('Real part')
    plt.ylabel('Imag part')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    c_values = [0.1, 1, 10, 100, 1000]
    # c_values = [0.01, 0.1, 10]
    sizes = [1000]
    colors = cm.rainbow(np.linspace(0, 1, len(c_values)*len(sizes)))
    spectral_rads = dict()
    for i, c_ in enumerate(c_values):
        for j, size in enumerate(sizes):
            A = create_discretized_helmholtz_matrix(size, c_)
            eigvals, spectral_rad = create_gauss_seidel_error_propagation_matrix(
                A)
            spectral_rads[(c_, size)] = spectral_rad
            # Plot the numerical and analytical solutions
            plt.scatter(np.real(eigvals), np.imag(eigvals),
                        label=f'h = {1/size:.1E}, c = {c_:.1E})',
                        color=colors[i*len(sizes)+j], facecolors='none')

    plt.xlabel('Real part')
    plt.ylabel('Imag part')
    plt.legend()
    plt.grid(True)
    plt.show()


def gauss_seidel_solver(A, rhs, tol=1e-5, num_iterations=20000):
    M_gs = np.tril(A, 0)  # get lower triangle with diagonal
    F = np.triu(A, 1)  # get upper triangle without diagonal
    u_sol = np.zeros(A.shape[1])
    u_sol_old = np.copy(u_sol)

    residual = rhs - A @ u_sol
    rhs_norm = np.linalg.norm(rhs)
    counter = 0
    # choice of stopping criterion p. 82
    while np.linalg.norm(residual)/rhs_norm > tol and counter < num_iterations:
        for i in range(u_sol.shape[0]):
            u_sol[i] = rhs[i] - np.dot(A[i, 0:i], u_sol[0:i])
            # indices out of bounds correctly return empty arrays
            u_sol[i] -= np.dot(A[i, i+1:], u_sol_old[i+1:])
            u_sol[i] /= A[i, i]
            u_sol_old[i] = u_sol[i]
        residual = rhs - A @ u_sol
        counter += 1

    if counter >= num_iterations:
        raise ValueError(
            f"GS solver did not converge after {num_iterations} iterations")

    return u_sol


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

    # Exercise 02
    experiments_exercise_2()
