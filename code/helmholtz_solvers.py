import numpy as np


def is_symmetric(A):
    return np.allclose(A, np.transpose(A), rtol=1e-5, atol=1e-5)


def gauss_seidel_solver(A, rhs, tol=1e-5, num_iterations=10000):
    M_gs = np.tril(A, k=0)  # get lower triangle with diagonal
    F = np.triu(A, k=1)  # get upper triangle without diagonal
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
    E = np.tril(A, k=-1)  # get lower triangle without diagonal
    F = np.triu(A, k=1)  # get upper triangle without diagonal

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


def conjugate_gradient_with_ritz(A, b, max_iter=100, tol=1e-8, ritz_values=None):
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


def preconditioned_conjugate_gradient_with_ritz(A, b, max_iter=100, tol=1e-8, ritz_values=None):
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
