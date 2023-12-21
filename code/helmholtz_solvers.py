import numpy as np


def is_symmetric(A):
    return np.allclose(A, np.transpose(A), rtol=1e-5, atol=1e-5)


def gauss_seidel_solver(A, rhs, tol=1e-5, max_iterations=10000):
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
    while np.linalg.norm(residual)/rhs_norm > tol and counter < max_iterations:
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

    if counter >= max_iterations:
        print(
            f"GS solver did not converge after {max_iterations} iterations")
    if np.linalg.norm(residual)/rhs_norm <= tol:
        convergence_flag = True

    return u_sol, rel_errors, convergence_flag


def ssor_solver(A, rhs, tol=1e-5, omega=1, max_iterations=10000):
    """Symmetric successive overrelaxation with parameter.

    Args:
        A (np.array_like): System matrix NxN
        rhs (np.array_like): Right-hand side of the problem Nx1
        tol (float, optional): Tolerance for the stopping criterion. Defaults to 1e-5.
        omega (float, optional): Relaxation parameter. Defaults to 1.
        max_iterations (int, optional): Maximum number of iterations. Defaults to 10000.
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

    while np.linalg.norm(residual)/rhs_norm > tol and counter < max_iterations:
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

    if counter >= max_iterations:
        print(
            f"SSOR solver did not converge after {max_iterations} iterations with omega {omega}.")
    if np.linalg.norm(residual)/rhs_norm <= tol:
        convergence_flag = True

    return u_sol, rel_errors, convergence_flag


def conjugate_gradient_with_ritz(A, f, max_iterations=100, tol=1e-8, ritz_values=None):
    """
    Conjugate Gradient algorithm with Ritz value computation for a one-dimensional problem.

    Parameters:
    - A: Symmetric positive definite matrix (1D array representing the diagonal elements).
    - f: Right-hand side vector.
    - max_iterations: Maximum number of iterations.
    - tol: Tolerance for convergence.

    Returns:
    - x: Solution vector.
    - ritz_values: List of Ritz values computed at each iteration.
    """

    n = len(f)
    u_sol = np.zeros(n)  # Initial guess
    r_sol = f.copy()
    # M_ssor = compute
    r = f - np.diag(A) * x
    p = r.copy()
    r_norm = np.linalg.norm(r, ord=2)
    T_k = np.array(r_norm)
    ritz_values = []

    for k in range(max_iterations):
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


def preconditioned_conjugate_gradient_with_ritz(A, f, M_inv=None, max_iterations=5000, tol=1e-8, ritz_values=None, residuals=None):
    """
    Conjugate Gradient algorithm with Ritz value computation for a one-dimensional problem.

    Parameters:
    - A: Symmetric positive definite matrix (1D array representing the diagonal elements).
    - f: Right-hand side vector.
    - M_inv: Inverted preconditioning matrix
    - max_iterations: Maximum number of iterations.
    - tol: Tolerance for convergence.
    - ritz_values: Empty list for Ritz values.
    - residual: Empty list for residuals.

    Returns:
    - u_sol: Solution vector.
    - convergence_flag: True if convergence was reached.
    """
    counter = 0
    convergence_flag = False
    n = len(f)
    if M_inv is None:
        M_inv = np.eye(n)

    u_sol = np.zeros(n)  # Initial guess
    z_sol = np.zeros(n)
    scp = 0
    scp_old = 0
    rhs_norm = np.linalg.norm(f, ord=2)
    residual = f.copy()
    if residuals is not None:
        residuals.append([residual])

    R_k_vectors = [residual/rhs_norm]
    T_k_matrices = [R_k_vectors[0].T @ A @ R_k_vectors[0]]  # R_k^T @ A @ R_k
    assert np.isscalar(T_k_matrices[0])  # should be k x N x N x k == 1

    if ritz_values is not None:
        ritz_values.append([T_k_matrices[0]])

    while np.linalg.norm(residual)/rhs_norm > tol and counter < max_iterations:
        z_sol = M_inv @ residual
        scp_old = scp
        scp = np.dot(residual, z_sol)
        if counter == 0:
            p_sol = z_sol.copy()
        else:
            beta_k = scp / scp_old
            p_sol = z_sol + beta_k * p_sol

        prod_A_p_sol = A @ p_sol
        alpha_k = scp / np.dot(p_sol, prod_A_p_sol)
        u_sol = u_sol + alpha_k * p_sol
        residual = residual - alpha_k * prod_A_p_sol

        if residuals is not None:
            residuals.append(residual)
        R_k_vectors.append(residual/np.linalg.norm(residual, ord=2))
        counter += 1

    # Compute Ritz matrix and values
    R_k_vectors = np.array(R_k_vectors).T
    if ritz_values is not None:
        for i in range(1, len(R_k_vectors)):
            T_k_matrix = R_k_vectors[:, 0:i].T @ A @ R_k_vectors[:, 0:i]
            T_k_matrices.append(T_k_matrix)
            ritz_values.append(np.linalg.eigvals(T_k_matrix))

    if counter >= max_iterations:
        print(
            f"Preconditioned CG solver did not converge after {max_iterations} iterations with M_inv {M_inv}.")
    if np.linalg.norm(residual)/rhs_norm <= tol:
        convergence_flag = True

    return u_sol, convergence_flag
