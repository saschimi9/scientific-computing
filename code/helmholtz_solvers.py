import numpy as np
import take_home_exam
import scipy.linalg as sp_la
from timeit import default_timer as timer


def is_symmetric(A):
    return np.allclose(A, np.transpose(A), rtol=1e-5, atol=1e-5)


def compute_symmetric_ssor_preconditioner(A, omega):
    """Return the preconditioning matrix M_SGS(w)^{-1} for the symmetric successive over relaxation (SSOR)
    with parameter $\omega$.

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

    iden = np.eye(D.shape[0])
    lower_inv = sp_la.solve_triangular((D - omega * E), iden, lower=True)
    upper_inv = sp_la.solve_triangular((D - omega * F), iden)
    M_ssor_inv = (omega*(2 - omega)) * upper_inv @ D @ lower_inv
    assert np.allclose(M_ssor @ M_ssor_inv, iden)

    return M_ssor_inv


def compute_symmetric_ssor_preconditioner_split(A, omega):
    """Return the preconditioning matrix M_SGS(w)^{-1} for the symmetric successive over relaxation (SSOR)
    with parameter $\omega$.

    Args:
        A (np.ndarray): System matrix NxN
        omega (float): Overrelaxation parameter. Setting it to 1 leads to Symmetric Gauss-Seidel
        precondioning
    """
    if not is_symmetric(A):
        raise TypeError(
            f"System matrix is not symmetric A[0:2,0:2]: {A[0:2,0:2]}")
    D = np.diag(A)  # return 1-D array
    D = np.diag(D)  # create 2-D matrix
    E = -np.tril(A, k=-1)  # get lower triangle without diagonal
    F = -np.triu(A, k=1)  # get upper triangle without diagonal

    iden = np.eye(D.shape[0])
    M1_ssor_inv = sp_la.solve_triangular((1/omega)*(D - E), iden, lower=True)
    M2_ssor_inv = sp_la.solve_triangular((1/omega)*(D - F), iden)

    return M1_ssor_inv, M2_ssor_inv


def compute_gauss_seidel_M_inverse(A_h, backwards=False):
    n = A_h.shape[0]
    if backwards:
        M_gs = np.triu(A_h, 0)  # get lower triangle with diagonal
        M_gs_inv = sp_la.solve_triangular(M_gs, np.eye(n), lower=False)
    else:
        M_gs = np.tril(A_h, 0)  # get lower triangle with diagonal
        M_gs_inv = sp_la.solve_triangular(M_gs, np.eye(n), lower=True)
    return M_gs_inv


def compute_gauss_seidel_M(A_h, backwards=False):
    if backwards:
        M_gs = np.triu(A_h, 0)  # get lower triangle with diagonal
    else:
        M_gs = np.tril(A_h, 0)  # get lower triangle with diagonal
    return M_gs


def gauss_seidel_iteration(A, rhs, u_initial, num_iterations=1, reverse=False):
    if reverse:
        generator = reversed(range(len(rhs)))
    else:
        generator = range(len(rhs))
    for _ in range(num_iterations):
        for i in generator:
            sweep1 = np.dot(A[i, 0:i], u_initial[0:i])
            sweep2 = np.dot(A[i, i+1:], u_initial[i+1:])
            # indices out of bounds correctly return empty arrays
            u_initial[i] = (rhs[i] - sweep1 - sweep2)/A[i, i]
    return u_initial


def gauss_seidel_solver(A, rhs, tol=1e-6, max_iterations=10000):
    u_sol = np.zeros(A.shape[1])
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
            sweep1 = np.dot(A[i, 0:i], u_sol[0:i])
            sweep2 = np.dot(A[i, i+1:], u_sol[i+1:])
            # indices out of bounds correctly return empty arrays
            u_sol[i] = (rhs[i] - sweep1 - sweep2)/A[i, i]

        rel_errors.append(np.linalg.norm(u_sol - u_sol_min_1) /
                          np.linalg.norm(u_sol_min_1 - u_sol_min_2))
        u_sol_min_2 = np.copy(u_sol_min_1)
        u_sol_min_1 = np.copy(u_sol)
        residual = rhs - A @ u_sol
        counter += 1

    if counter >= max_iterations:
        print(
            f"GS solver did not converge after {max_iterations} iterations")
    if np.linalg.norm(residual)/rhs_norm <= tol:
        convergence_flag = True
        print(f"GS converged after {counter} iterations on size {A.shape}")

    return u_sol, rel_errors, convergence_flag


def ssor_solver(A, rhs, tol=1e-8, omega=1, max_iterations=10000):
    """Symmetric successive overrelaxation with parameter.

    Args:
        A (np.array_like): System matrix NxN
        rhs (np.array_like): Right-hand side of the problem Nx1
        tol (float, optional): Tolerance for the stopping criterion. Defaults to 1e-5.
        omega (float, optional): Relaxation parameter. Defaults to 1.
        max_iterations (int, optional): Maximum number of iterations. Defaults to 10000.
    """
    if not is_symmetric(A):
        raise TypeError("A is not symmetric.")

    u_sol = np.zeros(A.shape[1])
    sigma = 0.0

    residual = rhs - A @ u_sol
    rhs_norm = np.linalg.norm(rhs)
    counter = 0
    convergence_flag = False

    while np.linalg.norm(residual)/rhs_norm > tol and counter < max_iterations:
        # Forward sweep
        for i in range(u_sol.shape[0]):
            sigma = u_sol[i]
            sweep1 = np.dot(A[i, 0:i], u_sol[0:i])
            sweep2 = np.dot(A[i, i+1:], u_sol[i+1:])
            u_sol[i] = (rhs[i] - sweep1 - sweep2)/A[i, i]
            u_sol[i] = omega * u_sol[i] + (1 - omega) * sigma
        # Backward sweep
        for i in reversed(range(u_sol.shape[0])):
            sigma = u_sol[i]
            sweep1 = np.dot(A[i, 0:i], u_sol[0:i])
            sweep2 = np.dot(A[i, i+1:], u_sol[i+1:])
            u_sol[i] = (rhs[i] - sweep1 - sweep2)/A[i, i]
            u_sol[i] = omega * u_sol[i] + (1 - omega) * sigma
        residual = rhs - A @ u_sol
        counter += 1

    if counter >= max_iterations:
        print(
            f"SSOR solver did not converge after {max_iterations} iterations with omega {omega}.")
    if np.linalg.norm(residual)/rhs_norm <= tol:
        convergence_flag = True
        print(f"SSOR converged after {counter} iterations on size {A.shape}")

    return u_sol, convergence_flag


def preconditioned_conjugate_gradient(A, rhs, M_inv=None, max_iterations=5000, tol=1e-8, residuals=None):
    """
    Conjugate Gradient algorithm with Ritz value computation for a one-dimensional problem.

    Parameters:
    - A: Symmetric positive definite matrix (1D array representing the diagonal elements).
    - f: Right-hand side vector.
    - M_inv: Inverted preconditioning matrix
    - max_iterations: Maximum number of iterations.
    - tol: Tolerance for convergence.
    - residual: Empty list for residuals.

    Returns:
    - u_sol: Solution vector.
    - convergence_flag: True if convergence was reached.
    """
    start_time = timer()
    counter = 0
    convergence_flag = False
    n = len(rhs)
    if M_inv is None:
        M_inv = np.eye(n)

    u_sol = np.zeros(n)  # Initial guess
    z_sol = np.zeros(n)
    scp = 0
    scp_old = 0
    rhs_norm = np.linalg.norm(rhs, ord=2)
    residual = rhs.copy()
    if residuals is not None:
        residuals.append(residual)

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

        counter += 1

    if counter >= max_iterations:
        print(
            f"Preconditioned CG solver did not converge after {max_iterations} iterations with M_inv {M_inv}.")
    if np.linalg.norm(residual)/rhs_norm <= tol:
        convergence_flag = True
        print(
            f"Prec. CG converged after {counter} iterations on size {A.shape}")

    end_time = timer()
    print(f'time spent: {end_time-start_time:.2g}')

    return u_sol, convergence_flag


def preconditioned_conjugate_gradient_type2(A, rhs, M_inv=None, max_iterations=5000, tol=1e-8, residuals=None):
    """
    Conjugate Gradient algorithm for a one-dimensional problem.
    Left preconditioning the system matrix A instead of the residual.
    -> convenient if N^2*(N + 1) < N^2 * iterations it takes to CG to converge.

    Parameters:
    - A: Symmetric positive definite matrix (1D array representing the diagonal elements).
    - rhs: Right-hand side vector.
    - M_inv: Inverted preconditioning matrix
    - max_iterations: Maximum number of iterations.
    - tol: Tolerance for convergence.
    - residual: Empty list for residuals.

    Returns:
    - u_sol: Solution vector.
    - convergence_flag: True if convergence was reached.
    """
    start_time = timer()
    counter = 0
    convergence_flag = False
    n = len(rhs)
    if M_inv is None:
        M_inv = np.eye(n)
    prec_A = M_inv @ A

    u_sol = np.zeros(n)  # Initial guess
    z_sol = np.zeros(n)
    scp = 0
    scp_old = 0
    rhs = M_inv @ rhs
    rhs_norm = np.linalg.norm(rhs, ord=2)
    residual = rhs.copy()
    if residuals is not None:
        residuals.append(residual)

    while np.linalg.norm(residual)/rhs_norm > tol and counter < max_iterations:
        z_sol = residual
        scp_old = scp
        scp = np.dot(residual, z_sol)
        if counter == 0:
            p_sol = z_sol.copy()
        else:
            beta_k = scp / scp_old
            p_sol = z_sol + beta_k * p_sol

        prod_A_p_sol = prec_A @ p_sol
        alpha_k = scp / np.dot(p_sol, prod_A_p_sol)
        u_sol = u_sol + alpha_k * p_sol
        residual = residual - alpha_k * prod_A_p_sol

        if residuals is not None:
            residuals.append(residual)

        counter += 1

    if counter >= max_iterations:
        print(
            f"Preconditioned CG solver did not converge after {max_iterations} iterations with M_inv {M_inv}.")
    if np.linalg.norm(residual)/rhs_norm <= tol:
        convergence_flag = True
        print(
            f"Prec. CG converged after {counter} iterations on size {A.shape}")

    end_time = timer()
    print(f'time spent: {end_time-start_time:.2g}')

    return u_sol, convergence_flag


def preconditioned_conjugate_gradient_type3(A, rhs, M_inv=None, max_iterations=5000, tol=1e-8, residuals=None):
    """
    Conjugate Gradient algorithm for a one-dimensional problem.
    Splitting the preconditioning matrix M = M1 * M2 such that the
    preconditioned system matrix is M1_inv * A * M2_inv.

    Parameters:
    - A: Symmetric positive definite matrix (1D array representing the diagonal elements).
    - rhs: Right-hand side vector.
    - M_inv: List of the inverted preconditioning matrices M1_inv and M2_inv.
    - max_iterations: Maximum number of iterations.
    - tol: Tolerance for convergence.
    - residual: Empty list for residuals.

    Returns:
    - u_sol: Solution vector.
    - convergence_flag: True if convergence was reached.
    """
    start_time = timer()
    counter = 0
    convergence_flag = False
    n = len(rhs)
    if len(M_inv) != 2:
        raise TypeError(
            f"Argument 'M_inv' needs to be a list of length 2, but has length {len(M_inv)}")
    prec_A = M_inv[0] @ A @ M_inv[1]

    u_sol = np.zeros(n)  # Initial guess
    z_sol = np.zeros(n)
    scp = 0
    scp_old = 0
    rhs = M_inv[0] @ rhs
    rhs_norm = np.linalg.norm(rhs, ord=2)
    residual = rhs.copy()
    if residuals is not None:
        residuals.append(residual)

    while np.linalg.norm(residual)/rhs_norm > tol and counter < max_iterations:
        z_sol = residual
        scp_old = scp
        scp = np.dot(residual, z_sol)
        if counter == 0:
            p_sol = z_sol.copy()
        else:
            beta_k = scp / scp_old
            p_sol = z_sol + beta_k * p_sol

        prod_A_p_sol = prec_A @ p_sol
        alpha_k = scp / np.dot(p_sol, prod_A_p_sol)
        u_sol = M_inv[1] @ (u_sol + alpha_k * p_sol)
        residual = residual - alpha_k * prod_A_p_sol

        if residuals is not None:
            residuals.append(residual)

        counter += 1

    if counter >= max_iterations:
        print(
            f"Preconditioned CG solver did not converge after {max_iterations} iterations with M_inv {M_inv}.")
    if np.linalg.norm(residual)/rhs_norm <= tol:
        convergence_flag = True
        print(
            f"Prec. CG converged after {counter} iterations on size {A.shape}")

    end_time = timer()
    print(f'time spent: {end_time-start_time:.2g}')

    return u_sol, convergence_flag


# Coarse grid correction
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


def create_coarse_grid_correction_M_inv(A_h):
    I_toFine = create_prolongation_matrix(A_h.shape[0]+1)
    I_toCoarse = create_coarsening_matrix(A_h.shape[0]+1)
    A_H = I_toCoarse @ A_h @ I_toFine
    A_H_inv = np.linalg.inv(A_H)

    M_cgc_inv = I_toFine @ A_H_inv @ I_toCoarse
    return M_cgc_inv


def create_coarse_grid_correction_error_propagation_matrix(A_h):
    M_cgc_inv = create_coarse_grid_correction_M_inv(A_h)
    B_cgc = np.eye(A_h.shape[0]) - M_cgc_inv @ A_h

    return B_cgc


def create_two_grid_method_error_propagation_matrix(A_h):
    pass


def create_two_grid_method_M_inv(A_h):
    pass


def coarse_grid_correction(A_h, rhs_h, max_iterations=10000, tol=1e-8, residuals=None):
    start_time = timer()
    counter = 0
    convergence_flag = False
    rhs_norm = np.linalg.norm(rhs_h, ord=2)

    u_sol = rhs_h.copy()
    residual = A_h @ u_sol - rhs_h

    M_CGC_inv = create_coarse_grid_correction_M_inv(A_h)
    rhs_prec = M_CGC_inv @ rhs_h
    B_cgc = create_coarse_grid_correction_error_propagation_matrix(A_h)

    while np.linalg.norm(residual)/rhs_norm > tol and counter < max_iterations:
        u_sol = B_cgc @ u_sol + rhs_prec
        residual = A_h @ u_sol - rhs_h
        if residuals is not None:
            residuals.append(residual)
        counter += 1

    if counter >= max_iterations:
        print(
            f"CGC solver did not converge after {max_iterations} iterations.")
    if np.linalg.norm(rhs_h - A_h @ u_sol)/rhs_norm <= tol:
        convergence_flag = True
        print(f"CGC converged after {counter} iterations on size {A_h.shape}")

    end_time = timer()
    print(f'time spent: {end_time-start_time:.2g}')
    return u_sol, convergence_flag


def two_grid_method_matrix(A_h, rhs_h, max_iterations=10000, tol=1e-8, num_presmoothing_iter=1, num_postsmoothing_iter=1, residuals=None):
    n = len(rhs_h)
    I_toCoarse = create_coarsening_matrix(n+1)
    I_toFine = create_prolongation_matrix(n+1)
    A_2h = I_toCoarse @ A_h @ I_toFine

    nu1 = num_presmoothing_iter
    nu2 = num_postsmoothing_iter

    M_gs_inv = compute_gauss_seidel_M_inverse(A_h)
    M_gs_inv_bw = compute_gauss_seidel_M_inverse(A_h, backwards=True)
    return None


def two_grid_method_point_wise(A_h, rhs_h, max_iterations=100, tol=1e-8, internal_solver='direct', num_presmoothing_iter=1, num_postsmoothing_iter=1, residuals=None):
    if internal_solver not in ['gs', 'cg', 'direct']:
        raise ValueError(
            "Argument 'internal_solver' needs to be 'cg' or 'gs' or 'direct'.")
    start_time = timer()
    counter = 0
    convergence_flag = False
    n = len(rhs_h)
    rhs_norm = np.linalg.norm(rhs_h, ord=2)
    I_toCoarse = create_coarsening_matrix(n+1)
    I_toFine = create_prolongation_matrix(n+1)

    u_sol = rhs_h.copy()
    u_h1 = np.zeros(n)
    u_h2 = np.zeros(n)
    r_h = rhs_h.copy()
    r_2h = np.zeros(n//2+1)
    A_2h = I_toCoarse @ A_h @ I_toFine

    nu1 = num_presmoothing_iter
    nu2 = num_postsmoothing_iter

    M_gs_inv = compute_gauss_seidel_M_inverse(A_h)
    M_gs_inv_bw = compute_gauss_seidel_M_inverse(A_h, backwards=True)
    M_sgs_inv = compute_symmetric_ssor_preconditioner(A_2h, omega=1.0)

    while np.linalg.norm(rhs_h - A_h @ u_sol)/rhs_norm > tol and counter < max_iterations:
        u_h1 = gauss_seidel_iteration(
            A_h, rhs_h, u_initial=u_sol, num_iterations=nu1)
        + M_gs_inv @ rhs_h
        r_h = rhs_h - A_h @ u_h1
        r_2h = I_toCoarse @ r_h
        if internal_solver == 'gs':
            e_2h, _, convergence_flag = gauss_seidel_solver(
                A_2h, r_2h, tol=1e-10)
        elif internal_solver == 'cg':
            e_2h, convergence_flag = preconditioned_conjugate_gradient(
                A_2h, r_2h, M_inv=M_sgs_inv, tol=1e-10)
        elif internal_solver == 'direct':
            e_2h = np.linalg.solve(A_2h, r_2h)
        # assert convergence_flag_gs
        e_h = I_toFine @ e_2h
        u_h2 = u_h1 + e_h
        u_sol = gauss_seidel_iteration(
            A_h, rhs_h, u_h2, num_iterations=nu2, reverse=True)
        + M_gs_inv_bw @ rhs_h
        if residuals is not None:
            residuals.append(A_h @ u_sol - rhs_h)
        counter += 1

    if counter >= max_iterations:
        print(
            f"Preconditioned CCG solver did not converge after {max_iterations} iterations.")
    if np.linalg.norm(rhs_h - A_h @ u_sol)/rhs_norm <= tol:
        convergence_flag = True
        print(f"CGC converged after {counter} iterations on size {A_h.shape}")

    end_time = timer()
    print(f'time spent: {end_time-start_time:.2g}')

    return u_sol, convergence_flag
