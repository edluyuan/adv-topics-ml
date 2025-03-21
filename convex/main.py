import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def generate_lasso_problem(n, d, s, std=0.06):
    """
    Generates the synthetic Lasso problem with specified parameters.

    Args:
        n (int): Number of data points.
        d (int): Number of features.
        s (int): Sparsity of the true solution (number of non-zero elements).
        std (float): Standard deviation of the noise.

    Returns:
        tuple: A tuple containing:
            - true_solution (np.ndarray): The true sparse solution vector.
            - design_matrix (np.ndarray): The design matrix A.
            - noisy_target (np.ndarray): The noisy target vector y.
    """
    assert s % 2 == 0, "s needs to be divisible by 2"

    # Create sparse true solution with positive and negative values
    positive_vals = 0.5 * (np.random.rand(s // 2) + 1)
    negative_vals = -0.5 * (np.random.rand(s // 2) + 1)
    true_solution = np.hstack([positive_vals, negative_vals, np.zeros(d - s)])
    np.random.shuffle(true_solution)  # Use numpy's shuffle for consistency

    # Generate design matrix A
    design_matrix = np.random.randn(n, d)

    # Generate noisy target vector y
    noisy_target = design_matrix @ true_solution + std * np.random.randn(n)

    return true_solution, design_matrix, noisy_target


def soft_thresholding(x, gamma):
    """
    Applies the soft-thresholding operator element-wise to a vector.

    Args:
        x (np.ndarray): The input vector.
        gamma (float): The threshold parameter.

    Returns:
        np.ndarray: The thresholded vector.
    """
    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0.0)


def compute_lasso_loss(A, x_sol, y, lam, n):
    """
    Computes the Lasso objective function value.

    Args:
        A (np.ndarray): The design matrix.
        x_sol (np.ndarray): The coefficient vector.
        y (np.ndarray): The target vector.
        lam (float): The regularization parameter.
        n (int): Number of data points.

    Returns:
        float: The value of the Lasso loss function.
    """
    residual = A @ x_sol - y
    loss = (1.0 / (2.0 * n)) * np.linalg.norm(residual) ** 2 + lam * np.linalg.norm(x_sol, ord=1)
    return loss


def proximal_stochastic_gradient_algorithm(x_init, y, n, A, lam, num_iterations, is_ergodic=False):
    """
    Proximal Stochastic Gradient Algorithm (PSGA) for solving the Lasso problem.

    Args:
        x_init (np.ndarray): The initial solution vector.
        y (np.ndarray): The target vector.
        n (int): Number of data points.
        A (np.ndarray): The design matrix.
        lam (float): The regularization parameter.
        num_iterations (int): The number of iterations to run the algorithm.
        is_ergodic (bool): Whether to compute the ergodic mean or not.

    Returns:
        tuple: A tuple containing:
            - solution (np.ndarray): The estimated solution after the specified number of iterations.
            - loss_history (list): The history of the Lasso loss function over iterations.
    """
    FroNormAsqrd = np.linalg.norm(A, ord='fro') ** 2
    x = x_init.copy()
    loss_history = []

    # Initialize for ergodic mean
    gamma_sum = 0.0
    gamma_x_sum = np.zeros_like(x)

    # Initialize progress bar
    with tqdm(total=num_iterations, desc="PSGA Progress", unit="iter") as pbar:
        for k in range(1, num_iterations + 1):
            # Randomly select an index i_k uniformly from {0, ..., n-1}
            ik = np.random.randint(0, n)

            # Step size gamma_k
            gamma_k = n / (FroNormAsqrd * np.sqrt(k))

            # Gradient step for PSGA
            gradient = (A[ik, :] @ x - y[ik]) * A[ik, :]
            t = x - gamma_k * gradient

            # Proximal operator (soft-thresholding)
            x = soft_thresholding(t, lam * gamma_k)



            # Update ergodic mean if needed
            if is_ergodic:
                gamma_sum += gamma_k
                gamma_x_sum += gamma_k * x
                ergodic_mean = gamma_x_sum / gamma_sum
                ergodic_loss = compute_lasso_loss(A, ergodic_mean, y, lam, n)
                loss_history.append(ergodic_loss)  # overwrite with ergodic loss

            else:
                # Compute and store loss
                current_loss = compute_lasso_loss(A, x, y, lam, n)
                loss_history.append(current_loss)

            # Update progress bar
            pbar.update(1)

    if is_ergodic:
        x = gamma_x_sum / gamma_sum

    return x, loss_history


def randomized_coordinate_proximal_gradient_algorithm(x_init, y, n, d, A, lam, num_iterations):
    """
    Randomized Coordinate Proximal Gradient Algorithm (RCPGA) for solving the Lasso problem.

    Args:
        x_init (np.ndarray): The initial solution vector.
        y (np.ndarray): The target vector.
        n (int): Number of data points.
        d (int): Number of features.
        A (np.ndarray): The design matrix.
        lam (float): The regularization parameter.
        num_iterations (int): The number of iterations to run the algorithm.

    Returns:
        tuple: A tuple containing:
            - solution (np.ndarray): The estimated solution after the specified number of iterations.
            - loss_history (list): The history of the Lasso loss function over iterations.
    """
    x = x_init.copy()
    loss_history = []

    with tqdm(total=num_iterations, desc="RCPGA Progress", unit="iter") as pbar:
        for k in range(1, num_iterations + 1):
            # Randomly select a coordinate j_k uniformly from {0, ..., d-1}
            jk = np.random.randint(0, d)

            # Step size gamma_j
            gamma_j = n / np.sum(A[:, jk] ** 2)

            # Gradient w.r.t. x_j
            grad_j = (A[:, jk] @ (A @ x - y)) / n

            # Gradient step for coordinate j_k
            t = x[jk] - gamma_j * grad_j

            # Proximal operator (soft-thresholding)
            x[jk] = soft_thresholding(t, lam * gamma_j)

            # Compute and store loss
            current_loss = compute_lasso_loss(A, x, y, lam, n)
            loss_history.append(current_loss)

            # Update progress bar
            pbar.update(1)

    return x, loss_history


def plot_solution(x_estimated, x_true, algorithm_name, d):
    """
    Plots the estimated solution against the true sparse solution.

    Args:
        x_estimated (np.ndarray): The estimated solution.
        x_true (np.ndarray): The true sparse solution.
        algorithm_name (str): The name of the algorithm for the plot title.
        d (int): Number of features.
    """
    plt.figure(figsize=(10, 6), dpi=100)

    # Plot true non-zero elements
    true_nonzero = np.where(np.abs(x_true) > 0)[0]
    plt.stem(true_nonzero, x_true[true_nonzero], linefmt='b-', markerfmt='bo', basefmt=' ', label='$x^*$')

    # Plot estimated non-zero elements
    estimated_nonzero = np.where(np.abs(x_estimated) > 1e-3)[0]
    plt.stem(estimated_nonzero, x_estimated[estimated_nonzero], linefmt='k--', markerfmt='k^', basefmt=' ', label=r'$x_{\gamma, \lambda}$')

    plt.axhline(0.0, color='red', linestyle='--', linewidth=1)
    plt.xlim([-10, d + 10])
    plt.ylim([-1.1, 1.1])
    plt.xlabel("Feature index $i$")
    plt.ylabel("$x_i$")
    plt.title(f"Sparse Solution vs Estimated Solution ({algorithm_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{algorithm_name}_solution.png")
    plt.close()


def plot_loss(loss_histories, legends, baseline_obj, algorithm_name):
    """
    Plots the loss history during the optimization process.

    Args:
        baseline_obj:  The baseline objective function value.
        loss_histories (list of lists): The history of the Lasso loss for different runs.
        legends (list of str): Labels for the loss curves.
        algorithm_name (str): The name of the algorithm for the plot title.
    """
    plt.figure(figsize=(10, 6), dpi=100)
    for loss, label in zip(loss_histories, legends):
        plt.plot(loss, label=label, linewidth=2)
    plt.axhline(
    y=baseline_obj,
    color='red',
    linewidth=2,
    linestyle='--',
    label=f"Baseline (true x) = {baseline_obj:.6f}"
    )
    plt.xlabel("Iteration")
    plt.ylabel("Objective function")
    plt.title(f"Loss vs Iterations ({algorithm_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{algorithm_name}_loss.png")
    plt.close()


def main():
    # Generate synthetic data
    n = 1000
    d = 500
    s = 50
    std = 0.06
    true_x, A, y = generate_lasso_problem(n, d, s, std)

    # Compute baseline objective function value using true x
    baseline_obj = compute_lasso_loss(A, true_x, y, lam=0.01, n=n)
    print(f"Baseline objective with true x: {baseline_obj:.6f}")

    # ------------------------
    # Proximal Stochastic Gradient Algorithm (PSGA)
    # ------------------------
    max_iters_psga = 1000000  # Max number of iterations for PSGA
    lam_psga = 0.05  # Regularization parameter for PSGA
    x0_psga = np.random.randn(d)  # Initialization of x0

    # Run PSGA without ergodic mean
    print("Running PSGA without ergodic mean...")
    estimated_x_psga, loss_history_psga = proximal_stochastic_gradient_algorithm(
        x_init=x0_psga,
        y=y,
        n=n,
        A=A,
        lam=lam_psga,
        num_iterations=max_iters_psga,
        is_ergodic=False
    )

    # Run PSGA with ergodic mean
    print("Running PSGA with ergodic mean...")
    estimated_x_psga_ergo, loss_history_psga_ergo = proximal_stochastic_gradient_algorithm(
        x_init=x0_psga,
        y=y,
        n=n,
        A=A,
        lam=lam_psga,
        num_iterations=max_iters_psga,
        is_ergodic=True
    )

    # Plot PSGAs loss histories
    plot_loss(
        loss_histories=[loss_history_psga, loss_history_psga_ergo],
        baseline_obj = baseline_obj,
        legends=["PSGA", "PSGA Ergodic"],
        algorithm_name="PSGA"
    )

    # Plot PSGA estimated solution vs true solution
    plot_solution(
        x_estimated=estimated_x_psga,
        x_true=true_x,
        algorithm_name="PSGA",
        d=d
    )

    # Plot PSGA with ergodic x estimated solution vs true solution
    plot_solution(
        x_estimated=estimated_x_psga_ergo,
        x_true=true_x,
        algorithm_name="PSGA_Ergodic",
        d=d
    )

    # ------------------------
    # Randomized Coordinate Proximal Gradient Algorithm (RCPGA)
    # ------------------------
    max_iters_rcpga = 10000  # Max number of iterations for RCPGA
    lam_rcpga = 0.05  # Regularization parameter for RCPGA
    x0_rcpga = np.random.randn(d)  # Initialization of x0

    # Run RCPGA
    print("Running RCPGA...")
    estimated_x_rcpga, loss_history_rcpga = randomized_coordinate_proximal_gradient_algorithm(
        x_init=x0_rcpga,
        y=y,
        n=n,
        d=d,
        A=A,
        lam=lam_rcpga,
        num_iterations=max_iters_rcpga
    )

    # Plot RCPGA loss history
    plot_loss(
        loss_histories=[loss_history_rcpga],
        baseline_obj = baseline_obj,
        legends=["RCPGA"],
        algorithm_name="RCPGA"
    )

    # Plot RCPGA estimated solution vs true solution
    plot_solution(
        x_estimated=estimated_x_rcpga,
        x_true=true_x,
        algorithm_name="RCPGA",
        d=d
    )

    # ------------------------
    # Combined Loss Plot (Optional)
    # ------------------------
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(loss_history_psga, label='PSGA', linewidth=2)
    plt.plot(loss_history_psga_ergo, label='PSGA Ergodic', linewidth=2)
    plt.plot(loss_history_rcpga, label='RCPGA', linewidth=2)
    plt.axhline(y=baseline_obj, color='red', linestyle='--', linewidth=2, label='Baseline (true x)')
    plt.xlabel("Iteration")
    plt.ylabel("Objective function")
    plt.title("Loss vs Iterations for PSGA and RCPGA")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Combined_Loss.png")
    plt.close()


if __name__ == "__main__":
    main()