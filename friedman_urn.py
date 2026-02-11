import numpy as np


def friedman_urn(red_balls, black_balls, alpha, beta, n):
    R = red_balls
    B = black_balls

    X = np.empty(n)

    for k in range(n):
        p_red = R / (R + B)

        if np.random.rand() < p_red:
            # sale roja
            R += alpha
            B += beta
        else:
            # sale negra
            R += beta
            B += alpha

        X[k] = R / (R + B)

    return X


def friedman_montecarlo(red_balls, black_balls, alpha, beta, n_steps, n_runs):
    Xf = np.empty(n_runs)
    for i in range(n_runs):
        Xf[i] = friedman_urn(red_balls, black_balls, alpha, beta, n_steps)[-1]
    return Xf



if __name__ == "__main__":

    import matplotlib.pyplot as plt

    red_balls, black_balls = 10, 10
    alpha, beta = 3, 1
    n = 10**6

    for _ in range(8):
        X = friedman_urn(red_balls, black_balls, alpha, beta, n)
        plt.plot(X, alpha=0.7)

    plt.axhline(0.5, color='k', linestyle='--')
    plt.xlabel("n")
    plt.ylabel("Proporción de rojas")
    plt.title("Urna de Friedman (matriz simétrica positiva)")
    plt.show()

    X_inf = friedman_montecarlo(red_balls, black_balls, alpha, beta, n, 8_000)

    plt.hist(X_inf, bins=60, density=True)
    plt.axvline(0.5, color='k', linestyle='--')
    plt.title("Distribución empírica de $X_n$ (Friedman)")
    plt.show()
