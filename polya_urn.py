import numpy as np


def polya_urn(red_balls, black_balls, reinforcement, n):
    R = red_balls
    B = black_balls

    X = np.empty(n)

    for n in range(n):
        p_red = R / (R + B)

        if np.random.rand() < p_red:
            R += reinforcement
        else:
            B += reinforcement

        X[n] = R / (R + B)

    return X


def polya_montecarlo(red_balls, black_balls, reinforcement, n_steps, n_runs):
    X_final = np.empty(n_runs)

    for i in range(n_runs):
        X = polya_urn(red_balls, black_balls, reinforcement, n_steps)
        X_final[i] = X[-1]

    return X_final


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.stats import beta

    # parameters

    # Uniform limit
    # r = 1
    # s = 1

    # Beta general limit
    r = 7
    s = 3
    c = 1
    n_steps = 10_000
    n_runs = 20_000

    X_inf = polya_montecarlo(r, s, c, n_steps, n_runs)

    # histograma
    plt.hist(X_inf, bins=80, density=True, alpha=0.6, label="Simulación")

    # beta teórica
    x = np.linspace(0, 1, 500)
    a = r / c
    b = s / c
    plt.plot(x, beta.pdf(x, a, b), 'r-', lw=2, label=f"Beta({a},{b})")

    plt.xlabel("Proporción de rojas")
    plt.ylabel("Densidad")
    plt.legend()
    plt.title("Urna de Polya clásica")
    plt.show()