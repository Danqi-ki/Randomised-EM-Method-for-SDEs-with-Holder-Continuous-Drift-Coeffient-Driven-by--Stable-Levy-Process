import numpy as np

############################################# the Weierstrass  function #######################################################################

def weierstrass_time(x, a=0.8, b=35, n_terms=50):
    """
    Compute the Weierstrass function at points x.

    Parameters:
    - x: float or numpy array
    - a: float, 0 < a < 1
    - b: odd integer > 1 such that ab > 1 + (3/2)*pi
    - n_terms: number of terms in the series (truncation)

    Returns:
    - float or numpy array: value(s) of the Weierstrass function
    """
    x = np.asarray(x)
    n = np.arange(n_terms).reshape(-1, 1)  # shape: (n_terms, 1)

    a_n = a ** n                          # shape: (n_terms, 1)
    b_n = b ** n                          # shape: (n_terms, 1)

    x = x.reshape(1, -1)                  # shape: (1, len(x))
    cos_terms = np.cos(np.pi * b_n * x)  # shape: (n_terms, len(x))

    result = np.sum(a_n * cos_terms, axis=0)  # sum over n_terms
    return result

def weierstrass_space(x, alpha=0.5, m_terms=50):
    j = np.arange(m_terms).reshape(-1, 1)
    coeffs = 2 ** (-alpha * j)
    freqs = 2 ** j
    x = np.asarray(x).reshape(1, -1)
    return np.sum(coeffs * np.cos(freqs * x), axis=0)

def drift_weierstrass(x, t, args):
    """
    Compute drift = W(t) * μ_alpha(x)

    Parameters:
    - x: float or np.array
    - t: float or np.array
    - args: dict with keys 'a', 'b', 'n_terms', 'alpha', 'm_terms'

    Returns:
    - float or array: drift value(s)
    """
    a,b,n_terms,m_terms=args
    W_t = weierstrass_time(t, a=a, b=b, n_terms=n_terms)
    mu_x = weierstrass_space(x, alpha=a, m_terms=m_terms)
    return W_t * mu_x










############################################# the jump-type time-dependent function #######################################################################

def g_vectorized(t, T):
    weights = np.linspace(1, 16, 16) / 16  
    coefficients = np.linspace(-1, -0.05, 16)   

    g = np.zeros_like(t, dtype=float)

    for w, c in zip(weights, coefficients):
        g += c * np.sign(w * T - t)   
    
    return g


def drift_jump(x, t, args):
    """
    Jump-type drift with a bounded state dependence:

        b(t, x) = g(t) * x / (1 + |x|)

    The soft saturation x -> x/(1+|x|) keeps |b| <= |g| and is globally
    Lipschitz, so the drift stays bounded in x as the assumptions require.
    """
    T = args[0]
    x = np.asarray(x)
    return g_vectorized(t, T) * x / (1.0 + np.abs(x))
