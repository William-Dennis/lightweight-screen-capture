def safe_division(a, b, default=0, tol=1e-7):
    if abs(b) < tol:
        return default

    return a / b
