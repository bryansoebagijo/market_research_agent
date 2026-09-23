"""
Features and models for TOTAL3, kept apart from the harness that grades them.

eval/total3_forecast.py trains these walk-forward and measures them against the baselines
a forecast has to clear. The dashboard calls predict_now() only alongside that verdict,
never on its own.

Two models, because "how far" and "which way" are different questions with different
baselines:

  * ridge on the log return — magnitude, graded against the random walk (no change).
  * logistic on the sign    — direction, graded against the base rate, the share of past
                              windows that were up. A classifier that always says "up" in
                              a rising year looks accurate while knowing nothing, so a
                              probability has to beat climatology before it means anything.

Everything here uses numpy only. Implementing ridge and logistic directly keeps
scikit-learn out of requirements.txt for two models that are a dozen lines each.
"""

import numpy as np

RIDGE_PENALTY = 1.0
LOGISTIC_PENALTY = 1.0
RSI_WINDOW = 14


def _rsi(levels: np.ndarray, window: int = RSI_WINDOW) -> np.ndarray:
    """Wilder's RSI, matching the definition used elsewhere in this project."""
    delta = np.diff(levels, prepend=levels[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros_like(levels)
    avg_loss = np.zeros_like(levels)
    avg_gain[window] = gain[1:window + 1].mean()
    avg_loss[window] = loss[1:window + 1].mean()
    for i in range(window + 1, len(levels)):
        avg_gain[i] = (avg_gain[i - 1] * (window - 1) + gain[i]) / window
        avg_loss[i] = (avg_loss[i - 1] * (window - 1) + loss[i]) / window
    rs = np.divide(avg_gain, np.where(avg_loss == 0, np.nan, avg_loss))
    rsi = 100 - 100 / (1 + rs)
    rsi[:window] = 50.0
    return np.nan_to_num(rsi, nan=50.0)


def build_features(levels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Features known at day t, using nothing from any day after it."""
    log_levels = np.log(levels)
    daily = np.diff(log_levels, prepend=log_levels[0])

    def momentum(days: int) -> np.ndarray:
        out = np.zeros_like(log_levels)
        out[days:] = log_levels[days:] - log_levels[:-days]
        return out

    def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
        out = np.zeros_like(values)
        for i in range(window, len(values)):
            out[i] = values[i - window + 1:i + 1].std()
        return out

    def distance_from_average(window: int) -> np.ndarray:
        out = np.zeros_like(log_levels)
        for i in range(window, len(log_levels)):
            out[i] = log_levels[i] - log_levels[i - window + 1:i + 1].mean()
        return out

    columns = {
        "momentum_3": momentum(3),
        "momentum_7": momentum(7),
        "momentum_14": momentum(14),
        "momentum_30": momentum(30),
        "volatility_20": rolling_std(daily, 20),
        "vs_ma50": distance_from_average(50),
        "vs_ma100": distance_from_average(100),
        "rsi_14": (_rsi(levels) - 50) / 50,
    }
    names = list(columns)
    return np.column_stack([columns[n] for n in names]), names


def _standardise(X: np.ndarray):
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0

    def design(rows: np.ndarray) -> np.ndarray:
        rows = np.atleast_2d(rows)
        return np.column_stack([np.ones(len(rows)), (rows - mean) / std])

    return design


def fit_ridge(X: np.ndarray, y: np.ndarray, penalty: float = RIDGE_PENALTY):
    """Closed-form ridge on standardised features; returns a predict function."""
    design = _standardise(X)
    Z = design(X)
    identity = np.eye(Z.shape[1])
    identity[0, 0] = 0.0  # never penalise the intercept
    weights = np.linalg.solve(Z.T @ Z + penalty * identity, Z.T @ y)
    return lambda rows: design(rows) @ weights


def fit_logistic(X: np.ndarray, y: np.ndarray, penalty: float = LOGISTIC_PENALTY,
                 iterations: int = 25):
    """Penalised logistic regression by Newton steps; returns a predict_proba function.

    Newton converges in a handful of passes on data this small, and the ridge penalty
    keeps it from diverging when a feature separates the classes almost perfectly — which
    happens easily with 180 training rows and 8 correlated features.
    """
    design = _standardise(X)
    Z = design(X)
    weights = np.zeros(Z.shape[1])
    identity = np.eye(Z.shape[1])
    identity[0, 0] = 0.0

    for _ in range(iterations):
        eta = np.clip(Z @ weights, -30, 30)
        probability = 1.0 / (1.0 + np.exp(-eta))
        variance = np.clip(probability * (1 - probability), 1e-6, None)
        gradient = Z.T @ (y - probability) - penalty * (identity @ weights)
        hessian = Z.T @ (Z * variance[:, None]) + penalty * identity
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        weights = weights + step
        if np.max(np.abs(step)) < 1e-8:
            break

    def predict_proba(rows: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(design(rows) @ weights, -30, 30)))

    return predict_proba


def predict_now(levels, horizon_days: int = 3) -> dict:
    """Today's direction probability, trained on every window whose outcome is known.

    Returned with the base rate beside it on purpose: the probability only carries
    information insofar as it differs from how often the market simply went up anyway.
    """
    levels = np.asarray([float(x) for x in levels], dtype=float)
    X, _ = build_features(levels)
    log_levels = np.log(levels)
    n = len(levels)

    usable = n - horizon_days          # rows whose future has already happened
    if usable < 60:
        return {"available": False, "reason": "not enough history"}

    direction = (log_levels[horizon_days:] - log_levels[:usable] > 0).astype(float)
    predict_proba = fit_logistic(X[:usable], direction)
    probability_up = float(predict_proba(X[-1])[0])

    return {
        "available": True,
        "horizon_days": horizon_days,
        "probability_up": probability_up,
        "probability_down": 1.0 - probability_up,
        "base_rate_up": float(direction.mean()),
        "trained_on": int(usable),
    }
