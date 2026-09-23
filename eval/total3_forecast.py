"""
Does a forecast model beat "tomorrow equals today" on TOTAL3?

A chart with a predicted line on it is easy; knowing whether the line is worth anything
is the hard part, and the honest answer is usually no. So this measures a model against
the naive baseline every price forecast has to clear — the random walk, which simply
predicts no change — and reports whichever result is true.

Method, and the reasons for it:

  * WALK-FORWARD, never a random split. Each prediction is made from an expanding window
    of the past only, and a training sample is used only once its own future is already
    realised. Shuffling market data leaks the future into the past and flatters a model
    into nonsense.
  * THE BASELINES ARE THE POINT. Random walk (no change) and momentum persistence (the
    last h days repeat) are both cheap, and a model that cannot beat them has no edge
    whatever its absolute error looks like.
  * DIRECTION IS GRADED AGAINST THE MAJORITY CLASS, not 50%. In a year that trended, a
    model that always says "down" can look 60% accurate while knowing nothing.
  * EFFECTIVE SAMPLE SIZE IS REPORTED. Predictions 30 days apart overlap almost entirely,
    so ~155 of them carry roughly 5 independent observations. Scores computed on
    overlapping windows look far more significant than they are.

What this cannot do: one year of history is one market regime, and the series is a
reconstruction whose constituents come from today's rankings. Neither is fixable with a
free data source, so a positive result here would still need multi-regime data before
anyone should believe it.

Usage:
    python -m eval.total3_forecast
"""

import json
import sys

import numpy as np

from config import ROOT_DIR
from agent.total3 import load_series

RESULTS = ROOT_DIR / "eval" / "total3_forecast.json"

HORIZONS = (7, 30)
INITIAL_TRAIN_DAYS = 180
RIDGE_PENALTY = 1.0

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def _rsi(levels: np.ndarray, window: int = 14) -> np.ndarray:
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
    """Features known at day t, using nothing from after it."""
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


def ridge_fit(X: np.ndarray, y: np.ndarray, penalty: float = RIDGE_PENALTY):
    """Closed-form ridge on standardised features; returns a predict function."""
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    Z = np.column_stack([np.ones(len(X)), (X - mean) / std])
    identity = np.eye(Z.shape[1])
    identity[0, 0] = 0.0  # never penalise the intercept
    weights = np.linalg.solve(Z.T @ Z + penalty * identity, Z.T @ y)

    def predict(rows: np.ndarray) -> np.ndarray:
        rows = np.atleast_2d(rows)
        return np.column_stack([np.ones(len(rows)), (rows - mean) / std]) @ weights

    return predict


def walk_forward(levels: np.ndarray, horizon: int) -> dict:
    """Predict the h-day-ahead log return at each step, training only on realised pasts."""
    X, feature_names = build_features(levels)
    log_levels = np.log(levels)
    n = len(levels)

    target = np.full(n, np.nan)
    target[:n - horizon] = log_levels[horizon:] - log_levels[:n - horizon]

    actuals, model_pred, momentum_pred = [], [], []
    for t in range(INITIAL_TRAIN_DAYS, n - horizon):
        # A sample may train the model only once its own future has already happened.
        usable = t - horizon
        if usable < 30:
            continue
        predict = ridge_fit(X[:usable], target[:usable])
        model_pred.append(float(predict(X[t])[0]))
        # Persistence: the move of the last h days simply continues.
        momentum_pred.append(float(log_levels[t] - log_levels[t - horizon]))
        actuals.append(float(target[t]))

    actuals = np.array(actuals)
    model_pred = np.array(model_pred)
    momentum_pred = np.array(momentum_pred)
    naive_pred = np.zeros_like(actuals)  # random walk: no change

    def score(prediction: np.ndarray) -> dict:
        error = prediction - actuals
        up = actuals > 0
        # A zero prediction has no direction, so it can't be graded on one.
        directional = (
            float(np.mean((prediction > 0) == up)) if np.any(prediction != 0) else None
        )
        return {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "directional_accuracy": directional,
        }

    results = {
        "horizon_days": horizon,
        "predictions": len(actuals),
        "independent_windows": round(len(actuals) / horizon, 1),
        "share_up": float(np.mean(actuals > 0)),
        "majority_class_rate": float(max(np.mean(actuals > 0), np.mean(actuals <= 0))),
        "naive_random_walk": score(naive_pred),
        "momentum_persistence": score(momentum_pred),
        "ridge_model": score(model_pred),
        "features": feature_names,
    }
    naive_mse = results["naive_random_walk"]["rmse"] ** 2
    for name in ("ridge_model", "momentum_persistence"):
        mse = results[name]["rmse"] ** 2
        results[name]["skill_vs_naive"] = float(1 - mse / naive_mse) if naive_mse else None
    return results


def verdict(result: dict) -> tuple[bool, str]:
    """Did the model earn the right to be shown as a forecast?"""
    model = result["ridge_model"]
    skill = model.get("skill_vs_naive")
    direction = model.get("directional_accuracy")
    majority = result["majority_class_rate"]

    if skill is None or skill <= 0:
        return False, (f"No. The model's squared error is {'worse than' if skill is not None and skill < 0 else 'no better than'} "
                       f"the random walk (skill {skill:+.3f}).")
    if direction is None or direction <= majority:
        return False, (f"No. It beats the random walk on error (skill {skill:+.3f}) but its "
                       f"direction calls ({direction:.1%}) don't beat always guessing the "
                       f"majority ({majority:.1%}).")
    if result["independent_windows"] < 10:
        return False, (f"Not established. Skill {skill:+.3f} and direction {direction:.1%} "
                       f"look positive, but on only {result['independent_windows']} "
                       "independent windows that is noise, not evidence.")
    return True, (f"Yes: skill {skill:+.3f} over the random walk and {direction:.1%} "
                  f"directional accuracy against a {majority:.1%} majority, across "
                  f"{result['independent_windows']} independent windows.")


def main() -> None:
    data = load_series()
    series = data["series"]
    levels = np.array([point["total3"] for point in series], dtype=float)
    print(f"TOTAL3 proxy: {len(levels)} days, "
          f"{series[0]['date']} to {series[-1]['date']}")
    print(f"built from {data['n_coins_used']} coins, {data['coverage_pct']:.1f}% coverage, "
          f"{data['stable_share_pct']:.1f}% stablecoins")
    print(f"level {levels[0]/1e12:.3f}T -> {levels[-1]/1e12:.3f}T "
          f"({(levels[-1]/levels[0]-1)*100:+.1f}% over the year)")

    payload = {"built_at": data["built_at"], "horizons": {}}
    for horizon in HORIZONS:
        result = walk_forward(levels, horizon)
        beat, explanation = verdict(result)
        result["beats_baseline"] = beat
        result["verdict"] = explanation
        payload["horizons"][str(horizon)] = result

        print(f"\n{'=' * 70}\n{horizon}-day horizon\n{'=' * 70}")
        print(f"{result['predictions']} predictions, but only "
              f"{result['independent_windows']} independent windows "
              f"({result['share_up']:.0%} of them were up)")
        print(f"{'method':24} {'MAE':>8} {'RMSE':>8} {'skill':>8} {'direction':>10}")
        for name in ("naive_random_walk", "momentum_persistence", "ridge_model"):
            s = result[name]
            skill = s.get("skill_vs_naive")
            direction = s.get("directional_accuracy")
            print(f"{name:24} {s['mae']:8.4f} {s['rmse']:8.4f} "
                  f"{(f'{skill:+.3f}' if skill is not None else '  —'):>8} "
                  f"{(f'{direction:.1%}' if direction is not None else '  —'):>10}")
        print(f"\nBeats the baseline? {explanation}")

    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved to {RESULTS.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
