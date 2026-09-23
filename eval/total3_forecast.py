"""
Does any model beat the baselines on TOTAL3, over 3 days?

A chart with a predicted line on it is easy; knowing whether the line is worth anything
is the hard part, and the honest answer is usually no. So this measures two models
against the baselines they have to clear, and reports whichever result is true.

  * MAGNITUDE — ridge on the log return, against the random walk, which predicts no
    change. Skill is 1 - MSE(model)/MSE(random walk); negative means worse than assuming
    nothing happens.
  * DIRECTION — logistic on the sign, against the base rate: the share of past windows
    that were up. Graded by Brier score, not accuracy alone, because a probability is
    only useful if it is calibrated as well as pointed the right way. In a year that
    drifted up, always saying "up" scores well while knowing nothing.

Method, and the reasons for it:

  * WALK-FORWARD, never a random split. Each prediction uses an expanding window of the
    past only, and a training row is used only once its own future has already happened.
    Shuffling market data leaks the future backwards and flatters a model into nonsense.
  * EFFECTIVE SAMPLE SIZE IS REPORTED. Predictions h days apart overlap, so ~182 of them
    at a 3-day horizon carry about 60 independent observations — far fewer than the raw
    count suggests, and the reason 3 days is measurable where 30 was not.
  * CALIBRATION IS CHECKED. When the model says 60%, does it happen about 60% of the
    time? A confident model that is wrong at the same rate is worse than useless.

What this cannot do: one year of history is one market regime, and the series is a
reconstruction whose constituents come from today's rankings. A positive result here
would still need data across several regimes before anyone should believe it.

Usage:
    python -m eval.total3_forecast
"""

import json
import sys

import numpy as np

from config import ROOT_DIR
from agent.total3 import load_series
from agent.total3_model import build_features, fit_logistic, fit_ridge

RESULTS = ROOT_DIR / "eval" / "total3_forecast.json"

# 3 days first: it's the horizon the dashboard shows, and the only one with enough
# independent windows to mean much. 7 and 30 stay as context.
HORIZONS = (3, 7, 30)
INITIAL_TRAIN_DAYS = 180
CALIBRATION_BINS = 4
MIN_INDEPENDENT_WINDOWS = 10

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def _calibration(probabilities: np.ndarray, outcomes: np.ndarray) -> list[dict]:
    """Predicted probability against what actually happened, in bins."""
    bins = []
    edges = np.linspace(0.0, 1.0, CALIBRATION_BINS + 1)
    for low, high in zip(edges, edges[1:]):
        mask = (probabilities >= low) & (probabilities < high if high < 1.0 else probabilities <= 1.0)
        if not mask.any():
            continue
        bins.append({
            "range": f"{low:.0%}-{high:.0%}",
            "predicted": float(probabilities[mask].mean()),
            "actual": float(outcomes[mask].mean()),
            "n": int(mask.sum()),
        })
    return bins


def walk_forward(levels: np.ndarray, horizon: int) -> dict:
    """Predict magnitude and direction at each step, training only on realised pasts."""
    X, feature_names = build_features(levels)
    log_levels = np.log(levels)
    n = len(levels)

    target = np.full(n, np.nan)
    target[:n - horizon] = log_levels[horizon:] - log_levels[:n - horizon]

    actuals, ridge_pred, momentum_pred, probability_up, base_rates = [], [], [], [], []
    for t in range(INITIAL_TRAIN_DAYS, n - horizon):
        # A row may train the model only once its own future has already happened.
        usable = t - horizon
        if usable < 30:
            continue
        y_train = target[:usable]
        ridge_pred.append(float(fit_ridge(X[:usable], y_train)(X[t])[0]))

        direction_train = (y_train > 0).astype(float)
        probability_up.append(float(fit_logistic(X[:usable], direction_train)(X[t])[0]))
        base_rates.append(float(direction_train.mean()))

        momentum_pred.append(float(log_levels[t] - log_levels[t - horizon]))
        actuals.append(float(target[t]))

    actuals = np.array(actuals)
    ridge_pred = np.array(ridge_pred)
    momentum_pred = np.array(momentum_pred)
    probability_up = np.array(probability_up)
    base_rates = np.array(base_rates)
    went_up = (actuals > 0).astype(float)
    naive_pred = np.zeros_like(actuals)  # random walk: no change

    def magnitude(prediction: np.ndarray) -> dict:
        error = prediction - actuals
        directional = (
            float(np.mean((prediction > 0) == (actuals > 0)))
            if np.any(prediction != 0) else None
        )
        return {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "directional_accuracy": directional,
        }

    majority = float(max(went_up.mean(), 1 - went_up.mean()))
    model_brier = float(np.mean((probability_up - went_up) ** 2))
    # Climatology: the base rate known at the time, not one computed with hindsight.
    base_brier = float(np.mean((base_rates - went_up) ** 2))

    results = {
        "horizon_days": horizon,
        "predictions": len(actuals),
        "independent_windows": round(len(actuals) / horizon, 1),
        "share_up": float(went_up.mean()),
        "majority_class_rate": majority,
        "naive_random_walk": magnitude(naive_pred),
        "momentum_persistence": magnitude(momentum_pred),
        "ridge_model": magnitude(ridge_pred),
        "direction_model": {
            "brier": model_brier,
            "base_rate_brier": base_brier,
            "brier_skill": float(1 - model_brier / base_brier) if base_brier else None,
            "accuracy": float(np.mean((probability_up > 0.5) == (went_up > 0.5))),
            "mean_probability_up": float(probability_up.mean()),
            "calibration": _calibration(probability_up, went_up),
        },
        "features": feature_names,
    }

    naive_mse = results["naive_random_walk"]["rmse"] ** 2
    for name in ("ridge_model", "momentum_persistence"):
        mse = results[name]["rmse"] ** 2
        results[name]["skill_vs_naive"] = float(1 - mse / naive_mse) if naive_mse else None
    return results


def magnitude_verdict(result: dict) -> tuple[bool, str]:
    """Did the magnitude model earn the right to be drawn as a forecast line?"""
    skill = result["ridge_model"].get("skill_vs_naive")
    if skill is None:
        return False, "No. Skill against the random walk couldn't be computed."
    if skill <= 0:
        worse = "worse than" if skill < 0 else "no better than"
        return False, (f"No. Its squared error is {worse} predicting no change at all "
                       f"(skill {skill:+.3f}).")
    if result["independent_windows"] < MIN_INDEPENDENT_WINDOWS:
        return False, (f"Not established. Skill {skill:+.3f} looks positive, but on "
                       f"{result['independent_windows']} independent windows that is "
                       "noise, not evidence.")
    return True, (f"Yes: skill {skill:+.3f} over the random walk across "
                  f"{result['independent_windows']} independent windows.")


def direction_verdict(result: dict) -> tuple[bool, str]:
    """Is the direction probability better than just quoting the base rate?"""
    direction = result["direction_model"]
    skill = direction.get("brier_skill")
    accuracy, majority = direction["accuracy"], result["majority_class_rate"]
    if skill is None:
        return False, "No. Brier skill couldn't be computed."
    if skill <= 0:
        return False, (f"No. Its Brier score ({direction['brier']:.4f}) is no better than "
                       f"quoting the base rate ({direction['base_rate_brier']:.4f}), "
                       f"skill {skill:+.3f}.")
    if accuracy <= majority:
        return False, (f"No. Brier skill {skill:+.3f} is positive, but it calls direction "
                       f"{accuracy:.1%} of the time against {majority:.1%} for always "
                       "guessing the majority.")
    if result["independent_windows"] < MIN_INDEPENDENT_WINDOWS:
        return False, (f"Not established. Skill {skill:+.3f} and {accuracy:.1%} accuracy "
                       f"on {result['independent_windows']} independent windows is noise.")
    return True, (f"Yes: Brier skill {skill:+.3f} over the base rate and {accuracy:.1%} "
                  f"accuracy against a {majority:.1%} majority.")


def main() -> None:
    data = load_series()
    series = data["series"]
    levels = np.array([point["total3"] for point in series], dtype=float)
    print(f"TOTAL3 proxy: {len(levels)} days, {series[0]['date']} to {series[-1]['date']}")
    print(f"built from {data['n_coins_used']} coins, {data['coverage_pct']:.1f}% coverage, "
          f"{data['stable_share_pct']:.1f}% stablecoins")

    payload = {"built_at": data["built_at"], "primary_horizon": HORIZONS[0], "horizons": {}}
    for horizon in HORIZONS:
        result = walk_forward(levels, horizon)
        beats_magnitude, magnitude_why = magnitude_verdict(result)
        beats_direction, direction_why = direction_verdict(result)
        result["beats_baseline"] = beats_magnitude
        result["verdict"] = magnitude_why
        result["direction_model"]["beats_base_rate"] = beats_direction
        result["direction_model"]["verdict"] = direction_why
        payload["horizons"][str(horizon)] = result

        print(f"\n{'=' * 72}\n{horizon}-day horizon"
              f"{'   <- shown on the dashboard' if horizon == HORIZONS[0] else ''}\n{'=' * 72}")
        print(f"{result['predictions']} predictions, {result['independent_windows']} "
              f"independent windows, {result['share_up']:.0%} of them up")

        print(f"\n{'magnitude':24} {'MAE':>8} {'RMSE':>8} {'skill':>8} {'direction':>10}")
        for name in ("naive_random_walk", "momentum_persistence", "ridge_model"):
            s = result[name]
            skill, accuracy = s.get("skill_vs_naive"), s.get("directional_accuracy")
            print(f"{name:24} {s['mae']:8.4f} {s['rmse']:8.4f} "
                  f"{(f'{skill:+.3f}' if skill is not None else '  —'):>8} "
                  f"{(f'{accuracy:.1%}' if accuracy is not None else '  —'):>10}")
        print(f"  -> {magnitude_why}")

        d = result["direction_model"]
        print(f"\n{'direction':24} {'Brier':>8} {'vs base':>8} {'skill':>8} {'accuracy':>10}")
        print(f"{'logistic':24} {d['brier']:8.4f} {d['base_rate_brier']:8.4f} "
              f"{d['brier_skill']:+8.3f} {d['accuracy']:9.1%}")
        print(f"  -> {direction_why}")
        if d["calibration"]:
            print("  calibration (said -> happened):  " + "   ".join(
                f"{b['range']}: {b['predicted']:.0%}->{b['actual']:.0%} (n={b['n']})"
                for b in d["calibration"]))

    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved to {RESULTS.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
