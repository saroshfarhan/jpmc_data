"""
Natural Gas Price Estimator
----------------------------
Estimates the purchase price of natural gas for any date in the past
or up to one year into the future using SARIMA(1,1,1)(0,1,1,12).

Usage:
    python task1.py <date>

Examples:
    python task1.py "2022-06-30"
    python task1.py "2025-08-31"
    python task1.py "10/31/20"
"""

import sys
import warnings
import pandas as pd
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).parent.parent / "data" / "Nat_Gas.csv"

# Best model parameters from experiments
SARIMA_ORDER          = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (0, 1, 1, 12)


def load_and_prepare_data(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df["date_col"] = pd.to_datetime(df["Dates"], format="%m/%d/%y")
    df = df.sort_values("date_col").reset_index(drop=True)
    ts = df.set_index("date_col")["Prices"]
    ts.index = pd.DatetimeIndex(ts.index, freq="ME")
    return ts


def fit_model(ts: pd.Series) -> object:
    model = SARIMAX(ts, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER)
    return model.fit(disp=False)


def snap_to_month_end(date: pd.Timestamp) -> pd.Timestamp:
    """Snap any date to its month-end (e.g. 2022-06-15 -> 2022-06-30)."""
    return date + pd.offsets.MonthEnd(0)


def get_price_estimate(date_input: str, data_path: Path = DATA_PATH) -> dict:
    """
    Estimate the natural gas price for a given date.

    Parameters
    ----------
    date_input : str
        Date string in any common format (e.g. '2022-06-30', '06/30/22').

    Returns
    -------
    dict with keys:
        date      - the month-end date used for estimation
        price     - estimated price
        lower_95  - lower 95% confidence bound (future dates only)
        upper_95  - upper 95% confidence bound (future dates only)
        type      - 'historical' or 'forecast'
    """
    # Parse input date
    try:
        query_date = pd.to_datetime(date_input)
    except Exception:
        raise ValueError(f"Could not parse date: '{date_input}'. Try format YYYY-MM-DD.")

    query_month_end = snap_to_month_end(query_date)

    # Load data and fit model
    ts     = load_and_prepare_data(data_path)
    fitted = fit_model(ts)

    first_date = ts.index[0]
    last_date  = ts.index[-1]
    max_future = last_date + pd.DateOffset(months=12)

    # Validate date range
    if query_month_end < first_date:
        raise ValueError(
            f"Date {query_month_end.date()} is before the earliest data point "
            f"({first_date.date()}). Cannot estimate."
        )

    if query_month_end > max_future:
        raise ValueError(
            f"Date {query_month_end.date()} is more than 12 months beyond the last "
            f"data point ({last_date.date()}). Forecasts beyond 1 year are unreliable."
        )

    # Historical: return in-sample fitted value
    if query_month_end <= last_date:
        in_sample = fitted.fittedvalues

        if query_month_end in in_sample.index:
            price = in_sample[query_month_end]
        else:
            nearest = min(in_sample.index, key=lambda d: abs(d - query_month_end))
            price = in_sample[nearest]
            query_month_end = nearest

        return {
            "date":     query_month_end.date(),
            "price":    round(float(price), 4),
            "lower_95": None,
            "upper_95": None,
            "type":     "historical",
        }

    # Future: forecast n months ahead
    months_ahead = (query_month_end.year - last_date.year) * 12 + \
                   (query_month_end.month - last_date.month)

    forecast_obj  = fitted.get_forecast(steps=months_ahead)
    forecast_mean = forecast_obj.predicted_mean
    forecast_ci   = forecast_obj.conf_int()

    price    = float(forecast_mean.iloc[-1])
    lower_95 = float(forecast_ci.iloc[-1, 0])
    upper_95 = float(forecast_ci.iloc[-1, 1])

    return {
        "date":     query_month_end.date(),
        "price":    round(price, 4),
        "lower_95": round(lower_95, 4),
        "upper_95": round(upper_95, 4),
        "type":     "forecast",
    }


def main():
    date_input = input("Enter a date: ")

    try:
        result = get_price_estimate(date_input)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Estimated price: ${result['price']:.4f}")


if __name__ == "__main__":
    main()
