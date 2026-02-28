"""
Gas Storage Contract Pricing Model
------------------------------------
Prices a natural gas storage contract given injection/withdrawal dates,
prices, rate, volume constraints, and storage costs.

Usage:
    python task2/task2.py
"""

import sys
from pathlib import Path
from datetime import date

sys.path.append(str(Path(__file__).parent.parent))
from task1.task1 import get_price_estimate


def price_contract(
    in_dates: list,
    in_prices: list,
    out_dates: list,
    out_prices: list,
    rate: float,
    storage_cost_rate: float,
    total_vol: float,
    injection_withdrawal_cost_rate: float,
) -> float:
    """
    Price a natural gas storage contract.

    Parameters
    ----------
    in_dates : list of date
        Dates on which gas is injected (purchased).
    in_prices : list of float
        Purchase prices on each injection date.
    out_dates : list of date
        Dates on which gas is withdrawn (sold).
    out_prices : list of float
        Sale prices on each withdrawal date.
    rate : float
        Volume (MMBtu) injected or withdrawn per event.
    storage_cost_rate : float
        Cost per MMBtu per day while gas is in storage.
    total_vol : float
        Maximum storage capacity (MMBtu).
    injection_withdrawal_cost_rate : float
        Cost per MMBtu for the physical act of injecting or withdrawing.

    Returns
    -------
    float
        Net contract value. Positive = profitable.
    """
    volume = 0.0
    buy_cost = 0.0
    cash_in = 0.0

    # Ensure dates are processed in sequence
    all_dates = sorted(set(in_dates + out_dates))

    for i in range(len(all_dates)):
        start_date = all_dates[i]

        if start_date in in_dates:
            # Inject on this date if there is room for a full injection
            if volume <= total_vol - rate:
                volume += rate

                # Cost to purchase gas
                buy_cost += rate * in_prices[in_dates.index(start_date)]

                # Injection operational cost
                injection_cost = rate * injection_withdrawal_cost_rate
                buy_cost += injection_cost

                print(f"Injected gas on {start_date} at a price of "
                      f"${in_prices[in_dates.index(start_date)]:.4f}")
            else:
                print(f"Injection is not possible on {start_date} — "
                      f"insufficient space in the storage facility")

        elif start_date in out_dates:
            # Withdraw on this date if there is enough gas stored
            if volume >= rate:
                volume -= rate

                # Revenue from selling gas
                cash_in += rate * out_prices[out_dates.index(start_date)]

                # Withdrawal operational cost
                withdrawal_cost = rate * injection_withdrawal_cost_rate
                buy_cost += withdrawal_cost

                print(f"Withdrew gas on {start_date} at a price of "
                      f"${out_prices[out_dates.index(start_date)]:.4f}")
            else:
                print(f"Withdrawal is not possible on {start_date} — "
                      f"insufficient gas in storage")

        # Accrue daily storage cost until the next event
        if i < len(all_dates) - 1:
            days = (all_dates[i + 1] - start_date).days
            buy_cost += volume * days * storage_cost_rate

    contract_value = cash_in - buy_cost
    return round(contract_value, 2)


if __name__ == "__main__":
    print("=" * 55)
    print("  Gas Storage Contract Pricing — Test Cases")
    print("=" * 55)

    # Test 1: Inject in summer (low price), withdraw in winter (high price) — profitable
    print("\nTest 1 — Inject Jun (cheap), withdraw Jan (expensive):")
    v1 = price_contract(
        in_dates=[date(2023, 6, 30)],
        in_prices=[get_price_estimate("2023-06-30")["price"]],
        out_dates=[date(2024, 1, 31)],
        out_prices=[get_price_estimate("2024-01-31")["price"]],
        rate=1_000,
        storage_cost_rate=0.005,
        total_vol=5_000,
        injection_withdrawal_cost_rate=0.01,
    )
    print(f"  Contract value: ${v1:,.2f}")

    # Test 2: Multiple inject/withdraw cycles
    print("\nTest 2 — Two summer injections, two winter withdrawals:")
    v2 = price_contract(
        in_dates=[date(2023, 6, 30), date(2023, 7, 31)],
        in_prices=[
            get_price_estimate("2023-06-30")["price"],
            get_price_estimate("2023-07-31")["price"],
        ],
        out_dates=[date(2023, 12, 31), date(2024, 1, 31)],
        out_prices=[
            get_price_estimate("2023-12-31")["price"],
            get_price_estimate("2024-01-31")["price"],
        ],
        rate=500,
        storage_cost_rate=0.01,
        total_vol=2_000,
        injection_withdrawal_cost_rate=0.01,
    )
    print(f"  Contract value: ${v2:,.2f}")

    # Test 3: Volume cap — 3 injections but max_vol only fits 2
    print("\nTest 3 — Three injections, max volume only fits two:")
    v3 = price_contract(
        in_dates=[date(2023, 5, 31), date(2023, 6, 30), date(2023, 7, 31)],
        in_prices=[
            get_price_estimate("2023-05-31")["price"],
            get_price_estimate("2023-06-30")["price"],
            get_price_estimate("2023-07-31")["price"],
        ],
        out_dates=[date(2024, 1, 31)],
        out_prices=[get_price_estimate("2024-01-31")["price"]],
        rate=1_000,
        storage_cost_rate=0.005,
        total_vol=2_000,
        injection_withdrawal_cost_rate=0.01,
    )
    print(f"  Contract value: ${v3:,.2f}")

    print("\n" + "=" * 55)
