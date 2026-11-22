"""Payoff functions for four stylised operator-contract archetypes.

These are textbook contract forms, NOT the terms of any real licence or concession. Every function
returns (operator, authority) and satisfies the accounting identity

    operator + authority == system surplus

where the system surplus is what the line earns before any payment between the two parties.
All functions work on floats or NumPy arrays (element-wise). Money is in S$ million per year.
"""
from __future__ import annotations

import numpy as np


def system_surplus(fare_rev, nonfare_rev, opex, capital_charge=0.0):
    """Fare and non-fare revenue less operating cost (and, for archetype D, the asset charge)."""
    return fare_rev + nonfare_rev - opex - capital_charge


def net_cost(fare_rev, nonfare_rev, opex, licence_charge, deferral_subsidy=0.0):
    """A. Net-cost concession: the operator keeps all revenue and pays a fixed licence charge.

    deferral_subsidy is a payment from the authority to the operator for fare increases that the
    formula allowed but the regulator deferred.
    """
    operator = fare_rev + nonfare_rev - opex - licence_charge + deferral_subsidy
    authority = licence_charge - deferral_subsidy
    return operator, authority


def incentive_payment(service_fee, reliability_score, rate, bonus_cap, penalty_cap):
    """Incentive (positive) or deduction (negative) as a share of the service fee.

    reliability_score is in standard deviations from target; the payment is linear in the score
    and clipped to [-penalty_cap, +bonus_cap] times the fee.
    """
    return service_fee * np.clip(rate * reliability_score, -penalty_cap, bonus_cap)


def gross_cost(fare_rev, nonfare_rev, opex, service_fee, incentive=0.0):
    """B. Gross-cost contract: the authority keeps all revenue and pays a service fee."""
    operator = service_fee + incentive - opex
    authority = fare_rev + nonfare_rev - service_fee - incentive
    return operator, authority


def shortfall_relief(fare_rev, projected_fare_rev, deadband, share):
    """Authority's contribution when fare revenue falls below projection by more than the deadband."""
    threshold = projected_fare_rev * (1.0 - deadband)
    return share * np.maximum(0.0, threshold - fare_rev)


def profit_share(pre_share_profit, revenue, band1, rate1, band2, rate2):
    """Extra licence charge on profit above band1 x revenue (rate1) and above band2 x revenue (rate2)."""
    lower = band1 * revenue
    upper = band2 * revenue
    tier1 = rate1 * np.maximum(0.0, np.minimum(pre_share_profit, upper) - lower)
    tier2 = rate2 * np.maximum(0.0, pre_share_profit - upper)
    return tier1 + tier2


def hybrid(fare_rev, nonfare_rev, opex, base_charge, projected_fare_rev, deadband, shortfall_share,
           band1, rate1, band2, rate2, deferral_subsidy=0.0):
    """C. Hybrid: net-cost base, plus revenue-shortfall sharing below and profit sharing above."""
    relief = shortfall_relief(fare_rev, projected_fare_rev, deadband, shortfall_share)
    revenue = fare_rev + nonfare_rev
    pre = revenue - opex - base_charge + relief + deferral_subsidy
    share = profit_share(pre, revenue, band1, rate1, band2, rate2)
    operator = pre - share
    authority = base_charge - relief - deferral_subsidy + share
    return operator, authority


def concession(fare_rev, nonfare_rev, opex, capital_charge, fixed_payment, revenue_share):
    """D. Long concession: the operator sets fares, finances operating assets and pays the grantors."""
    payment = fixed_payment + revenue_share * fare_rev
    operator = fare_rev + nonfare_rev - opex - capital_charge - payment
    authority = payment
    return operator, authority
