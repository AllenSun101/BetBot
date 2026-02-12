from typing import Tuple

def american_to_decimal(odds: float) -> float:
    """
    Convert American odds to decimal odds.
    """
    if odds > 0:
        return 1 + odds / 100
    else:
        return 1 + 100 / abs(odds)


def implied_probability(odds: float) -> float:
    """
    Convert American odds to implied probability.
    """
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def hedge_stakes(
    odds_a: float,
    odds_b: float,
    total_stake: float
) -> Tuple[float, float]:
    """
    Compute optimal hedge stakes that equalize payouts.
    """
    d_a = american_to_decimal(odds_a)
    d_b = american_to_decimal(odds_b)

    inv_sum = (1 / d_a) + (1 / d_b)

    stake_a = (total_stake / d_a) / inv_sum
    stake_b = (total_stake / d_b) / inv_sum

    return stake_a, stake_b


def payoff(
    stake_a: float,
    stake_b: float,
    odds_a: float,
    odds_b: float
) -> Tuple[float, float]:
    """
    Return payout if A wins, payout if B wins.
    """
    d_a = american_to_decimal(odds_a)
    d_b = american_to_decimal(odds_b)

    payout_a = stake_a * d_a
    payout_b = stake_b * d_b

    return payout_a, payout_b


def summarize(
    odds_a: float,
    odds_b: float,
    total_stake: float,
    stake_a_override: float | None = None,
    stake_b_override: float | None = None,
):
    print("=== Odds ===")
    print(f"A: {odds_a}, B: {odds_b}")

    p_a = implied_probability(odds_a)
    p_b = implied_probability(odds_b)

    print("\n=== Implied Probabilities ===")
    print(f"A: {p_a:.4%}")
    print(f"B: {p_b:.4%}")
    print(f"Sum: {(p_a + p_b):.4%}")

    stake_a, stake_b = hedge_stakes(odds_a, odds_b, total_stake)

    if stake_a_override is not None:
        stake_a = stake_a_override
    if stake_b_override is not None:
        stake_b = stake_b_override

    payout_a, payout_b = payoff(stake_a, stake_b, odds_a, odds_b)

    print("\n=== Stakes ===")
    print(f"A stake: {stake_a:.2f}")
    print(f"B stake: {stake_b:.2f}")
    print(f"Total:   {stake_a + stake_b:.2f}")

    print("\n=== Payoffs ===")
    print(f"If A wins: {payout_a:.2f} (profit {payout_a - (stake_a + stake_b):.2f})")
    print(f"If B wins: {payout_b:.2f} (profit {payout_b - (stake_a + stake_b):.2f})")


if __name__ == "__main__":
    odds_A = +185
    odds_B = +190
    total = 1000

    print("\n--- Optimal hedge ---")
    summarize(odds_A, odds_B, total)

    print("\n--- Rounded stakes (experiment) ---")
    summarize(
        odds_A,
        odds_B,
        total,
        stake_a_override=454,
        stake_b_override=546
    )
