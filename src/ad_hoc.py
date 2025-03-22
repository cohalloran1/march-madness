import pandas as pd
import math


def get_predictions():
    return pd.read_csv("./output/submission_2025.csv")


def get_teams():
    return pd.read_csv("../data/2025/MTeams.csv")


def get_matchups():
    return pd.read_csv("./tourney_updates/matchups.csv")


def win_probability_to_spread(
    win_probability, std_dev=11.0, calibration=1.8, tournament_mode=True
):
    """
    Convert a win probability to an implied point spread for basketball.

    Parameters:
    win_probability (float): Probability that Team 1 will win (between 0 and 1)
    std_dev (float): Standard deviation of scoring margin (default 11 for college basketball)
    calibration (float): Calibration factor for the conversion (lower = larger spreads)
    tournament_mode (bool): Enable special handling for tournament mismatches

    Returns:
    float: Implied point spread (negative means Team 1 is favored)
    """
    # Ensure win probability is within valid range
    win_probability = min(max(win_probability, 0.01), 0.99)

    # Convert win probability to a spread using the logit function
    logit = math.log(win_probability / (1 - win_probability))

    # Base calculation
    point_spread = -logit * std_dev / calibration

    # Special handling for tournament mode extreme mismatches
    if tournament_mode and win_probability > 0.85:
        # Apply additional scaling for very high probabilities
        # This helps match the observed spreads in tournament games
        extra_factor = (win_probability - 0.85) * 2.5
        point_spread = point_spread * (1 + extra_factor)

    return point_spread


def main():
    predictions = get_predictions()
    teams = get_teams()
    matchups = get_matchups()

    # Add TeamIDs to matchups
    matchups = matchups.merge(
        teams, left_on="Team1", right_on="TeamName", how="left"
    ).rename(columns={"TeamID": "Team1ID"})
    matchups = matchups.merge(
        teams, left_on="Team2", right_on="TeamName", how="left"
    ).rename(columns={"TeamID": "Team2ID"})

    # Create GameID with lower TeamID first
    matchups["GameID"] = (
        "2025_"
        + matchups[["Team1ID", "Team2ID"]].min(axis=1).astype(str)
        + "_"
        + matchups[["Team1ID", "Team2ID"]].max(axis=1).astype(str)
    )

    # No need for GameID_rev since we're ensuring consistent order
    matchups = matchups.merge(predictions, left_on="GameID", right_on="ID", how="left")

    # Calculate spread using win_probability_to_spread and round to nearest 0.5
    matchups["Spread"] = (
        matchups["Pred"]
        .apply(win_probability_to_spread)
        .round(1)
        .apply(lambda x: round(x * 2) / 2)
    )

    # Format Pred as percentage and write output CSV
    matchups["Pred"] = (matchups["Pred"] * 100).round(1).astype(str) + "%"
    matchups[["Team1", "Team2", "Pred", "Spread"]].to_csv(
        "./output/matchup_predictions.csv", index=False
    )


if __name__ == "__main__":
    main()
