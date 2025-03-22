#!/usr/bin/env python3

import os
import pandas as pd
from enhanced_bracket_analysis import (
    generate_enhanced_analysis,
    load_elo_data,
    load_feature_data,
    load_predictions,
)
from data_classes.processing import MarchMadnessMLModel
from data_classes.bracket import BracketSimulator


def load_team_mappings():
    """Load team name to ID mappings from Teams.csv"""
    teams_df = pd.read_csv("../data/2025/MTeams.csv")
    return {
        name.lower(): id for name, id in zip(teams_df["TeamName"], teams_df["TeamID"])
    }


def process_matchups(matchups_file, output_dir):
    """Process each matchup and generate analysis using pre-computed data"""
    # Load team name to ID mappings
    team_mappings = load_team_mappings()

    # Read matchups CSV
    matchups_df = pd.read_csv(matchups_file)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load pre-computed data
    elo_df = load_elo_data("M", 2025)
    feature_df = load_feature_data("M")
    predictions_df = load_predictions("M", 2025)

    if any(x is None for x in [elo_df, feature_df, predictions_df]):
        print("Error: Could not load required data files from output directory")
        return

    # Initialize empty model shells - these won't be used for predictions
    # but are needed for the API
    ml_model = MarchMadnessMLModel()
    simulator = BracketSimulator()

    # Process each matchup
    for _, row in matchups_df.iterrows():
        team1_name = row["Team1"].strip().lower()
        team2_name = row["Team2"].strip().lower()

        # Look up team IDs
        try:
            team1_id = team_mappings[team1_name]
            team2_id = team_mappings[team2_name]

            print(f"Generating analysis for {row['Team1']} vs {row['Team2']}")

            # Generate the matchup analysis using pre-computed data
            generate_enhanced_analysis(
                simulator=simulator,
                ml_model=ml_model,
                gender_code="M",
                year=2025,
                output_dir=output_dir,
                analysis_type="matchup",
                matchup_ids=(team1_id, team2_id),
            )

        except KeyError as e:
            print(f"Error: Could not find team ID for {e}")
            continue


def main():
    # Define input and output paths
    matchups_file = "./tourney_updates/matchups.csv"
    output_dir = "./output/matchup_analysis"

    print("Starting matchup analysis generation...")
    process_matchups(matchups_file, output_dir)
    print("Analysis generation complete!")


if __name__ == "__main__":
    main()
