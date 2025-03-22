import streamlit as st
import os
import re

# Set page configuration
st.set_page_config(
    page_title="March Madness Tournament Analysis",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define paths and constants
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)  # Get the directory of this script
ANALYSIS_DIR = os.path.join(
    BASE_DIR, "output/analysis"
)  # Base directory containing analysis files
BRACKET_DIR = os.path.join(BASE_DIR, "output")  # Directory containing bracket images
REGION_NAMES = {
    "W": "West Region",
    "X": "East Region",
    "Y": "South Region",
    "Z": "Midwest Region",
}


# Apply custom NCAA tournament theme
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #0066B2;
    }
    .stButton>button {
        background-color: #0066B2;
        color: white;
    }
    .stSelectbox label, .stRadio label {
        color: #0066B2;
        font-weight: bold;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Helper functions
def get_team_id_from_filename(filename):
    """Extract team ID from a filename like 'team_profile_1234' or 'team_profile_1234.md'"""
    match = re.search(r"team_profile_(\d+)(?:\.md)?$", filename)
    if match:
        return int(match.group(1))
    return None


def get_matchup_teams_from_filename(filename):
    """Extract team IDs from a filename like 'matchup_1234_vs_5678.md'"""
    match = re.search(r"matchup_(\d+)_vs_(\d+)\.md", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def get_region_from_filename(filename):
    """Extract region from a filename like 'region_W_summary.md'"""
    match = re.search(r"region_([A-Z])_summary\.md", filename)
    if match:
        return match.group(1)
    return None


def read_markdown_file(filepath):
    """Read a markdown file and return its contents"""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def extract_team_info(md_content):
    """Extract team name and seed from team profile markdown content"""
    # Try first format: "# West Region #03 Wisconsin Team Profile"
    match = re.search(
        r"#\s+([A-Z][a-z]+)\s+[A-Z][a-z]+\s+#?(\d{2})\s+([^#]+?)\s+Team Profile",
        md_content,
    )
    if match:
        region_name = f"{match.group(1)} Region"
        seed = match.group(2)
        team_name = match.group(3).strip()
        # Include the region name with the team name
        team_name_with_region = f"{region_name}: {team_name}"
        return (
            seed,
            team_name_with_region,
        )  # Return numeric seed and team name with region

    # Try second format: "# Z06 Missouri Team Profile"
    match = re.search(r"#\s+([W-Z])(\d{2})\s+([^#]+?)\s+Team Profile", md_content)
    if match:
        region_code = match.group(1)
        seed = match.group(2)
        team_name = match.group(3).strip()
        # Map region code to region name using REGION_NAMES
        region_name = REGION_NAMES.get(region_code, "Unknown Region")
        # Include the region name with the team name
        team_name_with_region = f"{region_name}: {team_name}"
        return (
            seed,
            team_name_with_region,
        )  # Return numeric seed and team name with region

    # Try to extract from Seed field in the document
    match = re.search(r"\*\*Seed:\*\*\s+([W-Z])(\d{2})", md_content)
    if match:
        region_code = match.group(1)
        seed = match.group(2)

        # Try to get team name from title
        title_match = re.search(
            r"#\s+[^#]+?([A-Za-z\s]+?)(?:Team Profile|$)", md_content
        )
        team_name = title_match.group(1).strip() if title_match else f"Team {seed}"

        # Map region code to region name
        region_name = REGION_NAMES.get(region_code, "Unknown Region")
        team_name_with_region = f"{region_name}: {team_name}"
        return seed, team_name_with_region

    return None, None


def extract_matchup_info(md_content):
    """Extract team names from matchup analysis markdown content"""
    # Look for the header pattern like "# West Region: #03 Wisconsin vs #14 Montana"
    match = re.search(
        r"#\s+(?:[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s*:)?)?\s*#?(\d+)\s+([^#]+?)\s+vs\s+#?(\d+)\s+([^#]+)",
        md_content,
    )
    if match:
        seed1 = match.group(1)
        team1 = match.group(2).strip()
        seed2 = match.group(3)
        team2 = match.group(4).strip()
        return f"{seed1} {team1}", f"{seed2} {team2}"
    return None, None


def extract_upset_probability(md_content):
    """Extract upset probability from matchup content for filtering"""
    lower_seed_win_pattern = r"(\d+\.?\d*)% chance of pulling the upset"
    match = re.search(lower_seed_win_pattern, md_content)
    if match:
        return float(match.group(1))
    return 0


# Load data
@st.cache_data
def load_analysis_files():
    """Load and categorize all analysis files"""
    data = {
        "team_profiles": {},  # team_id -> (filepath, seed, team_name)
        "matchups": {},  # (team1_id, team2_id) -> (filepath, team1_display, team2_display, upset_prob)
        "regions": {},  # region -> filepath
        "executive": None,  # filepath to executive summary
        "brackets": [],  # list of bracket image filepaths
        "team_id_to_info": {},  # team_id -> (seed, team_name) mapping for consistent reference
    }

    # Ensure ANALYSIS_DIR exists
    if not os.path.exists(ANALYSIS_DIR):
        st.error(f"Analysis directory '{ANALYSIS_DIR}' does not exist.")
        return data

    # First pass: collect all team profiles to build a complete team_id_to_info mapping
    for root, _, files in os.walk(ANALYSIS_DIR):
        for file in files:
            if file.endswith(".md"):
                filepath = os.path.join(root, file)

                # Team profile
                team_id = get_team_id_from_filename(file)
                if team_id is not None:
                    try:
                        md_content = read_markdown_file(filepath)
                        seed, team_name = extract_team_info(md_content)
                        if seed and team_name:
                            data["team_profiles"][team_id] = (filepath, seed, team_name)
                            # Store seed and team name for reference by team_id
                            data["team_id_to_info"][team_id] = (seed, team_name)
                    except Exception as e:
                        st.error(f"Error processing team profile {file}: {e}")
                    continue  # Skip further processing for this file

    # Second pass: process matchups, regions, etc.
    for root, _, files in os.walk(ANALYSIS_DIR):
        for file in files:
            if file.endswith(".md"):
                filepath = os.path.join(root, file)

                # Skip team profiles (already processed)
                if get_team_id_from_filename(file) is not None:
                    continue

                # Matchup analysis
                team1_id, team2_id = get_matchup_teams_from_filename(file)
                if team1_id is not None and team2_id is not None:
                    try:
                        md_content = read_markdown_file(filepath)
                        # Use team IDs to get team info
                        team1_info = data["team_id_to_info"].get(team1_id)
                        team2_info = data["team_id_to_info"].get(team2_id)

                        # If we have team info from the ID mapping, use it
                        if team1_info and team2_info:
                            team1_display = f"{team1_info[0]} {team1_info[1]}"
                            team2_display = f"{team2_info[0]} {team2_info[1]}"

                            # Include region name in team displays
                            if len(team1_info[0]) > 0:
                                region_code = team1_info[0][
                                    0
                                ]  # Extract region from seed
                                region_name = REGION_NAMES.get(region_code, "Unknown")
                            else:
                                region_name = "Unknown Region"
                        else:
                            # Fallback to extracting from markdown content
                            team1_display, team2_display = extract_matchup_info(
                                md_content
                            )

                            # For extracted matchups, use the first line to get region
                            first_line = (
                                md_content.split("\n")[0]
                                if "\n" in md_content
                                else md_content
                            )
                            region_match = re.search(
                                r"#\s+([A-Z][a-z]+)\s+[A-Z][a-z]+", first_line
                            )
                            if region_match:
                                region_name = f"{region_match.group(1)} Region"
                            else:
                                region_name = "Unknown Region"
                        # Clean up the extracted displays to avoid any Unknown prefixes

                        # Parse the first line of the markdown to get the correct header format
                        first_line = (
                            md_content.split("\n")[0]
                            if "\n" in md_content
                            else md_content
                        )

                        # Try to extract directly from the title line for a clean format
                        title_match = re.search(
                            r"#\s+([A-Z][a-z]+)\s+[A-Z][a-z]+(?:\s*:)?\s*#?(\d+)\s+([A-Za-z\s]+)\s+vs\s+#?(\d+)\s+([A-Za-z\s]+)",
                            first_line,
                        )
                        if title_match:
                            region_name = f"{title_match.group(1)} Region"
                            seed1 = title_match.group(2)
                            team1_name = title_match.group(3).strip()
                            seed2 = title_match.group(4)
                            team2_name = title_match.group(5).strip()

                            team1_display = f"{seed1} {region_name}: {team1_name}"
                            team2_display = f"{seed2} {region_name}: {team2_name}"
                        else:
                            # Fallback to the existing team displays but clean them up
                            team1_parts = team1_display.split()
                            team2_parts = team2_display.split()

                            if len(team1_parts) > 0:
                                seed1 = team1_parts[0]
                                team1_name = " ".join(team1_parts[1:])
                                team1_display = f"{seed1} {region_name}: {team1_name}"

                            if len(team2_parts) > 0:
                                seed2 = team2_parts[0]
                                team2_name = " ".join(team2_parts[1:])
                                team2_display = f"{seed2} {region_name}: {team2_name}"

                        upset_prob = extract_upset_probability(md_content)

                        if team1_display and team2_display:
                            data["matchups"][(team1_id, team2_id)] = (
                                filepath,
                                team1_display,
                                team2_display,
                                upset_prob,
                            )
                    except Exception as e:
                        st.error(f"Error processing matchup {file}: {e}")
                    continue

                # Region summary
                region = get_region_from_filename(file)
                if region is not None:
                    data["regions"][region] = filepath
                    continue

                # Executive summary
                if file == "tournament_executive_summary.md":
                    data["executive"] = filepath
                    continue

    # Process bracket images
    for root, _, files in os.walk(BRACKET_DIR):
        for file in files:
            if file.endswith(".png"):
                data["brackets"].append((os.path.join(root, file), file))

    return data


# App layout and functionality
def main():
    st.title("🏀 March Madness Tournament Analysis Dashboard")

    # Load data
    data = load_analysis_files()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    nav_option = st.sidebar.radio(
        "Select a view:",
        [
            "Tournament Overview",
            "Bracket Visualization",
            "Region Analysis",
            "Team Profiles",
            "Matchup Analysis",
            "Search",
        ],
    )

    # Display content based on navigation
    if nav_option == "Tournament Overview":
        display_tournament_overview(data)
    elif nav_option == "Bracket Visualization":
        display_bracket_visualization(data)
    elif nav_option == "Region Analysis":
        display_region_analysis(data)
    elif nav_option == "Team Profiles":
        display_team_profiles(data)
    elif nav_option == "Matchup Analysis":
        display_matchup_analysis(data)
    elif nav_option == "Search":
        display_search(data)


def display_tournament_overview(data):
    st.header("Tournament Executive Summary")

    if data["executive"]:
        md_content = read_markdown_file(data["executive"])
        st.markdown(md_content)
    else:
        st.warning(
            "Executive summary not found! Please make sure the analysis files are generated correctly."
        )

    st.markdown("---")
    st.markdown("### How to Use This Dashboard")
    st.markdown(
        """
        - **Tournament Overview**: View the executive summary with top contenders and predictions
        - **Region Analysis**: View analyses for each region (West, East, South, Midwest)
        - **Team Profiles**: View detailed profiles for individual teams
        - **Matchup Analysis**: View analyses for specific matchups including upset potential
        - **Search**: Find specific teams or matchups by keyword
    """
    )


def display_region_analysis(data):
    st.header("Region Analysis")

    if not data["regions"]:
        st.warning(
            "No region analyses found! Please make sure the analysis files are generated correctly."
        )
        return

    # Region selector
    regions = sorted(data["regions"].keys())

    region_options = [f"{REGION_NAMES.get(r, 'Unknown Region')}" for r in regions]
    region_to_code = {REGION_NAMES.get(r, "Unknown Region"): r for r in regions}
    selected_region_name = st.selectbox("Select a region:", region_options)
    selected_region = region_to_code[selected_region_name]

    # Display region analysis
    if selected_region in data["regions"]:
        md_content = read_markdown_file(data["regions"][selected_region])
        st.markdown(md_content)
    else:
        st.warning(f"Analysis for region {selected_region} not found!")


def display_team_profiles(data):
    st.header("Team Profiles")

    if not data["team_profiles"]:
        st.warning(
            "No team profiles found! Please make sure the analysis files are generated correctly."
        )
        return

    # Create a sorted list of teams by seed
    teams = []
    for team_id, (_, seed, team_name) in data["team_profiles"].items():
        teams.append((team_id, int(seed), f"{seed}. {team_name}"))

    teams_sorted = sorted(teams, key=lambda x: (x[1], x[2]))
    team_options = [team[2] for team in teams_sorted]
    team_id_map = {team[2]: team[0] for team in teams_sorted}

    # Team selector with seed-based categories
    seed_categories = {
        "Top Seeds (1-4)": [t for t in team_options if int(t.split(".")[0]) <= 4],
        "Middle Seeds (5-12)": [
            t for t in team_options if 5 <= int(t.split(".")[0]) <= 12
        ],
        "Lower Seeds (13-16)": [t for t in team_options if int(t.split(".")[0]) >= 13],
    }

    # Two-step selection process
    seed_category = st.selectbox("Seed range:", list(seed_categories.keys()))
    selected_team_display = st.selectbox(
        "Select a team:", seed_categories[seed_category]
    )
    selected_team_id = team_id_map[selected_team_display]

    # Display team profile
    if selected_team_id in data["team_profiles"]:
        filepath, _, _ = data["team_profiles"][selected_team_id]
        md_content = read_markdown_file(filepath)
        st.markdown(md_content)
    else:
        st.warning(f"Profile for {selected_team_display} not found!")


def display_matchup_analysis(data):
    st.header("Matchup Analysis")

    if not data["matchups"]:
        st.warning(
            "No matchup analyses found! Please make sure the analysis files are generated correctly."
        )
        return

    # Helper function to extract seed number from display string
    def get_seed_number(display):
        match = re.search(r"^#?(\d+)", display)
        return int(match.group(1)) if match else 999

    # Helper function to extract region from display string
    def get_region(display):
        for region_code, region_name in REGION_NAMES.items():
            if region_name in display:
                return region_code
        return "Z"  # Default to last region if not found

    # Sort matchups by region and seed numbers
    def matchup_sort_key(item):
        _, (_, team1_display, team2_display, _) = item
        region = get_region(team1_display)
        seed1 = get_seed_number(team1_display)
        seed2 = get_seed_number(team2_display)
        return (region, min(seed1, seed2), max(seed1, seed2))

    matchups_sorted = sorted(data["matchups"].items(), key=matchup_sort_key)

    # Create select box options with clean display format
    matchup_options = []
    matchup_file_map = {}

    for (team1_id, team2_id), (
        filepath,
        team1_display,
        team2_display,
        upset,
    ) in matchups_sorted:
        # Extract region name from either team display (they should be the same)
        region_name = None
        for r_name in REGION_NAMES.values():
            if r_name in team1_display:
                region_name = r_name
                break

        # Clean up team displays
        team1_parts = re.match(r".*?(\d+[^:]+)$", team1_display)
        team2_parts = re.match(r".*?(\d+[^:]+)$", team2_display)

        clean_team1 = team1_parts.group(1).strip() if team1_parts else team1_display
        clean_team2 = team2_parts.group(1).strip() if team2_parts else team2_display

        # Format the display with region
        region_prefix = f"{region_name}: " if region_name else ""
        matchup_display = f"{region_prefix}{clean_team1} vs {clean_team2}"

        matchup_options.append(matchup_display)
        matchup_file_map[matchup_display] = filepath

    # Select box for matchups
    selected_matchup_display = st.selectbox("Select a matchup:", matchup_options)

    # Display matchup analysis
    if selected_matchup_display in matchup_file_map:
        filepath = matchup_file_map[selected_matchup_display]
        md_content = read_markdown_file(filepath)
        st.markdown(md_content)
    else:
        st.warning(f"Analysis for {selected_matchup_display} not found!")


def display_bracket_visualization(data):
    """Display the tournament bracket images"""
    st.header("Tournament Bracket Visualization")

    if not data["brackets"]:
        st.warning(
            "No bracket visualizations found! Please make sure the bracket image files are in the output directory."
        )
        return

    # Filter brackets to specifically find the requested ones
    probability_bracket = None
    betting_odds_bracket = None
    other_brackets = []

    for filepath, filename in data["brackets"]:
        if "2025_mens_bracket_prob.png" in filename:
            probability_bracket = filepath
        elif "2025_mens_bracket.png" in filename:
            betting_odds_bracket = filepath
        else:
            other_brackets.append((filepath, filename))

    # Tab selection for different bracket types
    bracket_type = st.radio(
        "Select bracket type:",
        ["Win Probability", "Betting Odds"]
        + (["Historical Brackets"] if other_brackets else []),
    )

    if bracket_type == "Win Probability" and probability_bracket:
        st.image(
            probability_bracket,
            caption="2025 Men's Tournament Bracket (Win Probability)",
            use_container_width=True,
        )
    elif bracket_type == "Betting Odds" and betting_odds_bracket:
        st.image(
            betting_odds_bracket,
            caption="2025 Men's Tournament Bracket (Betting Odds)",
            use_container_width=True,
        )
    elif bracket_type == "Historical Brackets" and other_brackets:
        for filepath, filename in other_brackets:
            st.image(filepath, caption=filename, use_container_width=True)
    else:
        st.warning(f"Selected bracket type ({bracket_type}) not found!")

    st.markdown("---")
    st.subheader("Understanding the Bracket")
    st.markdown(
        """
    This bracket visualization shows the tournament structure with predictions:
    
    - **Win Probability Version**: Shows the % chance each team has to win their matchup
    - **Betting Odds Version**: Shows the betting line (spread) for each matchup
    
    The bracket is read from the outside edges toward the center, with winners advancing to the next round.
    """
    )


def display_search(data):
    st.header("Search Teams and Matchups")

    search_term = st.text_input("Enter a team name or keyword:", "")

    if search_term:
        search_term = search_term.lower()

        # Search team profiles
        st.subheader("Team Profiles")
        team_results = []

        for team_id, (filepath, seed, team_name) in data["team_profiles"].items():
            if search_term in team_name.lower():
                team_results.append((team_id, f"{seed}. {team_name}", filepath))

        if team_results:
            for team_id, display_name, filepath in sorted(
                team_results, key=lambda x: x[1]
            ):
                if st.button(f"View: {display_name}", key=f"team_{team_id}"):
                    st.markdown(read_markdown_file(filepath))
        else:
            st.info("No team profiles match your search.")

        # Search matchups
        st.subheader("Matchups")
        matchup_results = []

        for (team1_id, team2_id), (filepath, team1_display, team2_display, _) in data[
            "matchups"
        ].items():
            matchup_display = f"{team1_display} vs {team2_display}"
            if search_term in matchup_display.lower():
                matchup_results.append((team1_id, team2_id, matchup_display, filepath))

        if matchup_results:
            for team1_id, team2_id, display, filepath in sorted(
                matchup_results, key=lambda x: x[2]
            ):
                if st.button(f"View: {display}", key=f"matchup_{team1_id}_{team2_id}"):
                    st.markdown(read_markdown_file(filepath))
        else:
            st.info("No matchups match your search.")
    else:
        st.info("Enter a search term to find team profiles and matchups.")


if __name__ == "__main__":
    main()
