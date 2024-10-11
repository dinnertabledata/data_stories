#########################################################################
# 0. Imports

# Core DTD
from util_core import *


#########################################################################
# 1. Data

DATA_PATH = "../../data/"

####################################
# Load data on medals, events, teams, and athletes
def load_base_datasets():

    #############
    # Load medals
    medal_df = pd.read_csv(f"{DATA_PATH}/paris_2024/medals.csv")
    # Add unique event key
    medal_df["event_key"] = medal_df.apply(lambda x: f"{x['discipline']}: {x['event']}", axis=1)

    #############
    # Load events
    event_df = pd.read_csv(f"{DATA_PATH}/paris_2024/events.csv")
    # Add unique event key
    event_df["event_key"] = event_df.apply(lambda x: f"{x['sport']}: {x['event']}", axis=1)

    #############
    # Load teams
    team_df = pd.read_csv(f"{DATA_PATH}/paris_2024/teams.csv")
    # Filter on current
    orig_len = len(team_df)
    team_df = team_df[team_df["current"]==True]
    new_len = len(team_df)
    print(f"Notice: Dropped {orig_len-new_len}/{orig_len} ({(orig_len-new_len)/orig_len*100:.1f}%) non-current team records.")
    # Confirm event fields are all non-null
    null_df = team_df[team_df["discipline"].isnull() | team_df["events"].isnull()]
    print(f"null_df: {len(null_df)}")
    # Add unique event key
    team_df["event_key"] = team_df.apply(lambda x: f"{x['discipline']}: {x['events']}", axis=1)

    #############
    # Load athletes
    athlete_df = pd.read_csv(f"{DATA_PATH}/paris_2024/athletes.csv")
    # Filter on current
    orig_len = len(athlete_df)
    athlete_df = athlete_df[athlete_df["current"]==True]
    new_len = len(athlete_df)
    print(f"Notice: Dropped {orig_len-new_len}/{orig_len} ({(orig_len-new_len)/orig_len*100:.1f}%) non-current team records.")
    # Convert disciplines and events to lists
    athlete_df["disciplines"] = athlete_df["disciplines"].apply(lambda x: ast.literal_eval(x))
    athlete_df["events"] = athlete_df["events"].apply(lambda x: ast.literal_eval(x))
    # Build list of combined disciplines+events, then use to explode
    athlete_df["event_key"] = athlete_df.apply(lambda x: list(itertools.product(x["disciplines"], x["events"])), axis=1)
    athlete_df["event_key"] = athlete_df["event_key"].apply(lambda x: [f"{d}: {e}" for d,e in x])
    athlete_df = athlete_df.explode("event_key")
    new_len = len(athlete_df)
    print(f"Exploded to {new_len} records, one per athlete per event entered.")
    # Filter on only valid events
    valid_e = event_df["event_key"].unique()
    orig_len = len(athlete_df)
    athlete_df = athlete_df[athlete_df["event_key"].isin(valid_e)]
    new_len = len(athlete_df)
    print(f"Notice: Dropped {orig_len-new_len}/{orig_len} ({(orig_len-new_len)/orig_len*100:.1f}%) invalid event records.")

    #############
    # Combine
    all_data = {
        "medal_df": medal_df,
        "event_df": event_df,
        "team_df": team_df,
        "athlete_df": athlete_df,
    }

    #############
    # Return
    return all_data

####################################
# Combine team and athlete data into a single "event entry" dataset
def build_combined_event_entries(team_df, athlete_df):

    #############
    # Drop records for athletes on teams (only want to count single entry)
    team_events = team_df["event_key"].unique()
    orig_len = len(athlete_df)
    single_athlete_df = athlete_df[~athlete_df["event_key"].isin(team_events)].copy()
    new_len = len(single_athlete_df)
    print(f"Notice: Dropped {orig_len-new_len}/{orig_len} ({(orig_len-new_len)/orig_len*100:.1f}%) team event records.")

    #############
    # Define columns we want to keep
    team_keep_cols = [
        'event_key',
        'team_gender',
        'country_code',
        'country'
    ]
    athlete_keep_cols = [
        'event_key',
        'gender',
        'country_code',
        'country',
    ]
    team_clean_df = team_df[team_keep_cols].copy().rename(columns={x:y for x,y in list(zip(team_keep_cols, athlete_keep_cols))})
    athlete_clean_df = single_athlete_df[athlete_keep_cols].copy()

    #############
    # Concat
    entry_df = pd.concat([team_clean_df, athlete_clean_df])

    # #############
    # # Check entries per event per country
    # check_df = entry_df[["event_key", "country"]].value_counts().to_frame().reset_index()
    # check_df = check_df[check_df["count"]>1]
    # check_df["event_key"].value_counts()

    #############
    # Return
    return entry_df
