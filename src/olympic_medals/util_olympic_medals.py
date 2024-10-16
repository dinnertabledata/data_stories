#########################################################################
# 0. Imports

# Core DTD
from util_core import *


#########################################################################
# 1. Data

DATA_PATH = "../../data/"

####################################
# Supporting function

##########
# Filter to get records with valid event details
def filter_valid_events(df, col1, col2, label):
    orig_len = len(df)
    df = df.dropna(subset=[col1, col2])
    new_len = len(df)
    print(f"[{label}] Dropped {orig_len-new_len}/{orig_len} ({(orig_len-new_len)/orig_len*100:.1f}%) records with null event details.")
    return df 

##########
# Filter on current records
def filter_current(df, col, label):
    orig_len = len(df)
    df = df[df[col]==True].copy()
    new_len = len(df)
    print(f"[{label}] Dropped {orig_len-new_len}/{orig_len} ({(orig_len-new_len)/orig_len*100:.1f}%) non-current records.")
    return df 

##########
# Add event keys
def add_event_key(df, col1, col2, multi=False):
    if not multi:
        df["event_key"] = df.apply(lambda x: f"{x[col1].lower()}: {x[col2].lower()}", axis=1)
    else:
        df["event_key"] = df.apply(lambda x: list(itertools.product(x[col1], x[col2])), axis=1)
        df["event_key"] = df["event_key"].apply(lambda x: [f"{d.lower()}: {e.lower()}" for d,e in x])
    return df

##########
# Join medals
def join_medals(base_df, medal_df, join_cols, label):
    # Prep medal df
    medal_join_df = medal_df[join_cols+["medal_type"]].copy()
    medal_join_df[join_cols] = medal_join_df[join_cols].apply(lambda col: col.astype(str).str.lower())
    medal_join_df["medal_flag"] = True
    # Prep base df
    base_df[join_cols] = base_df[join_cols].apply(lambda col: col.astype(str).str.lower())
    # Join
    mod_df = pd.merge(base_df, medal_join_df, how="left", on=join_cols)
    # Add medal flag col
    mod_df["medal_flag"] = None
    mod_df.loc[mod_df["medal_type"].notna(), "medal_flag"] = True
    mod_df.loc[mod_df["medal_type"].isna(), "medal_flag"] = False
    right_cols = ["medal_flag", "medal_type"]
    left_cols = [x for x in mod_df.columns.tolist() if x not in right_cols]
    mod_df = mod_df[left_cols+right_cols]
    # Print summary
    medal_n = int(mod_df["medal_flag"].sum())
    total_n = len(mod_df)
    print(f"[{label}] Added medals for {medal_n}/{total_n} ({(medal_n)/total_n*100:.1f}%) records.")
    # Return
    return mod_df

####################################
# Load data on medals, events, teams, and athletes
def load_base_datasets():

    #############
    # Load medals - 1 record per medal per event
    medal_df = pd.read_csv(f"{DATA_PATH}/paris_2024/medals.csv")
    # Drop records with invalid event fields (some nulls)
    medal_df = filter_valid_events(medal_df, col1="discipline", col2="event", label="medal")
    # Add unique event key
    medal_df = add_event_key(medal_df, col1="discipline", col2="event")

    #############
    # Load medalists - 1 record per medal per athlete
    medalist_df = pd.read_csv(f"{DATA_PATH}/paris_2024/medallists.csv")
    # Drop records with invalid event fields (some nulls)
    medalist_df = filter_valid_events(medalist_df, col1="discipline", col2="event", label="medal")
    # Add unique event key
    medalist_df = add_event_key(medalist_df, col1="discipline", col2="event")

    #############
    # Load events
    event_df = pd.read_csv(f"{DATA_PATH}/paris_2024/events.csv")
    # Drop records with invalid event fields (some nulls)
    event_df = filter_valid_events(event_df, col1="sport", col2="event", label="event")
    # Add unique event key
    event_df = add_event_key(event_df, col1="sport", col2="event")

    #############
    # Load teams
    team_df = pd.read_csv(f"{DATA_PATH}/paris_2024/teams.csv")
    # Drop non-current records
    team_df = filter_current(team_df, col="current", label="team")
    # Drop records with invalid event fields (some nulls)
    team_df = filter_valid_events(team_df, col1="discipline", col2="events", label="team")
    # Add unique event key
    team_df = add_event_key(team_df, col1="discipline", col2="events")
    # Add medals
    team_df.rename(columns={"events": "event"}, inplace=True)
    team_df = join_medals(team_df, medal_df, join_cols=["code", "discipline", "event"], label="team")

    #############
    # Load athletes
    athlete_df = pd.read_csv(f"{DATA_PATH}/paris_2024/athletes.csv")
    # Drop non-current records
    athlete_df = filter_current(athlete_df, col="current", label="athlete")
    # Drop records with invalid event fields (some nulls)
    athlete_df = filter_valid_events(athlete_df, col1="disciplines", col2="events", label="athlete")
    # Convert disciplines and events to lists
    athlete_df["disciplines"] = athlete_df["disciplines"].apply(lambda x: ast.literal_eval(x))
    athlete_df["events"] = athlete_df["events"].apply(lambda x: ast.literal_eval(x))
    # Build list of combined disciplines+events, then use to explode
    athlete_df = add_event_key(athlete_df, col1="disciplines", col2="events", multi=True)
    athlete_df = athlete_df.explode("event_key")
    new_len = len(athlete_df)
    print(f"[athlete] Exploded to {new_len} records, one per athlete per event entered.")
    athlete_df["discipline"] = athlete_df["event_key"].apply(lambda x: x.split(": ")[0])
    athlete_df["event"] = athlete_df["event_key"].apply(lambda x: x.split(": ")[1])
    # Filter on only valid events
    valid_e = event_df["event_key"].unique()
    orig_len = len(athlete_df)
    athlete_df = athlete_df[athlete_df["event_key"].isin(valid_e)]
    new_len = len(athlete_df)
    print(f"[athlete] Dropped {orig_len-new_len}/{orig_len} ({(orig_len-new_len)/orig_len*100:.1f}%) invalid event records.")
    # Mark records that are team events
    team_e = team_df["event_key"].unique()
    athlete_df["event_type"] = None
    athlete_df.loc[athlete_df["event_key"].isin(team_e), "event_type"] = "Team"
    athlete_df.loc[~athlete_df["event_key"].isin(team_e), "event_type"] = "Individual"
    # Add medals
    medalist_df["code"] = medalist_df["code_athlete"]
    athlete_df = join_medals(athlete_df, medalist_df, join_cols=["code", "discipline", "event"], label="athlete")
    medalist_df.drop(columns=["code"], inplace=True)
    
    #############
    # Combine
    all_data = {
        "medal_df": medal_df,
        "medalist_df": medalist_df,
        "event_df": event_df,
        "athlete_df": athlete_df,
        "team_df": team_df
    }

    #############
    # Return
    return all_data

####################################
# Combine team and athlete data into a single "event entry" dataset
def build_combined_event_entries(team_df, athlete_df):

    #############
    # Isolate athletes in individual events
    orig_len = len(athlete_df)
    athelete_df_indiv = athlete_df[athlete_df["event_type"]=="Individual"].copy()
    new_len = len(athelete_df_indiv)
    print(f"[combining]: Removed {orig_len-new_len}/{orig_len} ({(orig_len-new_len)/orig_len*100:.1f}%) athlete records for Team type events.")
    athelete_df_indiv["num_athletes"] = 1

    #############
    # Define columns we want to keep
    team_keep_cols = [
        'event_key',
        'discipline',
        'event',
        'team_gender',
        'country_code',
        'country',
        'num_athletes',
        "medal_flag",
        "medal_type"

    ]
    athlete_keep_cols = [
        'event_key',
        'discipline',
        'event',
        'gender',
        'country_code',
        'country',
        'num_athletes',
        "medal_flag",
        "medal_type"
    ]
    team_lite_df = team_df[team_keep_cols].copy().rename(columns={x:y for x,y in list(zip(team_keep_cols, athlete_keep_cols))})
    athlete_lite_df = athelete_df_indiv[athlete_keep_cols]

    #############
    # Concat
    entry_df = pd.concat([team_lite_df, athlete_lite_df])

    # #############
    # # Check entries per event per country
    # check_df = entry_df[["event_key", "country"]].value_counts().to_frame().reset_index()
    # check_df = check_df[check_df["count"]>1]
    # check_df["event_key"].value_counts()

    #############
    # Return
    return entry_df

####################################
# Calc medal rates by country
def calc_country_medal_rates(entry_df):

    # Aggregate entries by country
    rate_df = entry_df.groupby("country").agg(
        entries=("event_key", "count"),
        total_medals=("medal_flag", "sum"),
        gold_medals=("medal_type", lambda x: (x == "Gold Medal").sum()),
        silver_medals=("medal_type", lambda x: (x == "Silver Medal").sum()),
        bronze_medals=("medal_type", lambda x: (x == "Bronze Medal").sum())
    ).reset_index()

    # Validate medal details
    detail_cols = ["gold_medals", "silver_medals", "bronze_medals"]
    rate_df["checksum"] = rate_df[detail_cols].sum(axis=1)
    rate_df["checksum_match"] = rate_df["checksum"] == rate_df["total_medals"]
    assert rate_df["checksum_match"].sum() == len(rate_df)
    rate_df.drop(columns=["checksum", "checksum_match"], inplace=True)

    # Weighted medal count
    def calc_weighted_medals(x, weights=[2, 1.5, 1], rate=False):
        gold_sum = x["gold_medals"] * weights[0]
        silver_sum = x["silver_medals"] * weights[1]
        bronze_sum = x["bronze_medals"] * weights[2]
        result = gold_sum + silver_sum + bronze_sum
        if rate: result /= x["entries"]
        return result
    rate_df["total_medals_weighted"] = rate_df.apply(lambda x: calc_weighted_medals(x), axis=1)

    # Unweighted medal rate
    rate_df["medal_rate_unweighted"] = rate_df["total_medals"] / rate_df["entries"]

    # Weighted medal rate
    rate_df["medal_rate_weighted"] = rate_df.apply(lambda x: calc_weighted_medals(x, rate=True), axis=1)

    # Add ranks and sort
    rate_df["medal_rate_unweighted_rank"] = rate_df["medal_rate_unweighted"].rank(method="min", ascending=False)
    rate_df["medal_rate_weighted_rank"] = rate_df["medal_rate_weighted"].rank(method="min", ascending=False)
    rate_df = rate_df.sort_values(["medal_rate_weighted_rank", "medal_rate_unweighted_rank"]).reset_index(drop=True)

    # Return
    return rate_df