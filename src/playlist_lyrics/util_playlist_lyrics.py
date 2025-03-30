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

####################################
# Athlete analysis
def run_athlete_analysis(athlete_df):

    ################################
    # Basic stats

    # Total number of athletes
    total_athletes = athlete_df['code'].nunique()

    # Athletes competing individually, on a team, or both.
    # Based on `event_type` column, assuming it contains 'Individual', 'Team', or both.
    individual_athletes = athlete_df[athlete_df['event_type'] == 'Individual']['code'].nunique()
    team_athletes = athlete_df[athlete_df['event_type'] == 'Team']['code'].nunique()

    # Group athletes by their name and check if they participate in both 'Individual' and 'Team' events
    # Filter for athletes who have both 'Individual' and 'Team' event types
    athlete_event_types = athlete_df.groupby('code')['event_type'].unique()
    both_athletes = athlete_event_types[athlete_event_types.apply(lambda x: set(['Individual', 'Team']).issubset(set(x)))].index.nunique()

    # Athletes earning 1+ medal
    medal_athletes = athlete_df[athlete_df['medal_flag'] == True]['code'].nunique()

    # Athletes earning 2+, 3+, 4+ medals
    athlete_medal_counts = athlete_df[athlete_df['medal_flag'] == True].groupby('code').size()
    medal_2_plus = (athlete_medal_counts >= 2).sum()
    medal_3_plus = (athlete_medal_counts >= 3).sum()
    medal_4_plus = (athlete_medal_counts >= 4).sum()

    # Do your chances of earning a medal increase in team sports?
    individual_medal_percentage = athlete_df[(athlete_df['event_type'] == 'Individual') & (athlete_df['medal_flag'] == True)]['code'].nunique() / individual_athletes
    team_medal_percentage = athlete_df[(athlete_df['event_type'] == 'Team') & (athlete_df['medal_flag'] == True)]['code'].nunique() / team_athletes

    ################################
    # Print

    # Print total number of athletes
    print(f"Total number of athletes: {total_athletes}")

    # Print athletes competing individually, on teams, or both
    print(f"\nAthletes competing individually: {individual_athletes}")
    print(f"Athletes competing on teams: {team_athletes}")
    print(f"Athletes competing in both individual and team events: {both_athletes}")

    # Print the number of athletes earning 1+ medal, 2+, 3+, and 4+ medals
    print(f"\nAthletes earning 1+ medal: {medal_athletes} ({medal_athletes/total_athletes*100:.1f}%)")
    print(f"Athletes earning 2+ medals: {medal_2_plus} ({medal_2_plus/total_athletes*100:.1f}%)")
    print(f"Athletes earning 3+ medals: {medal_3_plus} ({medal_3_plus/total_athletes*100:.1f}%)")
    print(f"Athletes earning 4+ medals: {medal_4_plus} ({medal_4_plus/total_athletes*100:.1f}%)")

    # Print the percentage of athletes who earn medals in individual and team sports
    print(f"\nPercentage of individual athletes earning a medal: {individual_medal_percentage * 100:.2f}%")
    print(f"Percentage of team athletes earning a medal: {team_medal_percentage * 100:.2f}%")

    ################################
    # More interesting view - event counts by athlete and by discipline

    # Step 1: Prepare the Data

    # Group by 'name' and aggregate event count, event details, and discipline counts
    athlete_event_summary = athlete_df.groupby(['code', 'name', 'country']).agg(
        event_count=('event_key', 'size'),  # Count of events for each athlete
        medal_count=('medal_flag', 'sum'),  # Count of medals won
        unique_discipline_count=('discipline', lambda x: len(set(x))),  # Count of unique disciplines per athlete
        discipline_list=('discipline', lambda x: '; '.join(sorted([f"{d}" for d in set(x)]))),  # Disciplines with counts
        discipline_detail=('discipline', lambda x: '; '.join(sorted([f"{d} ({x.tolist().count(d)}x)" for d in set(x)]))),  # Disciplines with counts
        event_details=('event_key', lambda x: '; '.join(sorted(x))),  # Semicolon-separated event details
    ).reset_index()

    athlete_event_summary = athlete_event_summary.sort_values(by='event_count', ascending=False)

    # Split into two DataFrames
    athlete_event_summary_single = athlete_event_summary[athlete_event_summary['unique_discipline_count'] == 1]
    athlete_event_summary_multi = athlete_event_summary[athlete_event_summary['unique_discipline_count'] > 1]

    ################################
    # Plotly viz

    # Step 2: Prepare Heatmap Data for Single Disciplines
    heatmap_data_single = (
        athlete_event_summary_single
        .groupby(['discipline_list', 'event_count'])
        .agg(unique_athlete_count=('code', 'nunique'))  # Count unique athletes
        .reset_index()
    )

    # Calculate totals for single discipline
    total_per_discipline_single = heatmap_data_single.groupby('discipline_list')['unique_athlete_count'].sum().reset_index()
    total_per_discipline_single.rename(columns={'unique_athlete_count': 'total_athletes'}, inplace=True)

    total_per_event_count_single = heatmap_data_single.groupby('event_count')['unique_athlete_count'].sum().reset_index()
    total_per_event_count_single.rename(columns={'unique_athlete_count': 'total_athletes'}, inplace=True)

    heatmap_data_single = heatmap_data_single.merge(total_per_discipline_single, on='discipline_list')
    heatmap_data_single['percentage'] = heatmap_data_single['unique_athlete_count'] / heatmap_data_single['total_athletes'] * 100

    heatmap_pivot_count_single = heatmap_data_single.pivot(index='discipline_list', columns='event_count', values='unique_athlete_count').fillna(0)
    heatmap_pivot_percent_single = heatmap_data_single.pivot(index='discipline_list', columns='event_count', values='percentage').fillna(0)

    # Step 3: Prepare Heatmap Data for Multiple Disciplines
    heatmap_data_multi = (
        athlete_event_summary_multi
        .groupby(['discipline_list', 'event_count'])
        .agg(unique_athlete_count=('code', 'nunique'))  # Count unique athletes
        .reset_index()
    )

    # Calculate totals for multiple discipline
    total_per_discipline_multi = heatmap_data_multi.groupby('discipline_list')['unique_athlete_count'].sum().reset_index()
    total_per_discipline_multi.rename(columns={'unique_athlete_count': 'total_athletes'}, inplace=True)

    total_per_event_count_multi = heatmap_data_multi.groupby('event_count')['unique_athlete_count'].sum().reset_index()
    total_per_event_count_multi.rename(columns={'unique_athlete_count': 'total_athletes'}, inplace=True)

    heatmap_data_multi = heatmap_data_multi.merge(total_per_discipline_multi, on='discipline_list')
    heatmap_data_multi['percentage'] = heatmap_data_multi['unique_athlete_count'] / heatmap_data_multi['total_athletes'] * 100

    heatmap_pivot_count_multi = heatmap_data_multi.pivot(index='discipline_list', columns='event_count', values='unique_athlete_count').fillna(0)
    heatmap_pivot_percent_multi = heatmap_data_multi.pivot(index='discipline_list', columns='event_count', values='percentage').fillna(0)

    # Fill in missing columns for multi
    max_single = max(heatmap_pivot_count_single.columns.tolist())
    max_multi = max(heatmap_pivot_count_multi.columns.tolist())
    missing = [1] + [x for x in range(max_multi+1, max_single+1)]
    for col in missing: 
        heatmap_pivot_count_multi[col] = 0
        heatmap_pivot_percent_multi[col] = 0
        new_row = pd.DataFrame([[col, 0]], columns=total_per_event_count_multi.columns)
        total_per_event_count_multi = pd.concat([total_per_event_count_multi, new_row], ignore_index=True)
    heatmap_pivot_count_multi = heatmap_pivot_count_multi[sorted(heatmap_pivot_count_multi.columns)]
    heatmap_pivot_percent_multi = heatmap_pivot_percent_multi[sorted(heatmap_pivot_percent_multi.columns)]

    # Sort y axis
    heatmap_pivot_count_single = heatmap_pivot_count_single.sort_values("discipline_list", ascending=False)
    heatmap_pivot_percent_single = heatmap_pivot_percent_single.sort_values("discipline_list", ascending=False)
    heatmap_pivot_count_multi = heatmap_pivot_count_multi.sort_values("discipline_list", ascending=False)
    heatmap_pivot_percent_multi = heatmap_pivot_percent_multi.sort_values("discipline_list", ascending=False)

    # Step 4: Create Subplots for Heatmaps
    fig = make_subplots(
        rows=2, 
        cols=1, 
        subplot_titles=('Athletes Competing in a Single Discipline', 'Athletes Competing in Multiple Disciplines'), 
        row_heights=[0.91, 0.09],
        vertical_spacing=0.1  # Decrease the vertical spacing between subplots
    )

    # Add heatmap for single disciplines
    fig.add_trace(go.Heatmap(
        z=heatmap_pivot_percent_single.values,
        x=heatmap_pivot_percent_single.columns,
        y=heatmap_pivot_percent_single.index,
        colorscale='tempo',  # Changed to tempo color scale
        colorbar=dict(title='Percentage of Athletes in Discipline'),
        text=[[
            f"Discipline: {heatmap_pivot_percent_single.index[i]}<br>"
            f"Event Count: {heatmap_pivot_percent_single.columns[j]}<br>"
            f"Unique Athletes: {int(heatmap_pivot_count_single.iat[i, j])}<br>"
            f"Percentage: {heatmap_pivot_percent_single.iat[i, j]:.1f}%"
            for j in range(len(heatmap_pivot_percent_single.columns))
        ] for i in range(len(heatmap_pivot_percent_single))],
        hoverinfo='text',  # Use custom text for tooltips
    ), row=1, col=1)

    # Add heatmap for multiple disciplines
    fig.add_trace(go.Heatmap(
        z=heatmap_pivot_percent_multi.values,
        x=heatmap_pivot_percent_multi.columns,
        y=heatmap_pivot_percent_multi.index,
        colorscale='tempo',  # Changed to tempo color scale
        colorbar=dict(title='Percentage of Athletes in Discipline'),
        text=[[
            f"Discipline: {heatmap_pivot_percent_multi.index[i]}<br>"
            f"Event Count: {heatmap_pivot_percent_multi.columns[j]}<br>"
            f"Unique Athletes: {int(heatmap_pivot_count_multi.iat[i, j])}<br>"
            f"Percentage: {heatmap_pivot_percent_multi.iat[i, j]:.1f}%"
            for j in range(len(heatmap_pivot_percent_multi.columns))
        ] for i in range(len(heatmap_pivot_percent_multi))],
        hoverinfo='text',  # Use custom text for tooltips
    ), row=2, col=1)

    # Update axis tick labels for single disciplines
    fig.update_yaxes(
        title='Discipline (Single)',
        tickvals=heatmap_pivot_count_single.index,
        ticktext=[f"{discipline} ({total_per_discipline_single.loc[total_per_discipline_single['discipline_list'] == discipline, 'total_athletes'].values[0]})" for discipline in heatmap_pivot_count_single.index],
        row=1, col=1
    )

    # Update x-axis labels for single disciplines
    fig.update_xaxes(
        title='Event Count per Athlete',
        tickvals=heatmap_pivot_count_single.columns,
        ticktext=[f"{count}\n({total_per_event_count_single.loc[total_per_event_count_single['event_count'] == count, 'total_athletes'].values[0]})" for count in heatmap_pivot_count_single.columns],
        row=1, col=1
    )

    # Update axis tick labels for multiple disciplines
    fig.update_yaxes(
        title='Discipline (Multiple)',
        tickvals=heatmap_pivot_count_multi.index,
        ticktext=[f"{discipline} ({total_per_discipline_multi.loc[total_per_discipline_multi['discipline_list'] == discipline, 'total_athletes'].values[0]})" for discipline in heatmap_pivot_count_multi.index],
        row=2, col=1
    )

    # Update x-axis labels for multiple disciplines
    fig.update_xaxes(
        title='Event Count per Athlete',
        tickvals=heatmap_pivot_count_multi.columns,
        ticktext=[f"{count}\n({total_per_event_count_multi.loc[total_per_event_count_multi['event_count'] == count, 'total_athletes'].values[0]})" for count in heatmap_pivot_count_multi.columns],
        row=2, col=1
    )

    # Adjust the layout heights for better spacing
    fig.update_layout(
        title='Heatmap of Unique Athletes by Discipline and Event Count',
        height=1200,  # Overall height of the figure
    )

    # Show the figure
    fig.show()

    ################################
    # Return
    return {
        "athlete_event_summary": athlete_event_summary,
        "fig": fig
    }