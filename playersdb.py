from pathlib import Path
import pandas as pd

# Obté el directori on es troba aquest arxiu (playersdb.py) i apunta a la carpeta 'csv'
BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "csv"

playersStatisticsData = pd.read_csv(CSV_DIR / 'PlayerStatistics.csv', low_memory=False)
playersData = pd.read_csv(CSV_DIR / 'Players.csv', low_memory=False)

def get_team_players(playersStatisticsData, playersData, selected_team_id, selected_year):
    if 'playerteamId' not in playersStatisticsData.columns:
        raise ValueError("The 'playerteamId' column is missing from playersStatisticsData.")
    if 'personId' not in playersStatisticsData.columns or 'personId' not in playersData.columns:
        raise ValueError("The 'personId' column is missing from one of the dataframes.")
    if 'gameDate' not in playersStatisticsData.columns:
        raise ValueError("The 'gameDate' column is missing from playersStatisticsData.")

    game_dates = pd.to_datetime(playersStatisticsData['gameDate'], errors='coerce')
    team_stats = playersStatisticsData[
        (playersStatisticsData['playerteamId'] == selected_team_id)
        & (game_dates.dt.year == selected_year)
    ].copy()

    sum_columns = [
        'win', 'home', 'numMinutes', 'points', 'assists', 'blocks', 'steals',
        'fieldGoalsAttempted', 'fieldGoalsMade', 'threePointersAttempted',
        'threePointersMade', 'freeThrowsAttempted', 'freeThrowsMade',
        'reboundsDefensive', 'reboundsOffensive', 'reboundsTotal',
        'foulsPersonal', 'turnovers', 'plusMinusPoints'
    ]
    sum_columns = [col for col in sum_columns if col in team_stats.columns]
    team_stats[sum_columns] = team_stats[sum_columns].apply(pd.to_numeric, errors='coerce')

    aggregation = {col: 'first' for col in team_stats.columns if col not in sum_columns and col != 'personId'}
    aggregation.update({col: 'sum' for col in sum_columns})
    team_stats = team_stats.groupby('personId', as_index=False).agg(aggregation)
    team_stats['gamesPlayed'] = (
        playersStatisticsData[
            (playersStatisticsData['playerteamId'] == selected_team_id)
            & (game_dates.dt.year == selected_year)
        ]
        .groupby('personId')
        .size()
        .reindex(team_stats['personId'])
        .to_numpy()
    )

    percentage_pairs = {
        'fieldGoalsPercentage': ('fieldGoalsMade', 'fieldGoalsAttempted'),
        'threePointersPercentage': ('threePointersMade', 'threePointersAttempted'),
        'freeThrowsPercentage': ('freeThrowsMade', 'freeThrowsAttempted'),
    }
    for percentage, (made, attempted) in percentage_pairs.items():
        if made in team_stats.columns and attempted in team_stats.columns:
            team_stats[percentage] = (
                team_stats[made].div(team_stats[attempted].where(team_stats[attempted] != 0))
                .fillna(0)
            )

    team_players = playersData[playersData['personId'].isin(team_stats['personId'])].copy()

    player_columns = [
        'personId', 'firstName', 'lastName', 'birthDate', 'country',
        'heightInches', 'bodyWeightLbs', 'jersey', 'guard', 'forward', 'center'
    ]
    player_columns = [col for col in player_columns if col in team_players.columns]
    team_players = team_players[player_columns]

    players_db = team_stats.merge(
        team_players,
        on='personId',
        how='left',
        suffixes=('_stats', '_player')
    )

    players_db = players_db.reset_index(drop=True)

    return selected_team_id, players_db

selected_team_id, players_db = get_team_players(
    playersStatisticsData,
    playersData,
    selected_team_id=1610612759,
    selected_year=2026,
)