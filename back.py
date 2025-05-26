import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats.mstats import winsorize
import time
import json
import math

# --- 1. Configuração da API ---
API_KEY = "9a4293083d36795d1d7081e201c0e688" # MANTENHA SUA CHAVE REAL AQUI
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v3.football.api-sports.io'
}

if API_KEY == "SUA_CHAVE_DE_API_AQUI" or not API_KEY:
    print("!!! ERRO CRÍTICO: CHAVE DE API NÃO CONFIGURADA !!!")

# --- Constantes Globais para o Modelo ---
PREV_SEASON_WINDOW_FOR_BASE = 19
CURRENT_SEASON_EWMA_SPAN = 7
TRANSITION_GAMES_COUNT = 6
H2H_LOOKBACK_GAMES = 4
H2H_EXPONENT_GLOBAL = 0.65
H2H_FACTOR_CLIP_MIN = 0.60
H2H_FACTOR_CLIP_MAX = 1.40
OUTLIER_LOWER_PERCENTILE = 0.02
OUTLIER_UPPER_PERCENTILE = 0.98
API_REQUEST_DELAY = 0.08 
DEFAULT_RHO_DIXON_COLES = -0.10

METRICS_BASE_NAMES = ['goals', 'xG', 'Shots', 'SoT', 'Corners', 'Fouls', 'YC', 'RC', 'SavePercentage']
API_STATS_TO_METRIC_MAP = {
    "Expected Goals": "xG", "Total Shots": "Shots", "Shots on Goal": "SoT",
    "Shots off Goal": "ShotsOff", "Blocked Shots": "ShotsBlocked",
    "Shots insidebox": "ShotsInBox", "Shots outsidebox": "ShotsOutBox",
    "Fouls": "Fouls", "Corner Kicks": "Corners", "Offsides": "Offsides",
    "Ball Possession": "Possession", "Yellow Cards": "YC", "Red Cards": "RC",
    "Goalkeeper Saves": "Saves"
}
MIN_LEAGUE_AVERAGES = {
    "goals_for_league_avg_home": 0.7, "goals_for_league_avg_away": 0.6,
    "goals_against_league_avg_home": 0.6, "goals_against_league_avg_away": 0.7,
    "xG_for_league_avg_home": 0.7, "xG_for_league_avg_away": 0.6,
    "xG_against_league_avg_home": 0.6, "xG_against_league_avg_away": 0.7,
    "Shots_for_league_avg_home": 7.0, "Shots_for_league_avg_away": 6.0,
    "SoT_for_league_avg_home": 2.0, "SoT_for_league_avg_away": 1.8,
    "Corners_for_league_avg_home": 3.0, "Corners_for_league_avg_away": 2.5,
    "Fouls_for_league_avg_home": 8.5, "Fouls_for_league_avg_away": 9.0,
    "YC_for_league_avg_home": 1.0, "YC_for_league_avg_away": 1.2,
    "RC_for_league_avg_home": 0.05, "RC_for_league_avg_away": 0.06,
    "SavePercentage_for_league_avg_home": 0.65, "SavePercentage_for_league_avg_away": 0.60,
}

# --- Funções Auxiliares para API ---
# ... (make_api_request, get_league_info, get_league_current_season, get_team_id, get_fixture_statistics mantidas como antes) ...
def make_api_request(endpoint, params):
    if API_KEY == "SUA_CHAVE_DE_API_AQUI" or not API_KEY: return None
    url = BASE_URL + endpoint
    time.sleep(API_REQUEST_DELAY)
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        json_response = response.json()
        errors = json_response.get("errors")
        if errors and ((isinstance(errors, list) and errors) or (isinstance(errors, dict) and errors)):
            is_critical = False
            if isinstance(errors, dict):
                if any(err_msg in str(errors.get(key_err,"")).lower() for key_err in errors for err_msg in ["token", "key", "subscription", "access restricted", "not subscribed", "quota", "limit", "plan"]):
                    is_critical = True
            elif isinstance(errors, list) and errors:
                if any(err_msg in str(err_item).lower() for err_item in errors for err_msg in ["token", "key", "subscription", "access restricted", "not subscribed", "quota", "limit", "plan"]):
                    is_critical = True
            if is_critical : print(f"ERRO CRÍTICO DA API: {endpoint}, {params} -> {json.dumps(errors, indent=2, ensure_ascii=False)}")
            return None if is_critical else json_response.get("response", [])
        message = json_response.get("message")
        if isinstance(message, str) and ("quota" in message.lower() or "limit" in message.lower() or "plan" in message.lower() or "invalid key" in message.lower() or "not subscribed" in message.lower()):
            print(f"ERRO DE PLANO/COTA/LIMITE/CHAVE DA API: {endpoint} -> {message}")
            return None
        return json_response.get("response", [])
    except requests.exceptions.HTTPError as http_err:
        print(f"Erro HTTP em {endpoint} com params {params}: {http_err}")
        if http_err.response is not None: print(f"Corpo: {http_err.response.text[:300]}...")
    except requests.exceptions.JSONDecodeError as json_err:
        print(f"Erro JSONDecodeError em {endpoint} com params {params}: {json_err}")
        if 'response' in locals() and hasattr(response, 'text'): print(f"Texto Recebido: {response.text[:300]}...")
    except Exception as e:
        print(f"Erro inesperado em {endpoint} com params {params}: {e}")
    return None

def get_league_info(league_name_query):
    print(f"Buscando informações para a liga: '{league_name_query}'...")
    response_data = make_api_request("/leagues", params={"search": league_name_query})
    if response_data:
        for item in response_data:
            league = item.get('league', {})
            country_name = item.get('country', {}).get('name', '')
            if league_name_query.lower() in league.get('name', '').lower():
                print(f"Liga encontrada: {league.get('name')} (ID: {league.get('id')}) País: {country_name}")
                return league, item.get('seasons', [])
        if response_data: 
            league = response_data[0].get('league', {})
            country_name = response_data[0].get('country', {}).get('name', '')
            print(f"AVISO: Usando primeira liga encontrada para '{league_name_query}': {league.get('name')} (ID: {league.get('id')}) País: {country_name}")
            return league, response_data[0].get('seasons', [])
    print(f"Nenhuma liga encontrada para '{league_name_query}'.")
    return None, []

def get_league_current_season(league_id):
    print(f"Buscando temporada atual para a Liga ID: {league_id}")
    data = make_api_request("/leagues", params={"id": str(league_id)})
    if data and len(data) > 0 and data[0].get("seasons"):
        seasons_info = data[0]["seasons"]
        current_year_api = None
        for season_info in seasons_info:
            if season_info.get("current") is True:
                current_year_api = int(season_info.get('year'))
                print(f"Temporada atual (current:true) encontrada para Liga ID {league_id}: {current_year_api}")
                return current_year_api
        
        # Se 'current:true' não foi encontrado, pegar o ano mais recente da lista
        if seasons_info:
            try:
                years = [s['year'] for s in seasons_info if isinstance(s, dict) and 'year' in s and isinstance(s['year'], int)]
                if years:
                    latest_season_from_list = max(years)
                    # Se a temporada mais recente listada for a próxima, mas hoje ainda não chegamos nela,
                    # podemos considerar a anterior como "atual" para dados históricos.
                    # Esta lógica pode precisar de ajuste dependendo da época do ano.
                    # Por simplicidade, vamos pegar a mais recente listada.
                    print(f"AVISO: 'current:true' não encontrado. Usando temporada mais recente listada: {latest_season_from_list}")
                    return latest_season_from_list
            except ValueError:
                print(f"ERRO: Nenhuma temporada válida (ano) encontrada para Liga ID {league_id} na lista de temporadas.")
    
    print(f"ERRO: Não foi possível determinar a temporada atual de forma confiável para a Liga ID {league_id}. Usando o ano atual: {datetime.now().year}")
    return int(datetime.now().year) 

def get_team_id(team_name_query, country_name_hint=None):
    params = {"search": team_name_query}
    log_msg = f"Buscando ID para '{team_name_query}'"
    if country_name_hint:
        params["country"] = country_name_hint
        log_msg += f" (País: {country_name_hint})"
    print(log_msg + "...")
    data = make_api_request("/teams", params=params)
    if data:
        exact_match = next((item.get('team') for item in data if isinstance(item.get('team'), dict) and item.get('team', {}).get('name', '').lower() == team_name_query.lower()), None)
        if exact_match:
            team_id = exact_match.get('id')
            print(f"ID com nome exato encontrado para '{team_name_query}': {team_id} ({exact_match.get('name')})")
            return team_id
        elif data and isinstance(data[0].get("team"), dict) : 
            first_match_team = data[0].get("team", {})
            team_id = first_match_team.get("id")
            print(f"AVISO: Nome exato '{team_name_query}' não encontrado. Usando ID do primeiro resultado da busca: {team_id} ({first_match_team.get('name')})")
            return team_id
    print(f"Nenhum time encontrado com o nome '{team_name_query}' (País: {country_name_hint if country_name_hint else 'Qualquer'}).")
    return None

def get_fixture_statistics(fixture_id):
    return make_api_request("/fixtures/statistics", params={"fixture": str(fixture_id)})

# --- ETAPA 1: COLETA E ORGANIZAÇÃO DA BASE ---
def get_fixtures_and_stats_for_seasons(league_id, team_ids_of_interest, season_years_to_fetch):
    # ... (mantida como antes) ...
    print(f"\n--- Coleta de Dados: Liga {league_id}, Times {team_ids_of_interest if team_ids_of_interest else 'TODOS'}, Temporadas {season_years_to_fetch} ---")
    all_games_data_list = []
    for season in season_years_to_fetch:
        print(f"   Processando Temporada: {season}")
        fixtures_this_season_for_teams = []
        if not team_ids_of_interest: 
            print(f"     Buscando TODOS os jogos da liga ID {league_id} para a temporada {season}...")
            all_league_fixtures = make_api_request("/fixtures", params={"league": str(league_id), "season": str(season)})
            if all_league_fixtures: fixtures_this_season_for_teams.extend(all_league_fixtures)
        elif all(isinstance(tid, (int, np.integer, str)) for tid in team_ids_of_interest): 
            for team_id in team_ids_of_interest:
                print(f"     Buscando jogos do Time ID {team_id} na Temporada {season}...")
                team_fixtures = make_api_request("/fixtures", params={"league": str(league_id), "season": str(season), "team": str(team_id)})
                if team_fixtures: fixtures_this_season_for_teams.extend(team_fixtures)

        unique_fixture_ids = set()
        processed_fixtures = []
        for fixture in fixtures_this_season_for_teams:
            if not isinstance(fixture, dict): continue
            fixture_id = fixture.get("fixture", {}).get("id")
            if fixture_id and fixture_id not in unique_fixture_ids:
                if fixture.get("fixture", {}).get("status", {}).get("short") == "FT": 
                    processed_fixtures.append(fixture)
                    unique_fixture_ids.add(fixture_id)
        print(f"     Total de {len(processed_fixtures)} partidas CONCLUÍDAS (únicas) encontradas para processar na temporada {season}.")

        for fixture_data in processed_fixtures:
            fixture_info = fixture_data.get("fixture", {})
            fixture_id = fixture_info.get("id")
            referee_name = fixture_info.get("referee")
            if referee_name is None: referee_name = "N/A" 
            if not fixture_id: continue

            home_team_info = fixture_data.get("teams", {}).get("home", {})
            away_team_info = fixture_data.get("teams", {}).get("away", {})
            if not home_team_info.get("id") or not away_team_info.get("id"): continue

            stats_response = get_fixture_statistics(fixture_id)
            fixture_stats_by_team = {}
            if stats_response:
                for team_block in stats_response:
                    if not isinstance(team_block, dict): continue
                    team_id_stat = team_block.get("team", {}).get("id")
                    team_metrics = {}
                    for stat_item in team_block.get("statistics", []):
                        if not isinstance(stat_item, dict): continue
                        api_stat_name = stat_item.get("type")
                        if api_stat_name in API_STATS_TO_METRIC_MAP:
                            metric_name_base = API_STATS_TO_METRIC_MAP[api_stat_name]
                            value_raw = stat_item.get("value")
                            processed_value = 0.0
                            if isinstance(value_raw, str) and '%' in value_raw:
                                try: processed_value = float(value_raw.replace('%', '')) / 100.0
                                except (ValueError, TypeError): processed_value = 0.0
                            elif value_raw is not None:
                                try: processed_value = float(value_raw)
                                except (ValueError, TypeError): processed_value = 0.0
                            team_metrics[metric_name_base] = processed_value
                    if team_id_stat: fixture_stats_by_team[team_id_stat] = team_metrics
            
            for is_home_perspective in [True, False]:
                game_row = {}
                current_team_info = home_team_info if is_home_perspective else away_team_info
                opponent_team_info = away_team_info if is_home_perspective else home_team_info
                current_team_id = current_team_info.get("id")
                if not current_team_id: continue
                
                game_row["fixture_id"] = fixture_id
                game_row["date"] = pd.to_datetime(fixture_info.get("date")).strftime('%Y-%m-%d') if fixture_info.get("date") else None
                game_row["season"] = int(season)
                game_row["referee_name"] = referee_name
                game_row["team_id"] = current_team_id
                game_row["team_name"] = current_team_info.get("name")
                game_row["opponent_id"] = opponent_team_info.get("id")
                game_row["opponent_name"] = opponent_team_info.get("name")
                game_row["is_home"] = 1 if is_home_perspective else 0
                
                goals_for_raw = fixture_data.get("goals", {}).get("home" if is_home_perspective else "away")
                goals_against_raw = fixture_data.get("goals", {}).get("away" if is_home_perspective else "home")
                try: game_row["goals_for"] = float(goals_for_raw) if goals_for_raw is not None else 0.0
                except (ValueError, TypeError): game_row["goals_for"] = 0.0
                try: game_row["goals_against"] = float(goals_against_raw) if goals_against_raw is not None else 0.0
                except (ValueError, TypeError): game_row["goals_against"] = 0.0

                team_stats = fixture_stats_by_team.get(game_row["team_id"], {})
                opponent_stats = fixture_stats_by_team.get(game_row["opponent_id"], {})

                for mapped_metric_name in API_STATS_TO_METRIC_MAP.values(): 
                    game_row[f"{mapped_metric_name}_for"] = team_stats.get(mapped_metric_name, 0.0)
                    game_row[f"{mapped_metric_name}_against"] = opponent_stats.get(mapped_metric_name, 0.0)

                game_row["xG_for"] = team_stats.get("xG", 0.0) 
                game_row["xG_against"] = opponent_stats.get("xG", 0.0)
                if pd.isna(game_row["xG_for"]) or game_row["xG_for"] == 0.0:
                    sot_f = team_stats.get("SoT", 0.0); shots_f = team_stats.get("Shots", 0.0)
                    game_row["xG_for"] = (float(sot_f if sot_f is not None else 0.0) * 0.30) + \
                                         (max(0, float(shots_f if shots_f is not None else 0.0) - float(sot_f if sot_f is not None else 0.0)) * 0.04)
                if pd.isna(game_row["xG_against"]) or game_row["xG_against"] == 0.0:
                    sot_a = opponent_stats.get("SoT", 0.0); shots_a = opponent_stats.get("Shots", 0.0)
                    game_row["xG_against"] = (float(sot_a if sot_a is not None else 0.0) * 0.30) + \
                                             (max(0, float(shots_a if shots_a is not None else 0.0) - float(sot_a if sot_a is not None else 0.0)) * 0.04)
                
                saves_made_by_team = game_row.get("Saves_for", 0.0) 
                sot_faced_by_team = game_row.get("SoT_against", 0.0) 
                if sot_faced_by_team > 0:
                    game_row["SavePercentage_for"] = saves_made_by_team / sot_faced_by_team
                elif saves_made_by_team == 0 and sot_faced_by_team == 0: 
                    game_row["SavePercentage_for"] = 1.0 
                else: 
                    game_row["SavePercentage_for"] = 0.0
                game_row["SavePercentage_for"] = np.clip(game_row["SavePercentage_for"], 0.0, 1.0)
                all_games_data_list.append(game_row)
    
    if not all_games_data_list:
        print("Nenhum dado de jogo foi efetivamente coletado e processado.")
        return pd.DataFrame() 
        
    df = pd.DataFrame(all_games_data_list)
    if df.empty: return df 

    df['date'] = pd.to_datetime(df['date'])
    sort_by_cols = ['team_id', 'is_home', 'date'] 
    if not all(col in df.columns for col in ['team_id', 'is_home']):
        sort_by_cols = ['date'] 
    df = df.sort_values(by=sort_by_cols).reset_index(drop=True)
    df.drop_duplicates(subset=['fixture_id', 'team_id'], inplace=True, keep='first')
    return df

# --- ETAPA 2: AJUSTES ESTATÍSTICOS INICIAIS ---
def apply_outlier_treatment(df, columns_to_treat, lower_p=OUTLIER_LOWER_PERCENTILE, upper_p=OUTLIER_UPPER_PERCENTILE):
    # ... (mantida como antes) ...
    print(f"Aplicando Winsorization ({lower_p*100:.1f}% / {100-upper_p*100:.1f}%) em: {', '.join(columns_to_treat)}")
    df_treated = df.copy()
    for col in columns_to_treat:
        if col in df_treated.columns and pd.api.types.is_numeric_dtype(df_treated[col]):
            non_nan_series = df_treated[col].dropna().astype(float)
            if not non_nan_series.empty and len(non_nan_series) > 2 : 
                winsorized_values = winsorize(non_nan_series, limits=(lower_p, 1-upper_p))
                df_treated.loc[non_nan_series.index, col] = winsorized_values
    return df_treated

def calculate_rolling_averages_with_season_dilution(df_all_games, current_season_year, 
                                                    metrics_to_average,
                                                    prev_season_window=PREV_SEASON_WINDOW_FOR_BASE, 
                                                    current_season_ewma_span=CURRENT_SEASON_EWMA_SPAN,
                                                    transition_games=TRANSITION_GAMES_COUNT):
    # ... (CORREÇÃO DO FUTUREWARNING APLICADA) ...
    print(f"Calculando médias com diluição (EWMA Span Atual={current_season_ewma_span}, Transição={transition_games} jogos)...")
    df_all_games = df_all_games.sort_values(by=['team_id', 'is_home', 'date']) 
    previous_season_year = int(current_season_year) - 1
    
    df_prev_season = df_all_games[df_all_games['season'] == previous_season_year].copy()
    prev_season_avg_stats_df = pd.DataFrame()
    if not df_prev_season.empty:
        prev_season_avg_stats_df = df_prev_season.groupby(['team_id', 'is_home'], group_keys=False)[metrics_to_average]\
            .apply(lambda x: x.tail(prev_season_window).mean(numeric_only=True))\
            .reset_index()\
            .rename(columns={m: f'{m}_prev_season_base' for m in metrics_to_average})

    df_current_season = df_all_games[df_all_games['season'] == current_season_year].copy()
    for metric in metrics_to_average:
        if metric in df_current_season.columns:
            df_current_season[f'{metric}_ewma_current'] = df_current_season.groupby(['team_id', 'is_home'], group_keys=False)\
                                                            [metric].transform(lambda x: x.ewm(span=current_season_ewma_span, min_periods=1, adjust=True).mean().shift(1))
        else:
            df_current_season[f'{metric}_ewma_current'] = np.nan 

    df_final_with_averages = df_all_games.copy()
    if not prev_season_avg_stats_df.empty:
        df_final_with_averages = pd.merge(df_final_with_averages, prev_season_avg_stats_df, on=['team_id', 'is_home'], how='left')
    
    ewma_cols_to_merge = [f'{m}_ewma_current' for m in metrics_to_average if f'{m}_ewma_current' in df_current_season.columns]
    if ewma_cols_to_merge:
        if 'fixture_id' in df_current_season.columns and 'team_id' in df_current_season.columns:
            df_current_season_for_merge = df_current_season[['fixture_id', 'team_id'] + ewma_cols_to_merge].drop_duplicates(subset=['fixture_id', 'team_id'])
            df_final_with_averages = pd.merge(df_final_with_averages, df_current_season_for_merge, 
                                              on=['fixture_id', 'team_id'], how='left', suffixes=('', '_ewma_temp'))
            for col_ewma in ewma_cols_to_merge:
                if f"{col_ewma}_ewma_temp" in df_final_with_averages.columns: 
                    df_final_with_averages[col_ewma] = df_final_with_averages[col_ewma].fillna(df_final_with_averages[f"{col_ewma}_ewma_temp"])
                    df_final_with_averages.drop(columns=[f"{col_ewma}_ewma_temp"], inplace=True) 
        else:
            print("AVISO: fixture_id ou team_id ausente em df_current_season. EWMA não pode ser merged corretamente.")

    if 'season' in df_final_with_averages.columns and 'date' in df_final_with_averages.columns:
        df_final_with_averages['game_num_current_season_cond'] = df_final_with_averages[
            df_final_with_averages['season'] == current_season_year
        ].groupby(['team_id', 'is_home'])['date'].rank(method='first', ascending=True)
        # CORREÇÃO DO FUTUREWARNING:
        df_final_with_averages['game_num_current_season_cond'] = df_final_with_averages['game_num_current_season_cond'].fillna(0)
    else:
        df_final_with_averages['game_num_current_season_cond'] = 0

    for metric in metrics_to_average:
        col_final_avg = f'{metric}_final_rolling_avg'
        col_prev_base = f'{metric}_prev_season_base'
        col_curr_ewma = f'{metric}_ewma_current'
        df_final_with_averages[col_final_avg] = np.nan 

        for index, row in df_final_with_averages.iterrows():
            default_value_for_metric = row.get(metric, 0.0) 
            if row['season'] != current_season_year:
                df_final_with_averages.loc[index, col_final_avg] = row.get(col_prev_base, default_value_for_metric)
                continue

            game_num = row['game_num_current_season_cond']
            current_ewma = row.get(col_curr_ewma, np.nan) 
            prev_base = row.get(col_prev_base, np.nan)

            if pd.isna(game_num) or game_num == 0: 
                 df_final_with_averages.loc[index, col_final_avg] = prev_base if not pd.isna(prev_base) else default_value_for_metric
                 continue
            
            final_avg = default_value_for_metric 
            if game_num <= transition_games :
                weight_current = game_num / transition_games if transition_games > 0 else 1.0
                weight_previous = 1.0 - weight_current
                val_for_current = current_ewma if not pd.isna(current_ewma) else prev_base 
                val_for_prev = prev_base if not pd.isna(prev_base) else current_ewma 

                if not pd.isna(val_for_current) and not pd.isna(val_for_prev):
                    final_avg = (val_for_prev * weight_previous) + (val_for_current * weight_current)
                elif not pd.isna(val_for_current): 
                    final_avg = val_for_current
                elif not pd.isna(val_for_prev): 
                    final_avg = val_for_prev
            else: 
                final_avg = current_ewma if not pd.isna(current_ewma) else default_value_for_metric
            
            df_final_with_averages.loc[index, col_final_avg] = final_avg
    return df_final_with_averages

# --- ETAPA 3: CRIAÇÃO DE RATINGS DE FORÇA ---
def calculate_strength_ratings(df_processed_all_league, current_season_year, metrics_base_names_for_ratings, previous_season_year):
    print("\n--- ETAPA 3: Calculando Ratings de Força ---")
    df_ratings_calc = df_processed_all_league.copy()
    league_averages_dict = {} 

    # Tenta usar dados da temporada atual para médias da liga
    df_for_lg_avg_calc = df_ratings_calc[df_ratings_calc['season'] == current_season_year].copy()
    
    if df_for_lg_avg_calc.empty:
        print(f"AVISO: Nenhum dado da temporada {current_season_year} para médias da liga. Tentando temporada anterior ({previous_season_year}).")
        df_for_lg_avg_calc = df_ratings_calc[df_ratings_calc['season'] == previous_season_year].copy()
        if df_for_lg_avg_calc.empty:
            print(f"AVISO: Nenhum dado da temporada {previous_season_year} para médias da liga. Usando MIN_LEAGUE_AVERAGES.")
            for metric_base in metrics_base_names_for_ratings:
                for suffix_type in ['for', 'against']:
                    if metric_base == 'SavePercentage' and suffix_type == 'against': continue 
                    for venue_suffix in ['home', 'away']:
                        key = f'{metric_base}_{suffix_type}_league_avg_{venue_suffix}'
                        league_averages_dict[key] = MIN_LEAGUE_AVERAGES.get(key, 1.0) 
            # Ratings serão 1.0 por default se não houver dados para calcular médias da liga
            # A função de VE usará MIN_LEAGUE_AVERAGES diretamente se league_averages_dict não tiver a chave
            # Não há necessidade de calcular ratings aqui se não há base para as médias da liga
            return df_ratings_calc, league_averages_dict # Retorna com ratings não calculados

    # Calcular médias da liga usando df_for_lg_avg_calc (que pode ser da temporada atual ou anterior)
    for metric_base in metrics_base_names_for_ratings:
        for suffix_type in ['for', 'against']:
            if metric_base == 'SavePercentage' and suffix_type == 'against': continue 
            col_avg_name = f'{metric_base}_{suffix_type}_final_rolling_avg' 
            for venue_flag, venue_name in [(1, "home"), (0, "away")]:
                key = f'{metric_base}_{suffix_type}_league_avg_{venue_name}' 
                if col_avg_name in df_for_lg_avg_calc.columns:
                    relevant_slice = df_for_lg_avg_calc[df_for_lg_avg_calc['is_home'] == venue_flag]
                    if not relevant_slice.empty and not relevant_slice[col_avg_name].isnull().all():
                        mean_val = relevant_slice[col_avg_name].mean()
                        league_averages_dict[key] = mean_val
                    else:
                        league_averages_dict[key] = np.nan 
                else:
                    league_averages_dict[key] = np.nan 
    
    for key_lg_avg, min_val_default in MIN_LEAGUE_AVERAGES.items():
        current_val = league_averages_dict.get(key_lg_avg, np.nan) 
        if key_lg_avg not in league_averages_dict or pd.isna(current_val) or current_val < min_val_default:
            league_averages_dict[key_lg_avg] = min_val_default
        current_val_after_potential_min = league_averages_dict.get(key_lg_avg, min_val_default)
        if current_val_after_potential_min == 0:
            league_averages_dict[key_lg_avg] = min_val_default if min_val_default > 0 else 0.01 

    # Calcular Ratings apenas para a temporada alvo (current_season_year)
    # Se a current_season_year não tem jogos, os ratings não serão calculados aqui e devem ser tratados no fallback em run_full_prediction_pipeline
    for index, row in df_ratings_calc.iterrows():
        if row['season'] != current_season_year: continue 
        is_home_flag = row['is_home'] == 1
        venue_suffix = "home" if is_home_flag else "away"
        for metric_base in metrics_base_names_for_ratings:
            if metric_base == 'SavePercentage': continue 
            team_off_avg_col = f'{metric_base}_for_final_rolling_avg'
            # Usa a média da liga (que pode ser da temporada anterior se a atual não tiver dados)
            league_off_avg = league_averages_dict.get(f'{metric_base}_for_league_avg_{venue_suffix}', 1.0) 
            if team_off_avg_col in row and not pd.isna(row[team_off_avg_col]) and league_off_avg != 0:
                df_ratings_calc.loc[index, f'{metric_base}_off_rating'] = row[team_off_avg_col] / league_off_avg
            else:
                df_ratings_calc.loc[index, f'{metric_base}_off_rating'] = 1.0 
            
            team_def_avg_col = f'{metric_base}_against_final_rolling_avg'
            league_def_avg = league_averages_dict.get(f'{metric_base}_against_league_avg_{venue_suffix}', 1.0)
            if team_def_avg_col in row and not pd.isna(row[team_def_avg_col]) and league_def_avg != 0:
                df_ratings_calc.loc[index, f'{metric_base}_def_rating'] = league_def_avg / row[team_def_avg_col] 
            else:
                df_ratings_calc.loc[index, f'{metric_base}_def_rating'] = 1.0
    return df_ratings_calc, league_averages_dict


# --- ETAPA 4 & 5: VE & AJUSTE H2H ---
# ... (calculate_ve_with_h2h mantida como antes) ...
def calculate_ve_with_h2h(
    latest_ratings_home_team_series, latest_ratings_away_team_series,
    league_averages_dict_ext, 
    h2h_fixtures_raw_list,
    team_id_home_match, team_id_away_match,
    metric_config 
):
    base_metric = metric_config['base_metric_name']
    event_display_name_debug = metric_config.get('display_name', base_metric.capitalize())
    key_home_avg_lookup = f"{base_metric}_for_league_avg_home" 
    key_away_avg_lookup = f"{base_metric}_for_league_avg_away"

    if base_metric in METRICS_BASE_NAMES and base_metric != 'SavePercentage':
        print(f"\nDEBUG VE para {event_display_name_debug} - Confronto: {latest_ratings_home_team_series.get('team_name','Mandante')} vs {latest_ratings_away_team_series.get('team_name','Visitante')}")
        print(f"   Média Liga Mandantes (usando chave '{key_home_avg_lookup}'): {league_averages_dict_ext.get(key_home_avg_lookup, 'N/A (usará MIN)')}")
        print(f"   Média Liga Visitantes (usando chave '{key_away_avg_lookup}'): {league_averages_dict_ext.get(key_away_avg_lookup, 'N/A (usará MIN)')}")

    rating_off_H = latest_ratings_home_team_series.get(f'{base_metric}_off_rating', 1.0) # Default para 1.0 se rating não existir (ex: dados da temporada anterior)
    rating_def_A = latest_ratings_away_team_series.get(f'{base_metric}_def_rating', 1.0) 
    league_avg_H_for = league_averages_dict_ext.get(key_home_avg_lookup, MIN_LEAGUE_AVERAGES.get(key_home_avg_lookup, 0.1))
    if pd.isna(league_avg_H_for) or league_avg_H_for <= 0: 
        league_avg_H_for = MIN_LEAGUE_AVERAGES.get(key_home_avg_lookup, 0.1)

    rating_off_A = latest_ratings_away_team_series.get(f'{base_metric}_off_rating', 1.0)
    rating_def_H = latest_ratings_home_team_series.get(f'{base_metric}_def_rating', 1.0) 
    league_avg_A_for = league_averages_dict_ext.get(key_away_avg_lookup, MIN_LEAGUE_AVERAGES.get(key_away_avg_lookup, 0.1))
    if pd.isna(league_avg_A_for) or league_avg_A_for <= 0:
        league_avg_A_for = MIN_LEAGUE_AVERAGES.get(key_away_avg_lookup, 0.1)

    ve_home_base = rating_off_H * rating_def_A * league_avg_H_for
    ve_away_base = rating_off_A * rating_def_H * league_avg_A_for
    
    h2h_factor_home_team_on_metric = 1.0
    h2h_factor_away_team_on_metric = 1.0
    if h2h_fixtures_raw_list:
        # ... (lógica H2H mantida) ...
        h2h_stats_for_home_match_team_metric_values = []
        h2h_stats_for_away_match_team_metric_values = []
        for fix_h2h_raw_data in h2h_fixtures_raw_list:
            h2h_fixture_id = fix_h2h_raw_data.get('fixture',{}).get('id')
            if not h2h_fixture_id: continue
            stats_resp_h2h = get_fixture_statistics(h2h_fixture_id) 
            if stats_resp_h2h:
                for team_block in stats_resp_h2h:
                    if not isinstance(team_block, dict): continue
                    team_id_stat = team_block.get("team", {}).get("id")
                    team_metrics_h2h = {}
                    for stat_item in team_block.get("statistics", []):
                        if not isinstance(stat_item, dict): continue
                        api_stat_name = stat_item.get("type")
                        if api_stat_name in API_STATS_TO_METRIC_MAP:
                            metric_name_base_h2h = API_STATS_TO_METRIC_MAP[api_stat_name]
                            value_raw = stat_item.get("value")
                            processed_value = 0.0
                            if isinstance(value_raw, str) and '%' in value_raw:
                                try: processed_value = float(value_raw.replace('%', '')) / 100.0
                                except (ValueError, TypeError): processed_value = 0.0
                            elif value_raw is not None:
                                try: processed_value = float(value_raw)
                                except (ValueError, TypeError): processed_value = 0.0
                            team_metrics_h2h[metric_name_base_h2h] = processed_value
                    
                    metric_value_in_h2h_game = team_metrics_h2h.get(base_metric, 0.0)
                    if team_id_stat == team_id_home_match: h2h_stats_for_home_match_team_metric_values.append(metric_value_in_h2h_game)
                    elif team_id_stat == team_id_away_match: h2h_stats_for_away_match_team_metric_values.append(metric_value_in_h2h_game)

        if h2h_stats_for_home_match_team_metric_values:
            avg_h2h_home = np.mean(h2h_stats_for_home_match_team_metric_values)
            avg_overall_home_metric_col = f'{base_metric}_for_final_rolling_avg' 
            avg_overall_home = latest_ratings_home_team_series.get(avg_overall_home_metric_col, league_avg_H_for)
            if pd.isna(avg_overall_home) or avg_overall_home == 0: avg_overall_home = league_avg_H_for 
            if avg_overall_home != 0: h2h_factor_home_team_on_metric = avg_h2h_home / avg_overall_home
        
        if h2h_stats_for_away_match_team_metric_values:
            avg_h2h_away = np.mean(h2h_stats_for_away_match_team_metric_values)
            avg_overall_away_metric_col = f'{base_metric}_for_final_rolling_avg'
            avg_overall_away = latest_ratings_away_team_series.get(avg_overall_away_metric_col, league_avg_A_for)
            if pd.isna(avg_overall_away) or avg_overall_away == 0: avg_overall_away = league_avg_A_for
            if avg_overall_away != 0: h2h_factor_away_team_on_metric = avg_h2h_away / avg_overall_away
    
    h2h_factor_home_team_on_metric = np.clip(h2h_factor_home_team_on_metric if not pd.isna(h2h_factor_home_team_on_metric) else 1.0, H2H_FACTOR_CLIP_MIN, H2H_FACTOR_CLIP_MAX)
    h2h_factor_away_team_on_metric = np.clip(h2h_factor_away_team_on_metric if not pd.isna(h2h_factor_away_team_on_metric) else 1.0, H2H_FACTOR_CLIP_MIN, H2H_FACTOR_CLIP_MAX)

    ve_home_final = ve_home_base * (h2h_factor_home_team_on_metric ** H2H_EXPONENT_GLOBAL)
    ve_away_final = ve_away_base * (h2h_factor_away_team_on_metric ** H2H_EXPONENT_GLOBAL)
    
    if base_metric in METRICS_BASE_NAMES and base_metric != 'SavePercentage':
         print(f"     VE Base Mandante ({base_metric}, antes H2H): {ve_home_base:.4f}, Fator H2H: {h2h_factor_home_team_on_metric:.4f}^{H2H_EXPONENT_GLOBAL} = {(h2h_factor_home_team_on_metric ** H2H_EXPONENT_GLOBAL):.4f} -> VE Final: {ve_home_final:.4f}")
         print(f"     VE Base Visitante ({base_metric}, antes H2H): {ve_away_base:.4f}, Fator H2H: {h2h_factor_away_team_on_metric:.4f}^{H2H_EXPONENT_GLOBAL} = {(h2h_factor_away_team_on_metric ** H2H_EXPONENT_GLOBAL):.4f} -> VE Final: {ve_away_final:.4f}")
    return ve_home_final, ve_away_final


# --- FUNÇÕES PARA PROBABILIDADE DE PLACAR (DIXON-COLES AJUSTADO) ---
# ... (calculate_score_probabilities mantida como antes, com BTTS) ...
def poisson_pmf(k, lambda_val):
    if lambda_val < 0: lambda_val = 0.0001 
    try:
        if lambda_val == 0 and k > 0: return 0.0
        if lambda_val == 0 and k == 0: return 1.0
        k_int = int(round(k)) 
        if k_int < 0: return 0.0 
        if k_int > 170 or lambda_val > 170 : 
            log_prob = k_int * math.log(lambda_val) - lambda_val - math.lgamma(k_int + 1)
            return math.exp(log_prob)
        else:
            return (lambda_val**k_int * math.exp(-lambda_val)) / math.factorial(k_int)
    except (OverflowError, ValueError) as e:
        if lambda_val == 0 and k_int == 0: return 1.0
        return 0.0

def calculate_score_probabilities(lambda_home, lambda_away, max_goals=7, top_n_scores=5, rho=None):
    lambda_home = max(0.01, lambda_home) 
    lambda_away = max(0.01, lambda_away)
    all_score_probs = []
    total_probability_mass = 0
    
    def tau(x, y, lh, la, r): 
        if r is None or r == 0: return 1.0 
        if x == 0 and y == 0: return 1 - (lh * la * r)
        elif x == 1 and y == 0: return 1 + (la * r)
        elif x == 0 and y == 1: return 1 + (lh * r)
        elif x == 1 and y == 1: return 1 - r
        else: return 1.0

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob_ij_ind = poisson_pmf(i, lambda_home) * poisson_pmf(j, lambda_away)
            adj_factor = tau(i, j, lambda_home, lambda_away, rho)
            prob_ij_adj = max(0, prob_ij_ind * adj_factor) 
            all_score_probs.append({'score': f"{i}-{j}", 'raw_prob': prob_ij_adj})
            total_probability_mass += prob_ij_adj
    
    prob_h_win, prob_draw, prob_a_win, prob_o25, prob_o15, prob_btts_yes = 0,0,0,0,0,0
    normalized_scores_list = []

    if total_probability_mass <= 0.000001: 
        print(f"AVISO: Massa total de probabilidade é ~zero ({total_probability_mass:.6f}). Lambdas: H={lambda_home:.2f}, A={lambda_away:.2f}, Rho={rho}")
        default_1x2 = {"home": 0.33, "draw": 0.34, "away": 0.33}
        default_ou = {"over": 0.5, "under": 0.5}
        default_btts = {"yes": 0.5, "no": 0.5}
        return {"prob_1X2": default_1x2, "prob_OverUnder15": default_ou, 
                "prob_OverUnder25": default_ou, "prob_BTTS": default_btts,
                "topScores": [{"score": "1-1", "probability": 0.1}]}

    for sp_item in all_score_probs:
        norm_prob = sp_item['raw_prob'] / total_probability_mass if total_probability_mass > 0 else 0
        normalized_scores_list.append({'score': sp_item['score'], 'probability': norm_prob})
        goals_h, goals_a = map(int, sp_item['score'].split('-'))
        if goals_h > goals_a: prob_h_win += norm_prob
        elif goals_a > goals_h: prob_a_win += norm_prob
        else: prob_draw += norm_prob
        if (goals_h + goals_a) > 2.5: prob_o25 += norm_prob
        if (goals_h + goals_a) > 1.5: prob_o15 += norm_prob
        if goals_h > 0 and goals_a > 0: prob_btts_yes += norm_prob 

    sum_1x2_final = prob_h_win + prob_draw + prob_a_win
    if sum_1x2_final > 0 and abs(sum_1x2_final - 1.0) > 0.001 : 
        prob_h_win /= sum_1x2_final; prob_draw /= sum_1x2_final; prob_a_win /= sum_1x2_final
    
    prob_u25 = 1.0 - prob_o25; prob_u15 = 1.0 - prob_o15; prob_btts_no = 1.0 - prob_btts_yes
    if prob_u25 < 0 : prob_u25 = 0
    if prob_u15 < 0 : prob_u15 = 0
    if prob_btts_no < 0 : prob_btts_no = 0

    if abs((prob_o25 + prob_u25) - 1.0) > 0.0001 and (prob_o25 + prob_u25) > 0:
        total_ou25 = prob_o25 + prob_u25
        prob_o25 /= total_ou25; prob_u25 /= total_ou25
    if abs((prob_o15 + prob_u15) - 1.0) > 0.0001 and (prob_o15 + prob_u15) > 0:
        total_ou15 = prob_o15 + prob_u15
        prob_o15 /= total_ou15; prob_u15 /= total_ou15
    if abs((prob_btts_yes + prob_btts_no) - 1.0) > 0.0001 and (prob_btts_yes + prob_btts_no) > 0:
        total_btts = prob_btts_yes + prob_btts_no
        prob_btts_yes /= total_btts; prob_btts_no /= total_btts

    normalized_scores_list.sort(key=lambda x: x['probability'], reverse=True)
    return {
        "prob_1X2": {"home": prob_h_win, "draw": prob_draw, "away": prob_a_win},
        "prob_OverUnder15": {"over": prob_o15, "under": prob_u15},
        "prob_OverUnder25": {"over": prob_o25, "under": prob_u25},
        "prob_BTTS": {"yes": prob_btts_yes, "no": prob_btts_no},
        "topScores": normalized_scores_list[:top_n_scores]
    }

# --- FUNÇÃO PARA FREQUÊNCIAS HISTÓRICAS DE MERCADO DA LIGA ---
# ... (calculate_historical_league_market_frequencies mantida como antes) ...
def calculate_historical_league_market_frequencies(df_league_games_season):
    print(f"\n--- Calculando Frequências Históricas de Mercado da Liga (Baseado em {len(df_league_games_season)} registros de times)...")
    if df_league_games_season.empty:
        print("AVISO: DataFrame de jogos da liga vazio para calcular frequências de mercado.")
        return { "prob1X2_LigaAvg": {"home": 0.45, "draw": 0.28, "away": 0.27}, "probOverUnder1_5_LigaAvg": {"over": 0.75, "under": 0.25}, "probOverUnder2_5_LigaAvg": {"over": 0.55, "under": 0.45}, "probBTTS_LigaAvg": {"yes": 0.50, "no": 0.50} }
    fixtures_processed = df_league_games_season[df_league_games_season['is_home'] == 1].copy()
    if fixtures_processed.empty:
        print("AVISO: Nenhum jogo da perspectiva do mandante encontrado para calcular frequências.")
        return { "prob1X2_LigaAvg": {"home": 0.45, "draw": 0.28, "away": 0.27}, "probOverUnder1_5_LigaAvg": {"over": 0.75, "under": 0.25}, "probOverUnder2_5_LigaAvg": {"over": 0.55, "under": 0.45}, "probBTTS_LigaAvg": {"yes": 0.50, "no": 0.50} }
    num_total_games = len(fixtures_processed)
    if num_total_games == 0:
        return { "prob1X2_LigaAvg": {"home": 0.45, "draw": 0.28, "away": 0.27}, "probOverUnder1_5_LigaAvg": {"over": 0.75, "under": 0.25}, "probOverUnder2_5_LigaAvg": {"over": 0.55, "under": 0.45}, "probBTTS_LigaAvg": {"yes": 0.50, "no": 0.50} }
    home_wins, draws, away_wins, over_1_5_count, over_2_5_count, btts_yes_count = 0,0,0,0,0,0
    for index, row in fixtures_processed.iterrows():
        gols_casa, gols_fora = row['goals_for'], row['goals_against']
        if pd.isna(gols_casa) or pd.isna(gols_fora): continue
        if gols_casa > gols_fora: home_wins += 1
        elif gols_fora > gols_casa: away_wins += 1
        else: draws += 1
        total_gols = gols_casa + gols_fora
        if total_gols > 1.5: over_1_5_count += 1
        if total_gols > 2.5: over_2_5_count += 1
        if gols_casa > 0 and gols_fora > 0: btts_yes_count += 1
    results = {
        "prob1X2_LigaAvg": {"home": home_wins / num_total_games if num_total_games > 0 else 0.33, "draw": draws / num_total_games if num_total_games > 0 else 0.34, "away": away_wins / num_total_games if num_total_games > 0 else 0.33 },
        "probOverUnder1_5_LigaAvg": {"over": over_1_5_count / num_total_games if num_total_games > 0 else 0.5, "under": (num_total_games - over_1_5_count) / num_total_games if num_total_games > 0 else 0.5 },
        "probOverUnder2_5_LigaAvg": {"over": over_2_5_count / num_total_games if num_total_games > 0 else 0.5, "under": (num_total_games - over_2_5_count) / num_total_games if num_total_games > 0 else 0.5 },
        "probBTTS_LigaAvg": {"yes": btts_yes_count / num_total_games if num_total_games > 0 else 0.5, "no": (num_total_games - btts_yes_count) / num_total_games if num_total_games > 0 else 0.5 }
    }
    print(f"     Frequências Históricas Calculadas: {results}")
    return results


# --- Função Adicional: Médias de Árbitros ---
# ... (calculate_referee_historical_averages mantida como antes) ...
def calculate_referee_historical_averages(all_league_games_df_with_stats, seasons_to_consider, ewma_span=15, min_games_ref=5):
    print("\n--- Calculando Médias Históricas de Árbitros (EWMA) ---")
    if 'referee_name' not in all_league_games_df_with_stats.columns:
        print("AVISO: Coluna 'referee_name' não encontrada.")
        return pd.DataFrame()
    df_refs_hist = all_league_games_df_with_stats[all_league_games_df_with_stats['season'].isin(seasons_to_consider) & all_league_games_df_with_stats['referee_name'].notna() & (all_league_games_df_with_stats['referee_name'] != 'N/A')].copy()
    if df_refs_hist.empty: return pd.DataFrame()
    required_ref_cols = ['YC_for', 'RC_for', 'Fouls_for'] 
    if not all(col in df_refs_hist.columns for col in required_ref_cols): return pd.DataFrame()
    df_refs_agg_hist = df_refs_hist.groupby(['fixture_id', 'referee_name', 'date', 'season']).agg(total_YC_game=('YC_for', 'sum'), total_RC_game=('RC_for', 'sum'), total_Fouls_game=('Fouls_for', 'sum')).reset_index()
    if df_refs_agg_hist.empty: return pd.DataFrame()
    df_refs_agg_hist = df_refs_agg_hist.sort_values(by=['referee_name', 'date'])
    ref_avg_metric_bases = ['total_YC_game', 'total_RC_game', 'total_Fouls_game']
    for metric in ref_avg_metric_bases:
        if metric in df_refs_agg_hist.columns:
            df_refs_agg_hist[f'{metric}_ewma_ref'] = df_refs_agg_hist.groupby('referee_name', group_keys=False)[metric].apply(lambda x: x.ewm(span=ewma_span, min_periods=1, adjust=True).mean().shift(1))
    latest_ref_stats_df = df_refs_agg_hist.groupby('referee_name').tail(1).copy()
    ref_game_counts_map = df_refs_agg_hist.groupby('referee_name')['fixture_id'].nunique()
    valid_refs_list = ref_game_counts_map[ref_game_counts_map >= min_games_ref].index
    latest_ref_stats_df = latest_ref_stats_df[latest_ref_stats_df['referee_name'].isin(valid_refs_list)]
    return latest_ref_stats_df


# --- Função Principal de Orquestração ---
def run_full_prediction_pipeline(team_name_home, team_name_away, league_name_query, target_season_year_input=None):
    print(f"\n--- INICIANDO PIPELINE: {team_name_home} (Casa) vs {team_name_away} (Fora) em {league_name_query} ---")
    league_info, _ = get_league_info(league_name_query)
    if not league_info: return None
    league_id = league_info['id']
    league_country_for_team_search = league_info.get('country', {}).get('name')
    current_season_year_param = target_season_year_input # Salva o input original
    current_season_year_for_data = target_season_year_input or get_league_current_season(league_id) # Ano para buscar dados
    
    print(f"Liga: {league_info.get('name')} (ID: {league_id}), País: {league_country_for_team_search}, Temporada Alvo para Predição: {current_season_year_param if current_season_year_param else current_season_year_for_data}")

    team_id_home = get_team_id(team_name_home, country_name_hint=league_country_for_team_search)
    team_id_away = get_team_id(team_name_away, country_name_hint=league_country_for_team_search)
    if not team_id_home or not team_id_away: return None
    
    # Determinar as temporadas para buscar dados históricos
    seasons_to_fetch_for_history = [current_season_year_for_data]
    # Verifica se precisamos da temporada anterior para diluição ou fallback
    home_fixtures_curr_season = make_api_request("/fixtures", params={"league": str(league_id), "season": str(current_season_year_for_data), "team": str(team_id_home), "status":"FT"})
    num_games_home_current_season = 0
    if home_fixtures_curr_season:
        num_games_home_current_season = len([f for f in home_fixtures_curr_season if isinstance(f,dict) and f.get("teams",{}).get("home",{}).get("id") == team_id_home]) 
    
    away_fixtures_curr_season = make_api_request("/fixtures", params={"league": str(league_id), "season": str(current_season_year_for_data), "team": str(team_id_away), "status":"FT"})
    num_games_away_current_season = 0
    if away_fixtures_curr_season:
        num_games_away_current_season = len([f for f in away_fixtures_curr_season if isinstance(f,dict) and f.get("teams",{}).get("away",{}).get("id") == team_id_away])
    
    print(f"Jogos do {team_name_home} como mandante na temporada {current_season_year_for_data}: {num_games_home_current_season}")
    print(f"Jogos do {team_name_away} como visitante na temporada {current_season_year_for_data}: {num_games_away_current_season}")

    # Adiciona temporada anterior se a atual tem poucos jogos OU se a temporada alvo é futura (sem jogos)
    if (num_games_home_current_season < TRANSITION_GAMES_COUNT or \
        num_games_away_current_season < TRANSITION_GAMES_COUNT) or \
       (target_season_year_input and target_season_year_input > datetime.now().year and num_games_home_current_season == 0 and num_games_away_current_season == 0) or \
       (not target_season_year_input and current_season_year_for_data > datetime.now().year and num_games_home_current_season == 0 and num_games_away_current_season == 0) : # Se target_season é futura E não tem jogos.
        
        previous_season_to_fetch = current_season_year_for_data - 1
        print(f"INFO: Dados insuficientes ou temporada futura ({current_season_year_for_data}). Buscando dados da temporada anterior ({previous_season_to_fetch}) para base e/ou diluição.")
        if previous_season_to_fetch not in seasons_to_fetch_for_history:
             seasons_to_fetch_for_history.append(previous_season_to_fetch)


    df_all_league_data = get_fixtures_and_stats_for_seasons(league_id, None, seasons_to_fetch_for_history)
    if df_all_league_data.empty:
        print("ERRO FATAL: Não foi possível coletar dados da liga. Encerrando.")
        return None

    metrics_for_processing = [
        f"{m_base}{suffix}" 
        for m_base in METRICS_BASE_NAMES 
        for suffix in ['_for', '_against'] 
        if f"{m_base}{suffix}" in df_all_league_data.columns and 
           not (m_base == 'SavePercentage' and suffix == '_against')
    ]
    df_all_league_data_processed = apply_outlier_treatment(df_all_league_data, metrics_for_processing)
    
    # Usar current_season_year_for_data para calcular médias rolantes e ratings
    df_league_with_final_averages = calculate_rolling_averages_with_season_dilution(df_all_league_data_processed, current_season_year_for_data, metrics_to_average=metrics_for_processing)
    df_league_with_ratings, league_averages_data_dict = calculate_strength_ratings(df_league_with_final_averages, current_season_year_for_data, METRICS_BASE_NAMES, current_season_year_for_data -1) # Passa o ano anterior para fallback
    
    # Dados para frequências de mercado: usa a temporada mais recente COM DADOS.
    season_for_market_freq = current_season_year_for_data
    df_season_games_for_freq = df_all_league_data[df_all_league_data['season'] == season_for_market_freq].copy()
    if df_season_games_for_freq.empty and (current_season_year_for_data -1) in seasons_to_fetch_for_history:
        print(f"AVISO: Sem jogos na temporada {season_for_market_freq} para frequências de mercado. Usando temporada {current_season_year_for_data -1}.")
        season_for_market_freq = current_season_year_for_data - 1
        df_season_games_for_freq = df_all_league_data[df_all_league_data['season'] == season_for_market_freq].copy()
    historical_market_league_avgs = calculate_historical_league_market_frequencies(df_season_games_for_freq)

    # Buscar dados recentes dos times
    # Prioriza a temporada alvo (current_season_year_for_data), depois a anterior
    
    latest_home_data_row = df_league_with_ratings[ (df_league_with_ratings['team_id'] == team_id_home) & (df_league_with_ratings['is_home'] == 1) & (df_league_with_ratings['season'] == current_season_year_for_data) ].sort_values('date', ascending=False).head(1)
    if latest_home_data_row.empty and (current_season_year_for_data -1) in seasons_to_fetch_for_history :
        print(f"AVISO: {team_name_home} sem dados como mandante em {current_season_year_for_data}. Usando {current_season_year_for_data-1}.")
        latest_home_data_row = df_league_with_final_averages[ (df_league_with_final_averages['team_id'] == team_id_home) & (df_league_with_final_averages['is_home'] == 1) & (df_league_with_final_averages['season'] == (current_season_year_for_data - 1)) ].sort_values('date', ascending=False).head(1)
        # Se ainda vazio, tenta qualquer jogo da temporada anterior para o time
        if latest_home_data_row.empty:
            latest_home_data_row = df_league_with_final_averages[ (df_league_with_final_averages['team_id'] == team_id_home) & (df_league_with_final_averages['season'] == (current_season_year_for_data - 1)) ].sort_values('date', ascending=False).head(1)


    latest_away_data_row = df_league_with_ratings[ (df_league_with_ratings['team_id'] == team_id_away) & (df_league_with_ratings['is_home'] == 0) & (df_league_with_ratings['season'] == current_season_year_for_data) ].sort_values('date', ascending=False).head(1)
    if latest_away_data_row.empty and (current_season_year_for_data-1) in seasons_to_fetch_for_history:
        print(f"AVISO: {team_name_away} sem dados como visitante em {current_season_year_for_data}. Usando {current_season_year_for_data-1}.")
        latest_away_data_row = df_league_with_final_averages[ (df_league_with_final_averages['team_id'] == team_id_away) & (df_league_with_final_averages['is_home'] == 0) & (df_league_with_final_averages['season'] == (current_season_year_for_data - 1)) ].sort_values('date', ascending=False).head(1)
        if latest_away_data_row.empty:
            latest_away_data_row = df_league_with_final_averages[ (df_league_with_final_averages['team_id'] == team_id_away) & (df_league_with_final_averages['season'] == (current_season_year_for_data - 1)) ].sort_values('date', ascending=False).head(1)

    if latest_home_data_row.empty or latest_away_data_row.empty:
        print(f"ERRO CRÍTICO: Dados de base (rolling averages) não encontrados para {team_name_home} ou {team_name_away} mesmo após fallback. Encerrando.")
        return None
    
    latest_home_series = latest_home_data_row.iloc[0]
    latest_away_series = latest_away_data_row.iloc[0]
    
    h2h_fixtures_list = get_h2h_fixtures(team_id_home, team_id_away, n_games=H2H_LOOKBACK_GAMES)
    
    valores_esperados_base_output = {}
    metrics_config_for_ve = {
        'gols': {'base_metric_name': 'goals'}, 'xG': {'base_metric_name': 'xG'},
        'finalizacoes': {'base_metric_name': 'Shots'}, 'sot': {'base_metric_name': 'SoT'},
        'escanteios': {'base_metric_name': 'Corners'}, 'faltas': {'base_metric_name': 'Fouls'},
        'cartoesAmarelos': {'base_metric_name': 'YC'}, 'cartoesVermelhos': {'base_metric_name': 'RC'}
    }

    for fe_key, config in metrics_config_for_ve.items():
        base_m_name = config['base_metric_name']
        ve_home, ve_away = calculate_ve_with_h2h(
            latest_home_series, latest_away_series, # Usar as series obtidas
            league_averages_data_dict, h2h_fixtures_list,
            team_id_home, team_id_away, {'base_metric_name': base_m_name, 'display_name': fe_key.capitalize()}
        )
        valores_esperados_base_output[f'{fe_key}_Home'] = ve_home
        valores_esperados_base_output[f'{fe_key}_Home_LigaAvg'] = league_averages_data_dict.get(f'{base_m_name}_for_league_avg_home', MIN_LEAGUE_AVERAGES.get(f'{base_m_name}_for_league_avg_home',0.1))
        valores_esperados_base_output[f'{fe_key}_Away'] = ve_away
        valores_esperados_base_output[f'{fe_key}_Away_LigaAvg'] = league_averages_data_dict.get(f'{base_m_name}_for_league_avg_away', MIN_LEAGUE_AVERAGES.get(f'{base_m_name}_for_league_avg_away',0.1))

    avg_save_pct_home = latest_home_series.get('SavePercentage_for_final_rolling_avg', MIN_LEAGUE_AVERAGES.get('SavePercentage_for_league_avg_home',0.7))
    avg_save_pct_away = latest_away_series.get('SavePercentage_for_final_rolling_avg', MIN_LEAGUE_AVERAGES.get('SavePercentage_for_league_avg_away',0.7))
    
    valores_esperados_base_output['TaxaDefesaGoleiro_Home'] = float(avg_save_pct_home) if not pd.isna(avg_save_pct_home) else MIN_LEAGUE_AVERAGES.get('SavePercentage_for_league_avg_home',0.7)
    valores_esperados_base_output['TaxaDefesaGoleiro_Home_LigaAvg'] = league_averages_data_dict.get('SavePercentage_for_league_avg_home', MIN_LEAGUE_AVERAGES.get('SavePercentage_for_league_avg_home',0.7))
    valores_esperados_base_output['TaxaDefesaGoleiro_Away'] = float(avg_save_pct_away) if not pd.isna(avg_save_pct_away) else MIN_LEAGUE_AVERAGES.get('SavePercentage_for_league_avg_away',0.7)
    valores_esperados_base_output['TaxaDefesaGoleiro_Away_LigaAvg'] = league_averages_data_dict.get('SavePercentage_for_league_avg_away', MIN_LEAGUE_AVERAGES.get('SavePercentage_for_league_avg_away',0.7))

    lambda_h_match = valores_esperados_base_output.get('gols_Home', 0.1) # Usar .get com default
    lambda_a_match = valores_esperados_base_output.get('gols_Away', 0.1)
    probabilidades_partida_backend = calculate_score_probabilities(lambda_h_match, lambda_a_match, rho=DEFAULT_RHO_DIXON_COLES)
    
    probabilidades_base_formatado_para_frontend = {
        "prob1X2": probabilidades_partida_backend.get("prob_1X2"),
        "prob1X2_LigaAvg": historical_market_league_avgs.get("prob1X2_LigaAvg"),
        "probOverUnder1_5": probabilidades_partida_backend.get("prob_OverUnder15"),
        "probOverUnder1_5_LigaAvg": historical_market_league_avgs.get("probOverUnder1_5_LigaAvg"),
        "probOverUnder2_5": probabilidades_partida_backend.get("prob_OverUnder25"),
        "probOverUnder2_5_LigaAvg": historical_market_league_avgs.get("probOverUnder2_5_LigaAvg"),
        "probBTTS": probabilidades_partida_backend.get("prob_BTTS"),
        "probBTTS_LigaAvg": historical_market_league_avgs.get("probBTTS_LigaAvg"),
        "topScores": probabilidades_partida_backend.get("topScores"),
    }

    arbitros_info = None
    if 'referee_name' in df_all_league_data.columns and not df_all_league_data.empty:
        ref_stats_df = calculate_referee_historical_averages(df_all_league_data, seasons_to_fetch_for_history)
        if ref_stats_df is not None and not ref_stats_df.empty:
            arbitros_info = ref_stats_df[['referee_name'] + [col for col in ref_stats_df.columns if '_ewma_ref' in col]].head(15).to_dict(orient='records')

    return {
        "valores_esperados_base": valores_esperados_base_output,
        "probabilidades_base_backend": probabilidades_base_formatado_para_frontend,
        "arbitros_disponiveis_com_media": arbitros_info 
    }

# --- Exemplo de Uso ---
if __name__ == "__main__":
    if API_KEY == "SUA_CHAVE_DE_API_AQUI" or not API_KEY:
        print("\nExecução de exemplo pulada pois a API_KEY não foi definida no script.")
    else:
        print("\nIniciando pipeline de exemplo com dados da API...")
        selected_team_a_name = "Liverpool" 
        selected_team_b_name = "Manchester City"
        selected_league_name = "Premier League" 
        target_season_for_prediction = 2025 # Testando uma temporada futura sem jogos

        resultados_para_frontend = run_full_prediction_pipeline(
            selected_team_a_name,
            selected_team_b_name,
            selected_league_name,
            target_season_year_input=target_season_for_prediction
        )
        if resultados_para_frontend:
            print("\n\n--- ESTRUTURA DE DADOS FINAL PARA O FRONTEND ---")
            print(json.dumps(resultados_para_frontend, indent=4, ensure_ascii=False))
        else:
            print("\nPipeline não pôde gerar os dados para o frontend.")
