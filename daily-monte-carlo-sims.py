import pandas as pd
import numpy as np
import nflreadpy as nfl
from datetime import datetime, timedelta

# --- CONFIGURATION ---
SIMULATIONS = 10000
HISTORY_DAYS = 840
CURRENT_SEASON = 2025
DECAY_RATE = 0.00475
GARBAGE_MIN = 0.05
GARBAGE_MAX = 0.95

# Context
WIND_THRESHOLD = 15
WIND_PASS_IMPACT = 0.85
HFA_DEFENSE_BOOST_DEFAULT = 0.03

TEAM_MAP = {
    'ARZ': 'ARI', 'BLT': 'BAL', 'CLV': 'CLE', 'HST': 'HOU',
    'LAR': 'LA', 'STL': 'LA', 'SD': 'LAC', 'OAK': 'LV'
}

def weighted_avg_and_std(values, weights):
    if len(values) == 0: return 0.0, 0.0
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return average, np.sqrt(variance)

def format_clock(seconds, phase="REG"):
    seconds = max(0, seconds)
    m, s = divmod(int(seconds), 60)
    if phase == "OT": return f"OT {m:02d}:{s:02d}"
    if seconds > 1800: return f"1H {m:02d}:{s:02d}"
    return f"2H {m:02d}:{s:02d}"

def format_field(yardline, possession):
    if yardline <= 50: return f"{possession} {int(yardline)}"
    return f"Opp {int(100-yardline)}"

class AdvancedNFLSimulator:
    def __init__(self):
        self.pbp = pd.DataFrame()
        self.profiles = {}
        self.def_mults = {}
        self.hfa_map = {} 
        self.league_avgs = {}
    
    def load_data(self, hfa_file="nfl_hfa_ratings.csv"):
        print("--- Loading Data & Calculating Advanced Profiles ---")
        try:
            hfa_df = pd.read_csv(hfa_file)
            self.hfa_map = hfa_df.set_index('Team')['HFA SR'].to_dict()
        except FileNotFoundError:
            self.hfa_map = {}

        seasons = [CURRENT_SEASON-2, CURRENT_SEASON-1, CURRENT_SEASON]
        try:
            df = nfl.load_pbp(seasons=seasons).to_pandas()
        except:
            print("CRITICAL ERROR: Could not load PBP data.")
            return

        df['game_date'] = pd.to_datetime(df['game_date'])
        cutoff = datetime.now() - timedelta(days=HISTORY_DAYS)
        df = df[df['game_date'] >= cutoff].copy()
        df = df[(df['wp'] >= GARBAGE_MIN) & (df['wp'] <= GARBAGE_MAX)]
        
        current_date = datetime.now()
        df['days_ago'] = (current_date - df['game_date']).dt.days.clip(lower=0)
        df['time_weight'] = np.exp(-DECAY_RATE * df['days_ago'])
        
        df = df[df['play_type'].isin(['run', 'pass', 'punt', 'field_goal', 'no_play'])]
        self.pbp = df
        
        self._build_profiles()

    def _build_profiles(self):
        print("--- Building Profiles ---")
        
        # 1. DISTANCE & CONTEXT
        def get_dist_bucket(dist):
            if dist <= 3: return 'short'
            if dist <= 7: return 'med'
            return 'long'
        self.pbp['dist_bucket'] = self.pbp['ydstogo'].apply(get_dist_bucket)
        
        self.pbp['score_diff'] = self.pbp['posteam_score'] - self.pbp['defteam_score']
        def get_context(row):
            if row['score_diff'] > 8: return 'leading'
            if row['score_diff'] < -8: return 'trailing'
            return 'neutral'
        self.pbp['context'] = self.pbp.apply(get_context, axis=1)

        # 2. PLAY CALLING
        league_groups = self.pbp.groupby(['down', 'dist_bucket', 'context'])
        self.league_pass_rates = {}
        for name, group in league_groups:
            is_pass = (group['play_type'] == 'pass').astype(int)
            self.league_pass_rates[name] = np.average(is_pass, weights=group['time_weight'])

        playcalling_dict = {}
        team_groups = self.pbp.groupby(['posteam', 'down', 'dist_bucket', 'context'])
        for name, group in team_groups:
            is_pass = (group['play_type'] == 'pass').astype(int)
            playcalling_dict[name] = np.average(is_pass, weights=group['time_weight'])
            
        # 3. PACE & CLOCK LOGIC
        self.pbp['next_snap_time'] = self.pbp.groupby(['game_id', 'drive'])['game_seconds_remaining'].shift(-1)
        self.pbp['seconds_consumed'] = self.pbp['game_seconds_remaining'] - self.pbp['next_snap_time']
        
        def get_pace_type(row):
            if row['play_type'] == 'run': return 'run'
            if row['play_type'] == 'pass':
                return 'pass_complete' if row['complete_pass'] == 1 else 'pass_incomplete'
            return 'other'
        self.pbp['pace_type'] = self.pbp.apply(get_pace_type, axis=1)
        valid_pace = self.pbp[(self.pbp['seconds_consumed'] >= 0) & (self.pbp['seconds_consumed'] < 60)]
        
        pace_stats = valid_pace.groupby(['posteam', 'pace_type']).apply(
            lambda x: np.average(x['seconds_consumed'], weights=x['time_weight'])
        )

        oob_plays = self.pbp[self.pbp['play_type'].isin(['run', 'pass'])]
        oob_rates = {}
        for team, group in oob_plays.groupby('posteam'):
             oob_rates[team] = np.average(group['out_of_bounds'].fillna(0), weights=group['time_weight'])
        
        incomplete_plays = valid_pace[valid_pace['pace_type'] == 'pass_incomplete']
        avg_play_duration = incomplete_plays['seconds_consumed'].mean()
        if np.isnan(avg_play_duration): avg_play_duration = 6.0

        # 4. EFFICIENCY
        self.pbp['field_zone'] = np.where(self.pbp['yardline_100'] <= 20, 'redzone', 'open')
        efficiency_dict = {}
        eff_plays = self.pbp[self.pbp['play_type'].isin(['run', 'pass'])]
        
        for (team, zone), team_group in eff_plays.groupby(['posteam', 'field_zone']):
            # RUN
            runs = team_group[team_group['play_type'] == 'run']
            if len(runs) > 0:
                r_mu, r_sigma = weighted_avg_and_std(runs['yards_gained'].fillna(0).values, runs['time_weight'].values)
                r_fumble = np.average(runs['fumble_lost'], weights=runs['time_weight'])
            else:
                r_mu, r_sigma, r_fumble = 3.5, 3.0, 0.01
            efficiency_dict[(team, zone, 'run')] = {'mu': r_mu, 'sigma': r_sigma, 'fumble': r_fumble}

            # PASS
            passes = team_group[team_group['play_type'] == 'pass']
            if len(passes) > 0:
                sack_rate = np.average(passes['sack'], weights=passes['time_weight'])
                non_sacks = passes[passes['sack'] == 0]
                if len(non_sacks) > 0:
                    comp_rate = np.average(non_sacks['complete_pass'], weights=non_sacks['time_weight'])
                    int_rate = np.average(non_sacks['interception'], weights=non_sacks['time_weight'])
                    completions = non_sacks[non_sacks['complete_pass'] == 1]
                    if len(completions) > 0:
                        p_mu, p_sigma = weighted_avg_and_std(completions['yards_gained'].values, completions['time_weight'].values)
                        p_fumble = np.average(completions['fumble_lost'], weights=completions['time_weight'])
                    else:
                        p_mu, p_sigma, p_fumble = 10.0, 5.0, 0.01
                else:
                    comp_rate, int_rate, p_fumble, p_mu, p_sigma = 0.6, 0.03, 0.01, 7.0, 5.0
            else:
                sack_rate, comp_rate, int_rate, p_fumble, p_mu, p_sigma = 0.07, 0.6, 0.03, 0.01, 7.0, 5.0

            efficiency_dict[(team, zone, 'pass')] = {
                'mu': p_mu, 'sigma': p_sigma, 'fumble': p_fumble, 
                'intercept': int_rate, 'complete': comp_rate, 'sack': sack_rate
            }

        # 5. DEFENSE MULTS
        self.def_mults = {}
        league_run = np.average(eff_plays[eff_plays['play_type']=='run']['yards_gained'], weights=eff_plays[eff_plays['play_type']=='run']['time_weight'])
        league_pass = np.average(eff_plays[(eff_plays['play_type']=='pass') & (eff_plays['complete_pass']==1)]['yards_gained'], 
                                 weights=eff_plays[(eff_plays['play_type']=='pass') & (eff_plays['complete_pass']==1)]['time_weight'])
        
        for team, group in eff_plays.groupby('defteam'):
            self.def_mults[team] = {}
            tr = group[group['play_type']=='run']
            self.def_mults[team]['run'] = (np.average(tr['yards_gained'], weights=tr['time_weight']) / league_run) if len(tr)>0 else 1.0
            tp = group[(group['play_type']=='pass') & (group['complete_pass']==1)]
            self.def_mults[team]['pass'] = (np.average(tp['yards_gained'], weights=tp['time_weight']) / league_pass) if len(tp)>0 else 1.0

        # 6. PENALTIES
        pen_dict = {}
        for team, group in self.pbp.groupby('posteam'):
            off_pen = group[(group['penalty'] == 1) & (group['penalty_team'] == team)]
            pen_dict[(team, 'off')] = np.sum(off_pen['time_weight']) / group['time_weight'].sum()
        
        def_pen_stats = {}
        for team, group in self.pbp.groupby('defteam'):
            def_pen_plays = group[(group['penalty'] == 1) & (group['penalty_team'] == team)]
            total_rate = np.sum(def_pen_plays['time_weight']) / group['time_weight'].sum()
            pen_dict[(team, 'def')] = total_rate
            
            if len(def_pen_plays) > 0:
                is_dpi = def_pen_plays['penalty_type'].str.contains('Pass Interference', na=False, case=False)
                is_major = (def_pen_plays['penalty_yards'] == 15) & (~is_dpi)
                w = def_pen_plays['time_weight']
                dpi_weight = w[is_dpi].sum()
                major_weight = w[is_major].sum()
                total_weight = w.sum()
                dpi_share = dpi_weight / total_weight
                major_share = major_weight / total_weight
                dpi_yards = def_pen_plays[is_dpi]['penalty_yards']
                if len(dpi_yards) > 0:
                    d_mu = np.average(dpi_yards, weights=w[is_dpi])
                    d_std = np.sqrt(np.average((dpi_yards - d_mu)**2, weights=w[is_dpi]))
                else:
                    d_mu, d_std = 15.0, 10.0
                def_pen_stats[team] = {'dpi_share': dpi_share, 'major_share': major_share, 'dpi_mu': d_mu, 'dpi_std': d_std}
            else:
                def_pen_stats[team] = {'dpi_share': 0.1, 'major_share': 0.15, 'dpi_mu': 15.0, 'dpi_std': 10.0}

        # 7. PUNTING
        punt_stats = {}
        punts = self.pbp[self.pbp['play_type'] == 'punt'].copy()
        punts['net_yards'] = punts['kick_distance'] - punts['return_yards'].fillna(0)
        for team, group in punts.groupby('posteam'):
             p_mu = np.average(group['net_yards'].fillna(40), weights=group['time_weight'])
             p_std = np.sqrt(np.average((group['net_yards'].fillna(40) - p_mu)**2, weights=group['time_weight']))
             punt_stats[team] = {'mu': p_mu, 'sigma': p_std}
             
        # 8. KICKING
        kicking_stats = {}
        fgs = self.pbp[self.pbp['play_type'] == 'field_goal'].copy()
        for team, group in fgs.groupby('posteam'):
            made_fgs = group[group['field_goal_result'] == 'made']
            max_made = made_fgs['kick_distance'].max()
            if np.isnan(max_made): max_made = 50.0
            
            short_try = group[group['kick_distance'] < 40]
            short_acc = np.average((short_try['field_goal_result']=='made'), weights=short_try['time_weight']) if len(short_try)>0 else 0.98
            
            med_try = group[(group['kick_distance'] >= 40) & (group['kick_distance'] < 50)]
            med_acc = np.average((med_try['field_goal_result']=='made'), weights=med_try['time_weight']) if len(med_try)>0 else 0.85

            long_try = group[group['kick_distance'] >= 50]
            long_acc = np.average((long_try['field_goal_result']=='made'), weights=long_try['time_weight']) if len(long_try)>0 else 0.65
            
            kicking_stats[team] = {'max_made': max_made, 'short_acc': short_acc, 'med_acc': med_acc, 'long_acc': long_acc}

        # 9. BREAKAWAY RUN RATES
        # Define a breakaway as a run of 15+ yards
        run_plays = self.pbp[self.pbp['play_type'] == 'run']
        breakaway_plays = run_plays[run_plays['yards_gained'] >= 15]
        
        # Calculate League Average Rate first
        if len(run_plays) > 0:
            league_bk_rate = len(breakaway_plays) / len(run_plays)
        else:
            league_bk_rate = 0.035 # Default fallback (3.5%)

        breakaway_stats = {}
        for team, group in run_plays.groupby('posteam'):
            n_runs = len(group)
            n_breakaways = len(group[group['yards_gained'] >= 15])
            
            # REGRESSION TO THE MEAN:
            # We add 50 "league average runs" to the team's sample.
            # This prevents a team with few runs from having a wild 0% or 10% rate.
            regressed_rate = (n_breakaways + (50 * league_bk_rate)) / (n_runs + 50)
            breakaway_stats[team] = regressed_rate

        # Store in profiles (Add this to your self.profiles dictionary below)
        self.profiles['breakaway_run'] = breakaway_stats
        self.profiles['league_breakaway_run'] = league_bk_rate        
        
        
        self.profiles = {
            'efficiency': efficiency_dict,
            'pace': pace_stats.to_dict(),
            'penalties': pen_dict,
            'penalty_details': def_pen_stats,
            'punting': punt_stats,
            'kicking': kicking_stats,
            'playcalling': playcalling_dict,
            'oob_rates': oob_rates,
            'play_duration': avg_play_duration,
            'breakaway_run': breakaway_stats,
            'league_breakaway_run': league_bk_rate
        }

    def _resolve_play_outcome(self, off, def_, zone, ptype, stats, def_mult, hfa_impact, verbose):
        """
        Calculates the result of a play, injecting 'Breakaway' logic to fix low totals.
        Returns: (yards, is_complete, is_turnover, desc_tag)
        """
        yards = 0
        is_complete = True
        is_turnover = False
        desc_tag = ""

        # Apply Defensive Multiplier & HFA to base efficiency
        # If defense is good (mult < 1.0), they reduce yardage.
        adjusted_mu = stats['mu'] * def_mult
        
        # --- RUN LOGIC ---
        if ptype == 'run':
            # 1. Check Fumble
            if np.random.random() < stats['fumble']:
                is_turnover = True
                yards = 0 # Fumbles usually happen at LOS or slight gain, simplifying to 0 for sim
                desc_tag = "FUMBLE"
            
            # 2. Check BREAKAWAY (Team Specific Rate)
            else:
                # RETRIEVE TEAM RATE HERE
                # Fallback to league average if team not found
                league_avg = self.profiles.get('league_breakaway_run', 0.035)
                bk_prob = self.profiles['breakaway_run'].get(off, league_avg)
                
                if np.random.random() < bk_prob:
                    # Log-normal distribution for breakaway yards
                    raw_yards = np.random.lognormal(3.0, 0.6) 
                    yards = int(max(15, raw_yards))
                    yards = min(yards, 99)
                    desc_tag = "BREAKAWAY RUN"
            
                # 3. Standard Run
                else:
                    raw_yards = np.random.normal(adjusted_mu, stats['sigma'])
                    yards = int(max(raw_yards, -3))
                    if np.random.random() < 0.10: 
                        yards = np.random.randint(-3, 1)

        # --- PASS LOGIC ---
        else:
            # 1. Check Sack
            if np.random.random() < stats['sack']:
                yards = -7
                is_complete = False
                desc_tag = "SACK"
                # Small chance of strip-sack
                if np.random.random() < 0.015: 
                    is_turnover = True
                    desc_tag += " / FUMBLE"

            # 2. Check Interception
            elif np.random.random() < stats['intercept']:
                is_turnover = True
                is_complete = False # Technically incomplete stats-wise for yardage calc
                yards = 0
                desc_tag = "INTERCEPTION"

            # 3. Check Completion
            elif np.random.random() > stats['complete']:
                is_complete = False
                yards = 0
                desc_tag = "INCOMPLETE"

            # 4. COMPLETED PASS
            else:
                # Check BREAKAWAY (The Fix for Totals)
                # ~7% of completions go for big yardage
                if np.random.random() < 0.07:
                    # Normal dist centered on 35 yards, high variance
                    raw_yards = np.random.normal(35, 12)
                    yards = int(max(20, raw_yards)) # Minimum 20 yards for a "breakaway"
                    yards = min(yards, 99)
                    desc_tag = "DEEP BALL"
                    
                    # Add fumble chance on long run after catch
                    if np.random.random() < 0.01:
                        is_turnover = True
                        desc_tag += " / FUMBLE"
                else:
                    # Standard Completion
                    raw_yards = np.random.normal(adjusted_mu, stats['sigma'])
                    yards = int(max(raw_yards, -2))
                    # Standard fumble chance
                    if np.random.random() < stats['fumble']:
                        is_turnover = True
                        desc_tag = "FUMBLE"

        return yards, is_complete, is_turnover, desc_tag

    def _get_kickoff_start(self, team):
        # NFL Kickoff Return Distribution (Approximate)
        roll = np.random.random()
        
        if roll < 0.40: 
            return 35 # Standard Touchback (New Rules is 30)
        elif roll < 0.60:
            # Poor/Normal return
            # FIX: Changed sigma from 25 to 4
            return int(np.random.normal(18, 4)) 
        elif roll < 0.85:
            # Good return
            # FIX: Changed sigma from 35 to 5
            return int(np.random.normal(26, 5))
        elif roll < 0.95:
            # Great return
            return int(np.random.randint(35, 50))
        elif roll < 0.995:
            # Explosive return into opponent territory
            # Return yardline (e.g. 80 means own 80, which is opp 20)
            return int(np.random.randint(50, 85)) 
        else:
            # KICKOFF RETURN TOUCHDOWN (0.5% chance)
            return 100

    def simulate_matchup(self, home, away, wind_speed=0, is_dome=False, print_sample_game=False):
        results = []
        wind_mod = 1.0
        if not is_dome and wind_speed > WIND_THRESHOLD:
            wind_mod = WIND_PASS_IMPACT
        
        h_lookup = TEAM_MAP.get(home, home)
        hfa_impact = self.hfa_map.get(h_lookup, HFA_DEFENSE_BOOST_DEFAULT)
        
        print(f"Simulating {home} vs {away} | HFA: {hfa_impact:.1%} | Wind: {wind_speed}mph")
        
        if print_sample_game:
            print(f"\n{'='*60}\nSAMPLE GAME LOG ({away} @ {home})\n{'='*60}")
            self._play_game(home, away, wind_mod, wind_speed, is_dome, hfa_impact, verbose=True)
            print(f"{'='*60}\nEND SAMPLE LOG\n{'='*60}\n")

        for _ in range(SIMULATIONS):
            res = self._play_game(home, away, wind_mod, wind_speed, is_dome, hfa_impact, verbose=False)
            results.append(res)
            
        return pd.DataFrame(results)

    def _attempt_pat(self, off, def_, scores, clock, phase, wind_speed, verbose):
        diff = scores[off] - scores[def_] 
        go_for_2 = False
        minutes_left = clock / 60.0
        is_late = (phase == 'REG' and minutes_left < 10) or (phase == 'OT')
        
        if is_late:
            if diff == -2: go_for_2 = True
            elif diff == -5: go_for_2 = True
            elif diff == -1: 
                if minutes_left < 2: go_for_2 = True
            elif diff == 1: go_for_2 = True
            elif diff == 5: go_for_2 = True
        
        points_added = 0
        desc = ""
        
        if go_for_2:
            success = np.random.random() < 0.48
            if success:
                points_added = 2
                desc = "2PT GOOD"
            else:
                desc = "2PT FAILED"
        else:
            pat_prob = 0.94
            if wind_speed > 15: pat_prob = 0.90
            success = np.random.random() < pat_prob
            if success:
                points_added = 1
                desc = "XP GOOD"
            else:
                desc = "XP MISS"
                
        scores[off] += points_added
        if verbose: print(f"   >>> {desc} ({off} {scores[off]} - {def_} {scores[def_]})")
        return

    def _play_game(self, home, away, wind_mod, raw_wind, is_dome, hfa_impact, verbose=False):
        clock = 3600
        phase = 'REG' 
        scores = {home: 0, away: 0}
        timeouts = {home: 3, away: 3}
        halftime_processed = False
        
        # --- OPENING COIN TOSS & KICKOFF ---
        possession = np.random.choice([home, away])
        opponent = away if possession == home else home # Define opponent early for PAT logic
        
        # Calculate the opening field position
        start_yard = self._get_kickoff_start(possession)
        
        if start_yard >= 100:
            # OPENING KICKOFF RETURN TD!
            scores[possession] += 6
            if verbose: print(f"[{format_clock(clock, phase)}] OPENING KICKOFF RETURN TOUCHDOWN {possession}!")
            
            # Attempt PAT (Use 'opponent' since 'def_' isn't defined yet)
            self._attempt_pat(possession, opponent, scores, clock, phase, raw_wind, verbose)
            
            # Since they scored, they kick off to the opponent.
            # The opponent gets the ball for the first drive of the loop.
            possession = opponent
            
            # For simplicity, we assume the next kickoff is a standard return 
            # (to avoid infinite recursion of return TDs at 0:00)
            yardline = self._get_kickoff_start(possession)
            if yardline >= 100: yardline = 25 # Safety valve: Force touchback if back-to-back return TDs
            
        else:
            # Normal Start
            yardline = start_yard

        # Standard Drive Setup
        down, dist = 1, 10
        ot_drive_count = 0
        game_active = True
        
        while game_active:
            # --- HALFTIME RESET ---
            if phase == 'REG' and clock <= 1800 and not halftime_processed:
                timeouts = {home: 3, away: 3}
                halftime_processed = True
                clock_running = False 
                if verbose: print(f"[{format_clock(clock, phase)}] --- HALFTIME (Timeouts Reset) ---")

            # --- PHASE TRANSITION ---
            if clock <= 0:
                if phase == 'REG' and scores[home] == scores[away]:
                    phase = 'OT'
                    clock = 600
                    possession = np.random.choice([home, away])
                    timeouts = {home: 2, away: 2} # Reset to 2 for OT
                    yardline = 32
                    down, dist = 1, 10
                    ot_drive_count = 0
                    clock_running = False
                    if verbose: print(f"\n[{format_clock(clock, phase)}] --- OVERTIME: {possession} wins toss ---")
                else:
                    game_active = False
                    break

            off = possession
            def_ = away if off == home else home
            
            # Context
            diff = scores[off] - scores[def_]
            if diff > 8: ctx = 'leading'
            elif diff < -8: ctx = 'trailing'
            else: ctx = 'neutral'
            
            if dist <= 3: d_bucket = 'short'
            elif dist <= 7: d_bucket = 'med'
            else: d_bucket = 'long'
            
            zone = 'redzone' if yardline >= 80 else 'open'
            time_left_in_half = clock - 1800 if clock > 1800 else clock

            # --- PLAY CALL ---
            pass_prob = self.profiles['playcalling'].get((off, down, d_bucket, ctx))
            if pass_prob is None: pass_prob = self.league_pass_rates.get((down, d_bucket, ctx), 0.55)
            
            # Standard Adjustments
            if phase == 'REG' and clock < 300:
                if diff > 0: pass_prob -= 0.4
                if diff < 0: pass_prob += 0.4
            
            # --- NEW 3RD/4TH DOWN LOGIC OVERRIDE ---
            if down == 3 or down == 4:
                if dist <= 2:
                    pass_prob = 0.50
                elif dist <= 4:
                    pass_prob = 0.85
                else:
                    pass_prob = 1.0

            pass_prob = np.clip(pass_prob, 0.01, 1.0)

            
            # --- DEFENSIVE PENALTY ---
            if np.random.random() < self.profiles['penalties'].get((def_, 'def'), 0.015):
                pen_stats = self.profiles['penalty_details'].get(def_, {'dpi_share': 0.1, 'major_share': 0.15})
                roll = np.random.random()
                
                if roll < pen_stats['dpi_share']:
                    raw_dpi = np.random.normal(pen_stats.get('dpi_mu', 15), pen_stats.get('dpi_std', 10))
                    p_yards = max(1, int(raw_dpi))
                    dist_to_goal = 100 - yardline
                    p_yards = min(p_yards, dist_to_goal - 1)
                    p_yards = max(1, p_yards)
                    if verbose: print(f"[{format_clock(clock, phase)}] {def_} PENALTY: Pass Interference ({p_yards} yds)")

                elif roll < (pen_stats['dpi_share'] + pen_stats['major_share']):
                    p_yards = 15
                    dist_to_goal = 100 - yardline
                    if dist_to_goal < 30: 
                        p_yards = int(dist_to_goal / 2)
                        p_yards = max(1, p_yards)
                    if verbose: print(f"[{format_clock(clock, phase)}] {def_} PENALTY: Major/Unnecessary Roughness ({p_yards} yds)")
                    
                else:
                    p_yards = 5
                    dist_to_goal = 100 - yardline
                    if dist_to_goal < 10:
                        p_yards = int(dist_to_goal / 2)
                        p_yards = max(1, p_yards)
                    if verbose: print(f"[{format_clock(clock, phase)}] {def_} PENALTY: Defensive Holding/Offsides ({p_yards} yds)")

                yardline += p_yards
                down, dist = 1, 10
                if yardline >= 100: yardline = 99
                clock_running = False 
                continue

            # --- OFFENSIVE PENALTY ---
            if np.random.random() < self.profiles['penalties'].get((off, 'off'), 0.055):
                yardline = max(1, yardline - 10)
                clock -= 5
                dist += 10
                if verbose: print(f"[{format_clock(clock, phase)}] {off} OFFENSIVE PENALTY")
                continue

            # --- 4TH DOWN DECISIONS ---
            if down == 4:
                minutes = clock / 60.0
                deficit = -diff if diff < 0 else 0
                is_4q_or_ot = (phase == 'OT' or minutes < 15)

                must_go_punt_range = False
                if phase == 'REG':
                    if (9 <= deficit <= 16 and minutes < 4) or (1 <= deficit <= 8 and minutes < 2):
                        must_go_punt_range = True
                if phase == 'OT': 
                    if scores[def_] >= scores[off]: must_go_punt_range = True

                must_go_fg_range = False
                if phase == 'REG':
                    if (4 <= deficit <= 8 and minutes < 4) or (12 <= deficit <= 16 and minutes < 5):
                        must_go_fg_range = True
                
                if is_4q_or_ot and deficit > 3:
                    must_go_fg_range = True

                aggressive_go = (dist <= 2 and yardline >= 50)
                attempt_play = False
                
                # FG LOGIC
                kick_dist = (100 - yardline) + 18
                k_stats = self.profiles['kicking'].get(off, {'max_made': 55, 'short_acc': 0.95, 'med_acc': 0.85, 'long_acc': 0.60})
                
                weather_max_dist = k_stats['max_made']
                weather_acc_mod = 1.0
                if not is_dome and raw_wind > 0:
                    weather_max_dist -= (raw_wind / 3.0)
                    if raw_wind > 15: weather_acc_mod = 0.90
                    if raw_wind > 25: weather_acc_mod = 0.75
                
                in_fg_range = kick_dist <= (weather_max_dist + 2)
                
                if in_fg_range and kick_dist <= 65:
                    if must_go_fg_range:
                        attempt_play = True
                        if verbose: print(f"[{format_clock(clock, phase)}] {off} NEED TD: Going for it on 4th!")
                    else:
                        if kick_dist < 40: base_prob = k_stats['short_acc']
                        elif kick_dist < 50: base_prob = k_stats['med_acc']
                        else: base_prob = k_stats['long_acc']
                        
                        final_prob = base_prob * weather_acc_mod
                        if kick_dist > (weather_max_dist - 3): final_prob *= 0.8 
                        made = np.random.random() < final_prob
                        
                        if verbose: print(f"[{format_clock(clock, phase)}] {off} {int(kick_dist)} yd FG Attempt... {'GOOD' if made else 'MISS'}")
                        clock -= 5
                        clock_running = False 
                        
                        if made:
                            scores[off] += 3
                            if phase == 'OT':
                                if ot_drive_count == 0:
                                    if verbose: print(f"   >>> OT: {def_} must score.")
                                else:
                                    if scores[off] > scores[def_]:
                                        game_active = False
                                        if verbose: print(f"   >>> OVERTIME WINNER: {off}!")
                                        break
                                    elif scores[def_] > scores[off]:
                                        game_active = False
                                        if verbose: print(f"   >>> OVERTIME WINNER: {def_}!")
                                        break
                            possession = def_
                            new_start = self._get_kickoff_start(possession)
                            
                            if new_start >= 100:
                                # KICK RETURN TD!
                                scores[possession] += 6
                                if verbose: print(f"   >>> KICKOFF RETURN TOUCHDOWN {possession}!")
                                self._attempt_pat(possession, off, scores, clock, phase, raw_wind, verbose)
                                # Kick it right back to the other team
                                possession = off 
                                yardline = 30 
                                continue # Skip to next iteration
                            
                            yardline = new_start
                            down, dist = 1, 10
                            if phase == 'OT': ot_drive_count += 1
                        else:
                            if phase == 'OT' and scores[def_] > scores[off]:
                                game_active = False
                                break
                            possession = def_
                            yardline = 100 - (yardline + 7)
                            if yardline < 0: yardline = 20
                            down, dist = 1, 10
                            if phase == 'OT': ot_drive_count += 1
                        continue

                else: 
                    if must_go_punt_range:
                        attempt_play = True
                        if verbose: print(f"[{format_clock(clock, phase)}] {off} DESPERATION: Going for it!")
                    elif aggressive_go:
                        attempt_play = True
                        if verbose: print(f"[{format_clock(clock, phase)}] {off} ANALYTICS: Going for it (4th & {dist})!")
                    else:
                        p_stats = self.profiles['punting'].get(off, {'mu': 41.0, 'sigma': 4.0})
                        adj_mu = p_stats['mu'] - (raw_wind / 2.0)
                        dist_to_goal = 100 - yardline
                        
                        if adj_mu > dist_to_goal:
                            if verbose: print(f"[{format_clock(clock, phase)}] {off} PUNT (Pinning Attempt)")
                            new_start = np.random.randint(1, 21) 
                            yardline = new_start 
                        else:
                            if verbose: print(f"[{format_clock(clock, phase)}] {off} PUNT")
                            punt_dist = np.random.normal(adj_mu, p_stats['sigma'])
                            punt_dist = max(10, punt_dist)
                            new_yardline = 100 - (yardline + punt_dist)
                            if new_yardline <= 0:
                                new_yardline = 20
                                if verbose: print(f"   >>> Touchback")
                            yardline = new_yardline

                        clock_running = False 
                        
                        if phase == 'OT' and scores[def_] > scores[off]:
                            game_active = False
                            if verbose: print(f"   >>> OVERTIME WINNER: {def_} (Stop)!")
                            break
                        
                        possession = def_
                        down, dist = 1, 10
                        clock -= 40
                        if phase == 'OT': ot_drive_count += 1
                        continue
                
                # Fall through to execute

            
            # --- EXECUTE PLAY ---
            is_pass = np.random.random() < pass_prob
            ptype = 'pass' if is_pass else 'run'

            # Get INITIAL stats profile
            stats = self.profiles['efficiency'].get((off, zone, ptype), 
                    {'mu': 4.0, 'sigma': 4.0, 'complete': 0.6, 'intercept': 0.03, 'fumble': 0.01, 'sack': 0.07})

            # OVERRIDE: Goal-to-Go Efficiency Boost
            # We only need a tiny nudge to convert ~1 extra drive per game from FG to TD
            is_goal_to_go = (100 - yardline) <= 10
            
#            if is_goal_to_go:
#                if ptype == 'run':
                    # PREVIOUS (Too Strong): mu += 1.3, sigma = 1.5 
                    # NEW (Marginal):
                    # Add 0.4 yards to the average surge (e.g., 3.5 -> 3.9)
#                    stats['mu'] += 0.0  
                    
                    # Don't crush the variance (sigma). Keep it around 3.0.
                    # This allows for 1-yard gains or 0-yard stuffs, which forces 
                    # 3rd & 4th down decisions rather than automatic 1st downs.
#                    stats['sigma'] = 3.0 
                    
#                else:
                    # PREVIOUS (Too Strong): complete += 0.05
                    # NEW (Marginal):
                    # Tiny bump to completion % (2.5%) for short throws
#                    stats['complete'] += 0.0025 
                    
                    # QBs are still careful, but picks happen (tipped balls).
                    # 1.5% is a realistic low floor.
#                    stats['intercept'] = 0.03
            
            # (Deleted the duplicate 'stats =' line that was here)

            # Get Defense Adjustments
            def_mult = self.def_mults.get(def_, {}).get(ptype, 1.0)
            if def_ == home: def_mult *= (1 - hfa_impact)
            
            # --- CALL THE NEW HELPER FUNCTION ---
            yards, is_complete, is_turnover, desc_tag = self._resolve_play_outcome(
                off, def_, zone, ptype, stats, def_mult, hfa_impact, verbose
            )
            
            # If verbose, append the specific tag (Deep Ball, Breakaway) to the printout later
            if verbose and desc_tag:
                # We'll save this tag to print it in the verbose section below
                pass

            # --- CHECK TURNOVER ON DOWNS ---
            if down == 4 and yards < dist:
                is_turnover = True
                if verbose: print(f"   >>> TURNOVER ON DOWNS!")

            # --- CLOCK LOGIC (OOB, Stoppage & TIMEOUTS) ---
            is_oob = False
            if ptype == 'run' or (ptype == 'pass' and is_complete):
                oob_prob = self.profiles['oob_rates'].get(off, 0.15)
                if np.random.random() < oob_prob: is_oob = True
            
            clock_stops = False
            if ptype == 'pass' and not is_complete and yards >= 0:
                clock_stops = True 
            elif is_oob:
                if (1800 < clock <= 1920) or (clock <= 300 and phase == 'REG') or (phase == 'OT'):
                    clock_stops = True
            
            clock_running = not clock_stops
            
            # --- TIMEOUT LOGIC ---
            is_two_minute = time_left_in_half <= 120
            
            if not clock_stops and is_two_minute:
                if scores[def_] <= scores[off] and timeouts[def_] > 0:
                    timeouts[def_] -= 1
                    clock_stops = True
                    clock_running = False
                    if verbose: print(f"   >>> TIMEOUT {def_} ({timeouts[def_]} left)")
                
                elif scores[off] <= scores[def_] and timeouts[off] > 0:
                    timeouts[off] -= 1
                    clock_stops = True
                    clock_running = False
                    if verbose: print(f"   >>> TIMEOUT {off} ({timeouts[off]} left)")

            if clock_stops:
                time_consumed = self.profiles.get('play_duration', 6.0)
            else:
                pace_t = 'run'
                if ptype == 'pass':
                    if is_complete: pace_t = 'pass_complete'
                    elif yards < 0: pace_t = 'sack'
                    else: pace_t = 'pass_incomplete'
                
                time_consumed = self.profiles['pace'].get((off, pace_t), 35.0)
                
                # FIX: Cap standard plays to prevent "huddle drift"
                # If the data has a weird outlier (like an injury play taking 90 seconds), 
                # it ruins the sim average.
                if pace_t == 'run' or pace_t == 'pass_complete':
                    time_consumed = min(time_consumed, 40) # Cap at 40s (play clock)
                elif pace_t == 'pass_incomplete' or is_oob:
                    time_consumed = min(time_consumed, 10) # Quick stoppage
                if phase == 'REG' and clock < 300:
                    if diff < 0: time_consumed = min(time_consumed, 15)
                    if diff > 0: time_consumed = max(time_consumed, 40)
                if phase == 'OT': time_consumed = min(time_consumed, 25)

            # HURRY UP LOGIC
            # If inside 2 mins of 2nd or 4th quarter and trailing or tied (or just wanting to score before half)
            is_end_of_half = (phase == 'REG' and 1800 < clock <= 1920) or (clock <= 120)
            trying_to_score = (scores[off] <= scores[def_] + 8) or (1800 < clock <= 1920) # Always try to score before half

            if is_end_of_half and trying_to_score and not clock_stops:
                # In hurry up, plays take 12-15 seconds total, not 35
                if is_complete or ptype == 'run':
                    time_consumed = min(time_consumed, 14) 

            clock -= time_consumed

            if verbose:
                loc = format_field(yardline, off)
                desc = f"Run {yards}" if ptype=='run' else (f"Pass {yards}" if is_complete else "Pass Inc")
                if yards < 0 and ptype == 'pass' and not is_complete: desc = "SACK"
                if is_turnover: desc += " TURNOVER"
                if is_oob: desc += " (OOB)"
                print(f"[{format_clock(clock, phase)}] {off} {down}&{dist} @ {loc} | {desc}")

            if is_turnover:
                 clock_running = False 
                 
                 # --- FIX: DEFENSIVE TOUCHDOWN LOGIC ---
                 # Approx 8% of turnovers result in a defensive score
                 if np.random.random() < 0.08:
                     scores[def_] += 6
                     if verbose: print(f"   >>> DEFENSIVE TOUCHDOWN (PICK-6/FUMBLE-6) {def_}!")
                     self._attempt_pat(def_, off, scores, clock, phase, raw_wind, verbose)
                     
                     # Kickoff logic
                     possession = off # Offense gets ball back
                     yardline = 32
                     down, dist = 1, 10
                     continue # Skip the rest, start new drive
                 # --------------------------------------

                 if phase == 'OT' and scores[off] == scores[def_]:
                      if verbose: print("   >>> OT: Turnover. Next score wins.")
                 
                 # Standard Turnover
                 possession = def_
                 yardline = 100 - (yardline + yards)
                 # Add variance to turnover return (sometimes they return it 20 yards)
                 return_yards = int(np.random.exponential(5)) # Avg 5 yard return
                 yardline += return_yards
                 yardline = min(yardline, 99) # Don't go past goal line
                 
                 down, dist = 1, 10
                 clock -= 10
                 if phase == 'OT': ot_drive_count += 1
                 continue

            yardline += yards
            dist -= yards
            
            if yardline >= 100:
                scores[off] += 6 
                if verbose: print(f"   >>> TOUCHDOWN {off}!")
                
                self._attempt_pat(off, def_, scores, clock, phase, raw_wind if not is_dome else 0, verbose)
                clock_running = False 

                if phase == 'OT':
                    if ot_drive_count == 0:
                        if verbose: print(f"   >>> OT: {def_} gets a chance to match!")
                    else:
                        if scores[off] > scores[def_]:
                            game_active = False
                            if verbose: print(f"   >>> OVERTIME WINNER: {off}!")
                            break
                        elif scores[off] == scores[def_]:
                            if verbose: print(f"   >>> OT: Game Tied. Next Score Wins!")

                possession = def_
                new_start = self._get_kickoff_start(possession)
                
                if new_start >= 100:
                    # KICK RETURN TD!
                    scores[possession] += 6
                    if verbose: print(f"   >>> KICKOFF RETURN TOUCHDOWN {possession}!")
                    self._attempt_pat(possession, off, scores, clock, phase, raw_wind, verbose)
                    # Kick it right back to the other team
                    possession = off 
                    yardline = 30 
                    continue # Skip to next iteration
                
                yardline = new_start
                down, dist = 1, 10
                if phase == 'OT': ot_drive_count += 1
            elif dist <= 0:
                down = 1
                dist = 10
            else:
                down += 1
        
        return {'Home': home, 'Away': away, 'Home_Score': scores[home], 'Away_Score': scores[away], 
                'Margin': scores[away] - scores[home]}

# --- MAIN ---
if __name__ == "__main__":
    sim = AdvancedNFLSimulator()
    sim.load_data() 
    
    # Configuration
    away_team = 'CAR'
    home_team = 'TB'
    
    # Run Simulation
    df_res = sim.simulate_matchup(home_team, away_team, wind_speed=5, is_dome=False, print_sample_game=True)
    
    if not df_res.empty:
        # Calculate Total Score
        df_res['Total'] = df_res['Home_Score'] + df_res['Away_Score']
        
        print(f"\nAGGREGATE REPORT ({SIMULATIONS} Sims) | {away_team} @ {home_team}")
        print("=" * 115)
        print(f"{'METRIC':<15} | {'AVG':<8} | {'MEDIAN':<8} | {'MIN':<8} | {'MAX':<8} | {'VARIANCE':<8} | {'VOLATILITY'}")
        print("-" * 115)

        def get_variance_label(val, metric_type='combined'):
            """
            Returns a qualitative label for variance based on NFL standards.
            metric_type: 'combined' (Spread/Total) or 'team' (Single Team Score)
            """
            if metric_type == 'combined':
                # Thresholds for Spreads/Totals (Standard Dev ~13-14)
                if val < 160: return "Low"
                if val < 185: return "Med-Low"
                if val < 215: return "Medium"
                if val < 250: return "Med-High"
                return "High"
            else:
                # Thresholds for Single Team Scores (Standard Dev ~10-11)
                if val < 80:  return "Low"
                if val < 100: return "Med-Low"
                if val < 125: return "Medium"
                if val < 150: return "Med-High"
                return "High"

        def print_metric_row(label, data, metric_type='combined'):
            avg = data.mean()
            med = data.median()
            low = data.min()
            high = data.max()
            var = data.var()
            
            # Get the qualitative label
            vol_label = get_variance_label(var, metric_type)
            
            print(f"{label:<15} | {avg:<8.2f} | {med:<8.1f} | {low:<8.1f} | {high:<8.1f} | {var:<8.2f} | {vol_label}")

        # 1. Spread (Margin) -> Use 'combined' scale
        print_metric_row("Spread (Home TM)", df_res['Margin'], 'combined')
        
        # 2. Total -> Use 'combined' scale
        print_metric_row("Total Points", df_res['Total'], 'combined')
        
        # 3. Home Team Total -> Use 'team' scale
        print_metric_row(f"{home_team} Score", df_res['Home_Score'], 'team')
        
        # 4. Away Team Total -> Use 'team' scale
        print_metric_row(f"{away_team} Score", df_res['Away_Score'], 'team')
        
        print("=" * 115)
