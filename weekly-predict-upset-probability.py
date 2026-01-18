###WITHOUT DIVISIONAL, AWAY, AND THURSDAY NIGHT GAMES

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# --- CONFIGURATION ---
TRAIN_FILE = "nfl_games_with_schematic_data_2008_2024.csv"
TEST_FILE = "nfl_games_with_schematic_data_2025_2025.csv"
OUTPUT_FILE = "nfl_2025_matchup_upset_predictions.csv"

# --- UPDATED CATEGORIES ---
MATCHUP_CATEGORIES = [
    'Overall', 'Run', 'Pass', 'Pass_Deep', 'Pass_Short', 
    'Redzone', '3rd_Down', '1st_Down', 'Play_Action', 
    'Quick_Game_Proxy', 'Under_Pressure'
]

def calculate_mismatches(df):
    """
    Creates new columns representing the difference between Offense and Defense.
    Positive Value = Offense has the advantage.
    Negative Value = Defense has the advantage.
    """
    df = df.copy()
    
    print("Engineering Matchup Features...")
    
    for cat in MATCHUP_CATEGORIES:
        # Construct column names based on your file structure
        # Format in file: home_Off_Run_EPA_Pct, away_Def_Run_EPA_Pct
        
        h_off_col = f"home_Off_{cat}_EPA_Pct"
        a_def_col = f"away_Def_{cat}_EPA_Pct"
        
        a_off_col = f"away_Off_{cat}_EPA_Pct"
        h_def_col = f"home_Def_{cat}_EPA_Pct"
        
        # Check if columns exist before calculating (Crucial for 'Under_Pressure')
        if h_off_col in df.columns and a_def_col in df.columns:
            # 1. Home Offense vs Away Defense Matchup
            df[f'Matchup_HomeOff_{cat}'] = df[h_off_col] - df[a_def_col]
            
        if a_off_col in df.columns and h_def_col in df.columns:
            # 2. Away Offense vs Home Defense Matchup
            df[f'Matchup_AwayOff_{cat}'] = df[a_off_col] - df[h_def_col]

    return df

def run_matchup_analysis():
    # 1. Load Data
    try:
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    if test_df.empty:
        print("WARNING: Test file is empty. Cannot predict.")
        return

    # 2. Engineer Features (Calculate Mismatches)
    train_df = calculate_mismatches(train_df)
    test_df = calculate_mismatches(test_df)

    # 3. Define Features
    # We explicitly include our new 'Matchup_' columns
    mismatch_cols = [c for c in train_df.columns if c.startswith('Matchup_')]
    
    base_cols = ['spread_line', 'total_line', 'home_moneyline_decimal', 'away_moneyline_decimal', 
                 'home_rest_adv', 'away_rest_adv']
    
    feature_cols = base_cols + mismatch_cols
    
    # Filter for columns that actually exist in BOTH datasets
    feature_cols = [c for c in feature_cols if c in train_df.columns and c in test_df.columns]
    
    print(f"Training on {len(feature_cols)} features...")

    # 4. Prepare Data
    X_train = train_df[feature_cols].copy()
    y_train = train_df['Upset'].astype(int)
    X_test = test_df[feature_cols].copy()

    # Impute missing data
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # 5. Train Model
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, 
                                class_weight='balanced', random_state=42)
    rf.fit(X_train_imputed, y_train)

    # 6. Predict
    probs = rf.predict_proba(X_test_imputed)[:, 1]
    
    # 7. Format Results
    results = test_df.copy()
    results['Upset_Probability'] = probs
    
    # --- INTERPRETATION: IDENTIFY THE "BAD MATCHUP" ---
    def find_key_mismatch(row):
        spread = row['spread_line']
        if pd.isna(spread): return "Unknown"
        
        # If Home is Favorite (Spread > 0), we look for Away Advantages
        if spread > 0: 
            cols = [c for c in mismatch_cols if 'AwayOff' in c]
            if not cols: return "None"
            # Get values for this row
            vals = row[cols]
            best_cat = vals.idxmax()
            score = vals.max()
            if score > 30: 
                clean_cat = best_cat.replace('Matchup_AwayOff_', '')
                return f"Underdog Edge: {clean_cat} (+{score:.1f})"
            return "No Glaring Mismatch"
            
        # If Away is Favorite (Spread < 0), we look for Home Advantages
        elif spread < 0:
            cols = [c for c in mismatch_cols if 'HomeOff' in c]
            if not cols: return "None"
            vals = row[cols]
            best_cat = vals.idxmax()
            score = vals.max()
            if score > 30:
                clean_cat = best_cat.replace('Matchup_HomeOff_', '')
                return f"Underdog Edge: {clean_cat} (+{score:.1f})"
            return "No Glaring Mismatch"
        return "Pick'em"

    results['Key_Schematic_Edge'] = results.apply(find_key_mismatch, axis=1)

    # Output columns
    out_cols = ['week', 'away_team', 'home_team', 'spread_line', 'Upset_Probability', 'Key_Schematic_Edge']
    results = results.sort_values('Upset_Probability', ascending=False)
    
    results[out_cols].to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nSUCCESS: Predictions saved to {OUTPUT_FILE}")
    print("\n--- Top Potential Upsets & The Matchup to Watch ---")
    print(results[out_cols].head(5))
    
    # Feature Importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\n--- Top Factors Driving the Model ---")
    for f in range(min(10, len(feature_cols))):
        print(f"{f+1}. {feature_cols[indices[f]]} ({importances[indices[f]]:.4f})")

if __name__ == "__main__":
    run_matchup_analysis()
