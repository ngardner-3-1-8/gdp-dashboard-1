import pandas as pd
import nflreadpy as nfl
import os
from datetime import datetime

def update_annual_schedule():
    # 1. Determine the Year (May 15 is always for the UPCOMING season)
    target_year = datetime.now().year
    print(f"Fetching official NFL schedule for {target_year}...")

    # 2. Load schedule using nflreadpy
    # This replaces all the BeautifulSoup and requests code
    try:
        schedule_raw = nfl.load_schedules([target_year])
        schedule_df = schedule_raw.to_pandas()
        
        if schedule_df.empty:
            print(f"⚠️ Warning: No schedule found for {target_year} yet.")
            return

        # 3. Create output directory if it doesn't exist
        output_dir = "nfl-schedules"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 4. Save to CSV
        file_path = f"{output_dir}/schedule_{target_year}.csv"
        schedule_df.to_csv(file_path, index=False)
        
        print(f"✅ Success! {target_year} schedule saved to {file_path}")
        print(f"Total Games Found: {len(schedule_df)}")

    except Exception as e:
        print(f"❌ Error fetching schedule: {e}")

if __name__ == "__main__":
    update_annual_schedule()
