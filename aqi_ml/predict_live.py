import pandas as pd
import joblib
import time
from datetime import timedelta
from supabase import create_client

# --- CONFIGURATION ---
SUPABASE_URL = "https://qdvkprgftjcjncjweymr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFkdmtwcmdmdGpjam5jandleW1yIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM2OTI1OTIsImV4cCI6MjA3OTI2ODU5Mn0.DKNp86ck7QviJlE46RuTajhfeB8KjibxmFhZ3kHXqQs"
TABLE_NAME = "realtime_data"

AQI_COL = "aqi"
TEMP_COL = "temperature"
HUM_COL = "humidity"
TIME_COL = "updated_at"

print("⏳ Loading Multi-Output Model...")
try:
    model = joblib.load('aqi_multi_model.pkl')
    feature_order = joblib.load('model_features.pkl')
except:
    print("❌ Error: Multi-output model not found. Please run the updated train_model.py first!")
    exit()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("📡 Live Trend Generator Online (Console Mode)...")

while True:
    try:
        # 1. Fetch recent data
        response = supabase.table(TABLE_NAME).select(f"{AQI_COL},{TEMP_COL},{HUM_COL},{TIME_COL}")\
            .order(TIME_COL, desc=True).limit(2000).execute()
        
        df_live = pd.DataFrame(response.data)
        df_live['timestamp'] = pd.to_datetime(df_live[TIME_COL])
        df_live.set_index('timestamp', inplace=True)
        df_live.sort_index(inplace=True)

        # 2. Resample
        df_resampled = df_live[[AQI_COL, TEMP_COL, HUM_COL]].resample('2min').mean().dropna()

        # 3. Check History (Need 15 steps)
        if len(df_resampled) >= 15:
            # Prepare Input
            recent_15 = df_resampled.tail(15)
            input_data = {}
            for i in range(1, 16):
                row_idx = -i 
                input_data[f'{AQI_COL}_Lag_{i}'] = recent_15[AQI_COL].iloc[row_idx]
                input_data[f'{TEMP_COL}_Lag_{i}'] = recent_15[TEMP_COL].iloc[row_idx]
                input_data[f'{HUM_COL}_Lag_{i}'] = recent_15[HUM_COL].iloc[row_idx]

            input_df = pd.DataFrame([input_data])[feature_order]

            # 4. PREDICT 30 VALUES
            # Returns a list of 30 AQI values
            predictions = model.predict(input_df)[0] 
            
            # 5. PRINT THE TREND (Disappear and New Come)
            # We print a lot of new lines to "clear" the visual space or just a big separator
            print("\n" * 2) 
            print("========================================")
            print(f"🌲 NEW 1-HOUR FORECAST GENERATED")
            print(f"Current Time: {pd.Timestamp.now().strftime('%H:%M:%S')}")
            print("========================================")
            print(f"{'TIME':<15} | {'PREDICTED AQI':<15}")
            print("-" * 35)

            current_time = pd.Timestamp.now()

            for i, pred_val in enumerate(predictions):
                # Calculate the future time for this specific dot
                minutes_ahead = (i + 1) * 2
                future_time = current_time + timedelta(minutes=minutes_ahead)
                time_str = future_time.strftime('%H:%M')
                
                print(f"{time_str:<15} | {pred_val:.2f}")

            print("========================================")
            print("Next update in 2 minutes...")

        else:
            print("⏳ Gathering data... Waiting for sensor history.")

    except Exception as e:
        print(f"⚠️ Error: {e}")

    # Wait 2 minutes
    time.sleep(120)