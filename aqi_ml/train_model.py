import pandas as pd
from supabase import create_client
from sklearn.ensemble import RandomForestRegressor
import joblib

# --- CONFIGURATION ---
SUPABASE_URL = "https://qdvkprgftjcjncjweymr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFkdmtwcmdmdGpjam5jandleW1yIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM2OTI1OTIsImV4cCI6MjA3OTI2ODU5Mn0.DKNp86ck7QviJlE46RuTajhfeB8KjibxmFhZ3kHXqQs"
TABLE_NAME = "realtime_data"

# COLUMN NAMES (Corrected for your database)
AQI_COL = "aqi"
TEMP_COL = "temperature"
HUM_COL = "humidity"
TIME_COL = "updated_at"  # <--- Important fix

print("⏳ Connecting to Supabase...")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 1. FETCH DATA
print("📥 Downloading raw data...")
query_cols = f"{AQI_COL}, {TEMP_COL}, {HUM_COL}, {TIME_COL}"
response = supabase.table(TABLE_NAME).select(query_cols).order(TIME_COL, desc=True).limit(70000).execute()

df = pd.DataFrame(response.data)
df['timestamp'] = pd.to_datetime(df[TIME_COL])
df.set_index('timestamp', inplace=True)
df.sort_index(inplace=True)

# 2. RESAMPLE TO 2-MINUTES
df_2min = df[[AQI_COL, TEMP_COL, HUM_COL]].resample('2min').mean().dropna()
print(f"✅ Data processed: {len(df_2min)} rows")

# 3. CREATE FEATURES (Inputs - Past 30 Mins)
feature_cols = []
lags = 15
for i in range(1, lags + 1):
    for col in [AQI_COL, TEMP_COL, HUM_COL]:
        col_name = f'{col}_Lag_{i}'
        df_2min[col_name] = df_2min[col].shift(i)
        feature_cols.append(col_name)

# 4. CREATE TARGETS (Outputs - Next 60 Mins = 30 Steps)
target_cols = []
# We want to predict t+1 (2mins), t+2 (4mins) ... up to t+30 (60mins)
for i in range(1, 31): 
    col_name = f'Step_{i}' 
    df_2min[col_name] = df_2min[AQI_COL].shift(-i)
    target_cols.append(col_name)

df_final = df_2min.dropna()

# 5. TRAIN MULTI-OUTPUT MODEL
print(f"🧠 Training to predict 30 future steps at once...")
X = df_final[feature_cols]
y = df_final[target_cols] 

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 6. SAVE (This creates the file you are missing!)
joblib.dump(model, 'aqi_multi_model.pkl')     
joblib.dump(feature_cols, 'model_features.pkl')
print("🎉 Multi-Output Model saved! NOW you can run predict_live.py")