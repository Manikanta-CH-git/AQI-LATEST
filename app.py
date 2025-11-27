import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
import traceback
import joblib
from datetime import timedelta

# ==================================================
# ☁ SUPABASE CONFIG (FROM secrets.toml)
# ==================================================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

# Initialize Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"Failed to initialize Supabase client. Check secrets configuration. Error: {e}")
    st.stop()


# ==================================================
# ⚙ PAGE SETTINGS
# ==================================================
st.set_page_config(page_title="AQI Dashboard", layout="wide")

# ==================================================
# 🌈 GLOBAL CSS
# ==================================================
st.markdown("""
<style>
    .aqi-bar-container { display: flex; height: 45px; border-radius: 10px; overflow: hidden; margin-top: 10px; }
    .seg { flex: 1; text-align: center; font-weight: bold; padding-top: 12px; color: white; font-family: sans-serif; font-size: 14px; }

    .good { background: #00e400; }
    .moderate { background: #ffff00; color: black !important; }
    .poor { background: #ff7e00; }
    .unhealthy { background: #ff0000; }
    .veryunhealthy { background: #8f3f97; }
    .hazardous { background: #7e0023; }

    .ticks { width: 100%; display: flex; justify-content: space-between; margin-top: 4px; font-size: 12px; color: #aaa; }
    .big-aqi-value { font-size: 48px; font-weight: 800; text-align: center; margin-top: 15px; transition: color 0.5s ease; }
    .status-text { font-size: 24px; text-align: center; margin-bottom: 10px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==================================================
# 📥 FETCH LATEST DATA
# ==================================================
def get_latest_data(table_name, limit=200):
    try:
        response = (
            supabase.table(table_name)
            .select("*")
            .order("id", desc=True)
            .limit(limit)
            .execute()
        )
        if hasattr(response, 'data') and response.data is not None:
            return response.data
        else:
            st.warning(f"Query to {table_name} succeeded, but 'data' field was empty or missing.")
            return []

    except Exception as e:
        st.error(f"🛑 Supabase Data Fetch Error from {table_name}")
        st.code(f"Error Type: {type(e).__name__}\nMessage: {e}\n\nTraceback:\n{traceback.format_exc()}", language="python")
        return []

# ==================================================
# 🧠 LOAD AI MODEL
# ==================================================
@st.cache_resource
def load_ai_model():
    try:
        # Assumes the 'aqi_ml' folder is in the same directory as app.py
        model = joblib.load('aqi_ml/aqi_multi_model.pkl')
        features = joblib.load('aqi_ml/model_features.pkl')
        return model, features
    except FileNotFoundError:
        return None, None

# ==================================================
# 🧭 SIDEBAR
# ==================================================
refresh_seconds = st.sidebar.slider("⏱ Auto Refresh (Seconds)", 2, 60, 5)
choice = st.sidebar.radio("📌 Select View", ["Current Data", "Stored Data", "Future AQI Forecasting"])

# ==================================================
# 🟢 LIVE MONITOR
# ==================================================
@st.fragment(run_every=refresh_seconds)
def show_live_monitor():
    rows = get_latest_data("realtime_data", 50)
    
    if not rows:
        st.info("Waiting for data from device (realtime_data), or check error message above...")
        return

    df = pd.DataFrame(rows)
    
    if "created_at" in df.columns:
        df.rename(columns={"created_at": "Timestamp"}, inplace=True)
    elif "updated_at" in df.columns:
        df.rename(columns={"updated_at": "Timestamp"}, inplace=True)
    else:
        st.error(f"Missing timestamp column. Available columns: {list(df.columns)}")
        return

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    
    try:
        df["Timestamp"] = df["Timestamp"].dt.tz_convert("Asia/Kolkata")
    except Exception:
        df["Timestamp"] = df["Timestamp"].dt.tz_localize(None)

    if df.empty:
        st.warning("No valid timestamp data available after cleanup.")
        return

    latest = df.iloc[0]

    try:
        aqi = int(latest["aqi"])
    except (ValueError, KeyError):
        st.error("Invalid or missing AQI value found in the latest data.")
        return
    
    temp = latest.get("temperature", "N/A")
    hum = latest.get("humidity", "N/A")

    if aqi <= 50: status, color = "Good", "#00e400"
    elif aqi <= 100: status, color = "Moderate", "#ffff00"
    elif aqi <= 150: status, color = "Poor", "#ff7e00"
    elif aqi <= 200: status, color = "Unhealthy", "#ff0000"
    elif aqi <= 300: status, color = "Very Unhealthy", "#8f3f97"
    else: status, color = "Hazardous", "#7e0023"

    st.title("🌍 Live AQI Monitoring")

    col1, col2, col3 = st.columns(3)
    col1.metric("AQI", aqi)
    col2.metric("Temperature (°C)", temp)
    col3.metric("Humidity (%)", hum)

    st.caption(f"Last Updated: {latest['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

    st.markdown(f"""
    <div class="status-text">Current Status: {status}</div>
    <div class="aqi-bar-container">
        <div class="seg good">Good</div>
        <div class="seg moderate">Moderate</div>
        <div class="seg poor">Poor</div>
        <div class="seg unhealthy">Unhealthy</div>
        <div class="seg veryunhealthy">Very Unhealthy</div>
        <div class="seg hazardous">Hazardous</div>
    </div>
    <div class="ticks">
        <span>0</span><span>50</span><span>100</span><span>150</span><span>200</span><span>300</span><span>300+</span>
    </div>
    <div class="big-aqi-value" style="color:{color};">{aqi} AQI</div>
    """, unsafe_allow_html=True)

    st.subheader("📈 Live AQI Trend")
    df_sorted = df.sort_values("Timestamp")
    fig = px.line(df_sorted, x="Timestamp", y="aqi", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# 📁 STORED DATA PAGE
# ==================================================
def show_history():
    st.title("📊 Historical AQI Data")
    rows = get_latest_data("sensor_data", 1000)
    
    if not rows:
        st.warning("No data available in 'sensor_data' table.")
        return

    df = pd.DataFrame(rows)
    
    if "created_at" in df.columns:
        df.rename(columns={"created_at": "Timestamp"}, inplace=True)
    elif "updated_at" in df.columns:
        df.rename(columns={"updated_at": "Timestamp"}, inplace=True)
    else:
        st.error(f"Missing timestamp column. Available columns: {list(df.columns)}")
        return

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    
    try:
        df["Timestamp"] = df["Timestamp"].dt.tz_convert("Asia/Kolkata")
    except Exception:
        df["Timestamp"] = df["Timestamp"].dt.tz_localize(None)

    if df.empty:
        st.warning("No valid timestamp data available after cleanup.")
        return

    df_sorted = df.sort_values("Timestamp")

    st.subheader("📈 AQI, Temperature & Humidity Trends")
    fig = px.line(df_sorted, x="Timestamp", y=["aqi", "temperature", "humidity"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📄 Data Table")
    st.dataframe(df_sorted.set_index("Timestamp"), use_container_width=True)

# ==================================================
# 🔮 FUTURE PREDICTION (UPDATED)
# ==================================================
def show_future():
    st.title("🔮 Future AQI Predictions (1-Hour)")
    st.markdown("Uses Random Forest Regression to predict the next 30 intervals (2 mins each).")
    
    model, feature_order = load_ai_model()

    if model is None:
        st.error("⚠️ Model files not found! Make sure `aqi_ml/aqi_multi_model.pkl` and `model_features.pkl` exist.")
        return

    # Button to Trigger Prediction
    if st.button("Generate 1-Hour Forecast", type="primary"):
        with st.spinner("Fetching recent data and calculating trends..."):
            
            # 1. Fetch data specifically for prediction logic
            # We use 'updated_at' because that was the column name in your training
            TABLE_NAME = "realtime_data"
            AQI_COL, TEMP_COL, HUM_COL, TIME_COL = "aqi", "temperature", "humidity", "updated_at"
            
            response = supabase.table(TABLE_NAME).select(f"{AQI_COL},{TEMP_COL},{HUM_COL},{TIME_COL}")\
                .order(TIME_COL, desc=True).limit(1000).execute()
            
            df_live = pd.DataFrame(response.data)
            df_live['timestamp'] = pd.to_datetime(df_live[TIME_COL])
            df_live.set_index('timestamp', inplace=True)
            df_live.sort_index(inplace=True)

            # 2. Resample to 2-Minutes (Must match training!)
            df_resampled = df_live[[AQI_COL, TEMP_COL, HUM_COL]].resample('2min').mean().dropna()

            if len(df_resampled) >= 15:
                # 3. Prepare Input
                recent_15 = df_resampled.tail(15)
                input_data = {}
                for i in range(1, 16):
                    row_idx = -i 
                    input_data[f'{AQI_COL}_Lag_{i}'] = recent_15[AQI_COL].iloc[row_idx]
                    input_data[f'{TEMP_COL}_Lag_{i}'] = recent_15[TEMP_COL].iloc[row_idx]
                    input_data[f'{HUM_COL}_Lag_{i}'] = recent_15[HUM_COL].iloc[row_idx]

                input_df = pd.DataFrame([input_data])[feature_order]

                # 4. Predict
                predictions = model.predict(input_df)[0]

                # 5. Visualize
                future_times = [pd.Timestamp.now() + timedelta(minutes=(i+1)*2) for i in range(30)]
                
                forecast_df = pd.DataFrame({
                    "Time": future_times,
                    "Predicted AQI": predictions
                })
                
                # METRICS ROW
                m1, m2, m3 = st.columns(3)
                curr_aqi = recent_15[AQI_COL].iloc[-1]
                max_pred = max(predictions)
                min_pred = min(predictions)
                


                # CHART
                st.subheader("📈 Forecast Trend")
                fig = px.line(forecast_df, x="Time", y="Predicted AQI", markers=True, title="Next 60 Minutes Projection")
                fig.update_traces(line_color='#00FF00')
                st.plotly_chart(fig, use_container_width=True)

                # DATA TABLE
                st.subheader("📋 Forecast Details")
                st.dataframe(forecast_df.set_index("Time"), use_container_width=True)
                
            else:
                st.warning(f"⏳ Not enough continuous history to predict. Found {len(df_resampled)}/15 intervals (Need 30 mins of data).")

# ==================================================
# ROUTING
# ==================================================
if choice == "Current Data":
    show_live_monitor()
elif choice == "Stored Data":
    show_history()
elif choice == "Future AQI Forecasting":
    show_future()