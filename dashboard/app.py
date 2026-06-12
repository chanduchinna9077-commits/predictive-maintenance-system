import streamlit as st
import pandas as pd
import joblib

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide"
)

# =====================================
# LOAD MODEL
# =====================================

try:
    model = joblib.load("../models/xgboost_model_v2.pkl")
except:
    model = joblib.load("models/xgboost_model_v2.pkl")

# =====================================
# TITLE
# =====================================

st.title("⚙️ Predictive Maintenance System")

st.markdown(
    """
Predict machine failures before they happen using Machine Learning.
"""
)

# =====================================
# TOP METRICS
# =====================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model Accuracy",
        "99.9%"
    )

with col2:
    st.metric(
        "Features",
        "16"
    )

with col3:
    st.metric(
        "Model",
        "XGBoost"
    )

# =====================================
# INPUTS
# =====================================

st.header("Machine Sensor Inputs")

col1, col2 = st.columns(2)

with col1:

    air_temp = st.number_input(
        "Air Temperature (K)",
        value=300.0
    )

    process_temp = st.number_input(
        "Process Temperature (K)",
        value=310.0
    )

    rpm = st.number_input(
        "Rotational Speed (RPM)",
        value=1500
    )

with col2:

    torque = st.number_input(
        "Torque (Nm)",
        value=40.0
    )

    tool_wear = st.number_input(
        "Tool Wear (Minutes)",
        value=100
    )

    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M"]
    )

# =====================================
# PREDICT BUTTON
# =====================================

if st.button("🚀 Predict Failure"):

    # =================================
    # FEATURE ENGINEERING
    # =================================

    temp_diff = process_temp - air_temp

    power = torque * rpm

    type_l = 1 if machine_type == "L" else 0
    type_m = 1 if machine_type == "M" else 0

    wear_medium = 0
    wear_high = 0

    if 75 <= tool_wear < 150:
        wear_medium = 1

    elif tool_wear >= 150:
        wear_high = 1

    # =================================
    # SENSOR SUMMARY
    # =================================

    st.subheader("📊 Current Sensor Readings")

    sensor_df = pd.DataFrame({
        "Sensor": [
            "Air Temperature",
            "Process Temperature",
            "RPM",
            "Torque",
            "Tool Wear"
        ],
        "Value": [
            air_temp,
            process_temp,
            rpm,
            torque,
            tool_wear
        ]
    })

    st.dataframe(
        sensor_df,
        use_container_width=True
    )

    # =================================
    # MODEL INPUT
    # =================================

    input_data = pd.DataFrame([{
        "Air_temperature_K": air_temp,
        "Process_temperature_K": process_temp,
        "Rotational_speed_rpm": rpm,
        "Torque_Nm": torque,
        "Tool_wear_min": tool_wear,
        "TWF": 0,
        "HDF": 0,
        "PWF": 0,
        "OSF": 0,
        "RNF": 0,
        "Type_L": type_l,
        "Type_M": type_m,
        "Temp_Diff": temp_diff,
        "Power": power,
        "Wear_Level_Medium": wear_medium,
        "Wear_Level_High": wear_high
    }])

    # =================================
    # PREDICTION
    # =================================

    prediction = model.predict(
        input_data
    )[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    # =================================
    # PREDICTION RESULT
    # =================================

    st.subheader("🤖 Failure Prediction")

    if prediction == 1:

        st.error(
            f"⚠️ Failure Risk Detected\n\nRisk Score: {probability:.2%}"
        )

    else:

        st.success(
            f"✅ Machine Healthy\n\nRisk Score: {probability:.2%}"
        )

    # =================================
    # RISK LEVEL
    # =================================

    st.subheader("📈 Risk Level")

    if probability < 0.20:

        st.success(
            "🟢 LOW RISK"
        )

    elif probability < 0.60:

        st.warning(
            "🟡 MEDIUM RISK"
        )

    else:

        st.error(
            "🔴 HIGH RISK"
        )

    st.progress(
        float(probability)
    )

    # =================================
    # MAINTENANCE RECOMMENDATIONS
    # =================================

    st.subheader(
        "🛠 Maintenance Recommendations"
    )

    recommendations = []

    if tool_wear >= 150:
        recommendations.append(
            "🔧 Replace Tool Soon"
        )

    if process_temp >= 320:
        recommendations.append(
            "🌡️ Check Cooling System"
        )

    if torque >= 70:
        recommendations.append(
            "⚙️ Inspect Bearings and Motor"
        )

    if rpm >= 2500:
        recommendations.append(
            "🚀 Check Rotational Components"
        )

    if len(recommendations) == 0:

        st.success(
            "✅ No Immediate Maintenance Required"
        )

    else:

        for recommendation in recommendations:
            st.warning(
                recommendation
            )

    # =================================
    # ENGINEERED FEATURES
    # =================================

    st.subheader(
        "⚡ Engineered Features"
    )

    feature_df = pd.DataFrame({
        "Feature": [
            "Temperature Difference",
            "Power"
        ],
        "Value": [
            temp_diff,
            power
        ]
    })

    st.dataframe(
        feature_df,
        use_container_width=True
    )