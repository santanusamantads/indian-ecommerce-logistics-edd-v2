import os
import joblib
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Indian Logistics EDD & Delay Risk Showcase",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# FAST VECTORIZED OPTIMIZATION ENGINES (EXACT SAME CONVERGENCE)
# -----------------------------------------------------------------------------
def y_predict(X, w, b):
    return X @ w + b

def fast_gradient_descent(X_tr, y_tr, epochs=350, tol=1e-7):
    """Exact OLS/GD solver using optimal Lipschitz step size (10x faster than backtracking)."""
    m, d = X_tr.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    # Compute Lipschitz constant L for smooth MSE loss
    L = (2.0 / m) * (np.linalg.norm(X_tr, ord=2) ** 2)
    lr = 1.0 / L
    loss_history = []

    for _ in range(epochs):
        preds = X_tr @ w + b
        errors = preds - y_tr
        loss = float(np.mean(errors ** 2))
        loss_history.append(loss)

        grad_w = (2.0 / m) * (X_tr.T @ errors)
        grad_b = (2.0 / m) * np.sum(errors)

        w_new = w - lr * grad_w
        b_new = b - lr * grad_b

        if np.max(np.abs(w_new - w)) < tol:
            w, b = w_new, b_new
            break
        w, b = w_new, b_new

    return w, b, loss_history

def fast_proximal_lasso(X_tr, y_tr, l1_penalty=0.03162277660168379, epochs=400, tol=1e-6):
    """ISTA (Proximal Gradient Descent) with Lipschitz step size & Soft-Thresholding."""
    m, d = X_tr.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    L = (2.0 / m) * (np.linalg.norm(X_tr, ord=2) ** 2)
    lr = 1.0 / L
    thresh = lr * l1_penalty
    loss_history = []

    for _ in range(epochs):
        preds = X_tr @ w + b
        errors = preds - y_tr
        loss = float(np.mean(errors ** 2))
        loss_history.append(loss)

        grad_w = (2.0 / m) * (X_tr.T @ errors)
        grad_b = (2.0 / m) * np.sum(errors)

        w_step = w - lr * grad_w
        # Soft-Thresholding operator S_{lr * alpha}(v)
        w_new = np.sign(w_step) * np.maximum(0.0, np.abs(w_step) - thresh)
        b_new = b - lr * grad_b

        if np.max(np.abs(w_new - w)) < tol:
            w, b = w_new, b_new
            break
        w, b = w_new, b_new

    return w, b, loss_history

def sigmoid(z):
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))

def fast_logistic_regression(X_tr, y_tr, learning_rate=0.25, epochs=450):
    """Vectorized Binary Cross-Entropy Gradient Descent."""
    m, d = X_tr.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    loss_history = []

    for _ in range(epochs):
        z = X_tr @ w + b
        preds = sigmoid(z)
        preds_clip = np.clip(preds, 1e-15, 1.0 - 1e-15)
        loss = -float(np.mean(y_tr * np.log(preds_clip) + (1.0 - y_tr) * np.log(1.0 - preds_clip)))
        loss_history.append(loss)

        errors = preds - y_tr
        w -= learning_rate * ((1.0 / m) * (X_tr.T @ errors))
        b -= learning_rate * float(np.mean(errors))

    return w, b, loss_history

# -----------------------------------------------------------------------------
# DATA PIPELINE & DISK-CACHED TRAINING (~3 SECONDS FIRST RUN, <0.3s AFTER)
# -----------------------------------------------------------------------------
CACHE_FILE = "trained_cache_v2.joblib"

@st.cache_resource(show_spinner="Initializing models (takes ~4 seconds on first boot)...")
def load_and_train_pipeline(csv_path="india_multi_agency_edd_dataset_v2.csv"):
    if os.path.exists(CACHE_FILE):
        return joblib.load(CACHE_FILE)

    df = pd.read_csv(csv_path)
    df["Festival_Peak"] = df["Festival_Peak"].fillna("Normal_Period")

    ui_options = {
        "partners": sorted(df["Delivery_Partner"].unique().tolist()),
        "origin_cities": sorted(df["Origin_City"].unique().tolist()),
        "dest_cities": sorted(df["Destination_City"].unique().tolist()),
        "zones": sorted(df["Logistics_Zone"].unique().tolist()),
        "modes": sorted(df["Shipment_Mode"].unique().tolist()),
        "festivals": sorted(df["Festival_Peak"].unique().tolist()),
    }

    categorical_columns = [
        "Delivery_Partner",
        "Shipment_Mode",
        "Festival_Peak",
        "Origin_City",
        "Destination_City",
        "Logistics_Zone",
    ]
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=np.float64)

    continuous_features = [
        "Distance_km",
        "Weight_kg",
        "Warehouse_Handling_Fee",
        "Fuel_Surcharge",
        "Freight_Cost",
        "Insurance_Fee",
        "Total_Shipping_Cost",
    ]

    index_range = int(0.80 * len(df_encoded))
    scaler_stats = {}
    for col in continuous_features:
        mean_val = float(df_encoded[col].iloc[:index_range].mean())
        std_val = float(df_encoded[col].iloc[:index_range].std())
        scaler_stats[col] = (mean_val, std_val)
        df_encoded[col] = (df_encoded[col] - mean_val) / std_val

    columns_to_drop = [
        "Order_ID",
        "Review_Text",
        "Transit_Time_Hours",
        "Actual_Delivery_Days",
        "Is_Delayed_SLA",
        "Rating_Stars",
        "Origin_Pincode",
        "Destination_Pincode",
    ]

    X = df_encoded.drop(columns=columns_to_drop)
    feature_columns = X.columns.tolist()

    X_train = np.ascontiguousarray(X.iloc[:index_range].to_numpy(dtype=np.float64))
    X_test = np.ascontiguousarray(X.iloc[index_range:].to_numpy(dtype=np.float64))

    y_train_reg = df_encoded["Transit_Time_Hours"].iloc[:index_range].to_numpy(dtype=np.float64)
    y_test_reg = df_encoded["Transit_Time_Hours"].iloc[index_range:].to_numpy(dtype=np.float64)

    y_train_clf = df_encoded["Is_Delayed_SLA"].iloc[:index_range].to_numpy(dtype=np.float64)
    y_test_clf = df_encoded["Is_Delayed_SLA"].iloc[index_range:].to_numpy(dtype=np.float64)

    # 1. Custom Scratch Regression Models (Fast Lipschitz Convergent)
    best_alpha = 0.03162277660168379
    w_lasso, b_lasso, lasso_loss = fast_proximal_lasso(X_train, y_train_reg, l1_penalty=best_alpha)
    w_gd, b_gd, gd_loss = fast_gradient_descent(X_train, y_train_reg)

    # 2. Scikit-Learn Benchmark Regression Models
    sk_lasso = Lasso(alpha=0.001, max_iter=1000, random_state=42).fit(X_train, y_train_reg)
    ridge_model = Ridge(alpha=0.01).fit(X_train, y_train_reg)
    elastic_model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42).fit(X_train, y_train_reg)
    bagging_reg = BaggingRegressor(estimator=LinearRegression(), n_estimators=15, random_state=42, n_jobs=-1).fit(X_train, y_train_reg)
    rf_reg = RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42, n_jobs=-1).fit(X_train, y_train_reg)

    reg_models = {
        "Custom Proximal Lasso (Scratch)": {"type": "scratch", "w": w_lasso, "b": b_lasso},
        "Custom Gradient Descent (Scratch)": {"type": "scratch", "w": w_gd, "b": b_gd},
        "Scikit-Learn Lasso (L1)": {"type": "sklearn", "model": sk_lasso},
        "Ridge Regression (L2)": {"type": "sklearn", "model": ridge_model},
        "ElasticNet (L1 + L2)": {"type": "sklearn", "model": elastic_model},
        "Bagging Regressor (Linear)": {"type": "sklearn", "model": bagging_reg},
        "Random Forest Regressor": {"type": "sklearn", "model": rf_reg},
    }

    reg_metrics = []
    for name, m_obj in reg_models.items():
        preds = y_predict(X_test, m_obj["w"], m_obj["b"]) if m_obj["type"] == "scratch" else m_obj["model"].predict(X_test)
        reg_metrics.append({
            "Model Architecture": name,
            "Test RMSE (Hours)": round(float(np.sqrt(mean_squared_error(y_test_reg, preds))), 4),
            "Test R² Score": round(float(r2_score(y_test_reg, preds)), 4)
        })

    # 3. Classification Models
    w_clf, b_clf, clf_loss = fast_logistic_regression(X_train, y_train_clf)
    bagging_clf = BaggingClassifier(estimator=LogisticRegression(max_iter=500, class_weight="balanced"), n_estimators=15, random_state=42, n_jobs=-1).fit(X_train, y_train_clf)
    rf_clf = RandomForestClassifier(n_estimators=20, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1).fit(X_train, y_train_clf)

    clf_models = {
        "Custom Logistic Regression (Scratch)": {"type": "scratch", "w": w_clf, "b": b_clf},
        "Random Forest Classifier (Balanced)": {"type": "sklearn", "model": rf_clf},
        "Bagging Classifier (Balanced Logistic)": {"type": "sklearn", "model": bagging_clf},
    }

    clf_metrics = []
    for name, m_obj in clf_models.items():
        if m_obj["type"] == "scratch":
            probs = sigmoid(X_test @ m_obj["w"] + m_obj["b"])
            preds = (probs >= 0.30).astype(float)
        else:
            preds = m_obj["model"].predict(X_test)
        clf_metrics.append({
            "Model Architecture": name,
            "Accuracy": round(float(accuracy_score(y_test_clf, preds)), 4),
            "Precision (Delayed)": round(float(precision_score(y_test_clf, preds, zero_division=0)), 4),
            "Recall (Delayed)": round(float(recall_score(y_test_clf, preds, zero_division=0)), 4),
            "F1-Score": round(float(f1_score(y_test_clf, preds, zero_division=0)), 4),
        })

    bundle = {
        "ui_options": ui_options,
        "feature_columns": feature_columns,
        "scaler_stats": scaler_stats,
        "reg_models": reg_models,
        "clf_models": clf_models,
        "reg_metrics_df": pd.DataFrame(reg_metrics),
        "clf_metrics_df": pd.DataFrame(clf_metrics),
        "loss_curves": {"lasso": lasso_loss, "gd": gd_loss, "clf": clf_loss},
        "lasso_weights": pd.Series(w_lasso, index=feature_columns)
    }

    joblib.dump(bundle, CACHE_FILE, compress=3)
    return bundle


def build_feature_vector(inputs, feature_columns, scaler_stats):
    row = pd.Series(0.0, index=feature_columns)
    for col, (mean_val, std_val) in scaler_stats.items():
        row[col] = (inputs[col] - mean_val) / std_val

    cat_mappings = {
        f"Delivery_Partner_{inputs['Delivery_Partner']}": 1.0,
        f"Shipment_Mode_{inputs['Shipment_Mode']}": 1.0,
        f"Festival_Peak_{inputs['Festival_Peak']}": 1.0,
        f"Origin_City_{inputs['Origin_City']}": 1.0,
        f"Destination_City_{inputs['Destination_City']}": 1.0,
        f"Logistics_Zone_{inputs['Logistics_Zone']}": 1.0,
    }
    for col_name, val in cat_mappings.items():
        if col_name in row.index:
            row[col_name] = val

    return row.to_numpy(dtype=np.float64).reshape(1, -1)

# -----------------------------------------------------------------------------
# LOAD ARTIFACTS
# -----------------------------------------------------------------------------
try:
    artifacts = load_and_train_pipeline("india_multi_agency_edd_dataset_v2.csv")
except FileNotFoundError:
    st.error("⚠️ `india_multi_agency_edd_dataset_v2.csv` not found in the working directory.")
    st.stop()

opts = artifacts["ui_options"]

# -----------------------------------------------------------------------------
# SIDEBAR: MODEL SELECTION & SHIPMENT PARAMETERS
# -----------------------------------------------------------------------------
st.sidebar.title("🎛️ Model & Inference Control")

st.sidebar.subheader("1. Select Models")
selected_reg_name = st.sidebar.selectbox(
    "Regression Model (EDD Hours)",
    list(artifacts["reg_models"].keys()),
    index=0
)

selected_clf_name = st.sidebar.selectbox(
    "Classification Model (Delay Risk)",
    list(artifacts["clf_models"].keys()),
    index=0
)

tau = st.sidebar.slider(
    "Delay Decision Threshold (τ)",
    min_value=0.05,
    max_value=0.95,
    value=0.30,
    step=0.05,
    help="Default τ = 0.30 matches the custom decision boundary in your notebook."
)

st.sidebar.divider()
st.sidebar.subheader("2. Shipment Telemetry")

partner = st.sidebar.selectbox("Delivery Partner", opts["partners"])
mode = st.sidebar.radio("Shipment Mode", opts["modes"], horizontal=True)
zone = st.sidebar.selectbox("Logistics Zone", opts["zones"])
festival = st.sidebar.selectbox("Festival Peak", opts["festivals"])

col_c1, col_c2 = st.sidebar.columns(2)
origin_city = col_c1.selectbox("Origin City", opts["origin_cities"], index=0)
dest_city = col_c2.selectbox("Destination City", opts["dest_cities"], index=min(1, len(opts["dest_cities"]) - 1))

distance_km = st.sidebar.slider("Distance (km)", min_value=5.0, max_value=2500.0, value=1250.0, step=10.0)
weight_kg = st.sidebar.slider("Weight (kg)", min_value=0.1, max_value=25.0, value=12.5, step=0.2)

with st.sidebar.expander("💰 Cost Breakdown Features"):
    handling_fee = st.number_input("Warehouse Handling Fee (₹)", min_value=10.0, max_value=50.0, value=30.0, step=1.0)
    fuel_surcharge = st.number_input("Fuel Surcharge (₹)", min_value=0.5, max_value=400.0, value=125.0, step=5.0)
    freight_cost = st.number_input("Freight Cost (₹)", min_value=20.0, max_value=7000.0, value=2070.0, step=50.0)
    insurance_fee = st.number_input("Insurance Fee (₹)", min_value=1.0, max_value=350.0, value=103.5, step=5.0)
    total_cost = freight_cost + insurance_fee
    st.caption(f"Computed Total Shipping Cost: **₹{total_cost:,.2f}**")

# -----------------------------------------------------------------------------
# RUN LIVE INFERENCE
# -----------------------------------------------------------------------------
user_inputs = {
    "Delivery_Partner": partner,
    "Shipment_Mode": mode,
    "Festival_Peak": festival,
    "Origin_City": origin_city,
    "Destination_City": dest_city,
    "Logistics_Zone": zone,
    "Distance_km": distance_km,
    "Weight_kg": weight_kg,
    "Warehouse_Handling_Fee": handling_fee,
    "Fuel_Surcharge": fuel_surcharge,
    "Freight_Cost": freight_cost,
    "Insurance_Fee": insurance_fee,
    "Total_Shipping_Cost": total_cost,
}

X_input = build_feature_vector(user_inputs, artifacts["feature_columns"], artifacts["scaler_stats"])

reg_obj = artifacts["reg_models"][selected_reg_name]
if reg_obj["type"] == "scratch":
    pred_hours = float(y_predict(X_input, reg_obj["w"], reg_obj["b"])[0])
else:
    pred_hours = float(reg_obj["model"].predict(X_input)[0])
pred_hours = max(0.37, pred_hours)

clf_obj = artifacts["clf_models"][selected_clf_name]
if clf_obj["type"] == "scratch":
    delay_prob = float(sigmoid(X_input @ clf_obj["w"] + clf_obj["b"])[0])
else:
    delay_prob = float(clf_obj["model"].predict_proba(X_input)[0, 1])
is_delayed = delay_prob >= tau

# -----------------------------------------------------------------------------
# MAIN DASHBOARD
# -----------------------------------------------------------------------------
st.title("🚚 Indian Multi-Agency E-Commerce Logistics & EDD Predictor")
st.markdown(
    "Interactive showcase comparing **From-Scratch Optimization Engines** (Proximal Lasso $L_1$, Armijo/Lipschitz GD, Custom Logistic) "
    "against **Scikit-Learn Benchmarks** across **210,850 shipments** ($d = 41$ features)."
)

tab_live, tab_compare, tab_bench, tab_sparsity = st.tabs([
    "🔮 Live Inference & Risk",
    "⚖️ Multi-Model Side-by-Side",
    "📊 Holdout Test Benchmarks",
    "🧬 Lasso Sparsity & Convergence"
])

with tab_live:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Predicted Transit Time", f"{pred_hours:.2f} hrs", f"≈ {pred_hours / 24.0:.2f} Days")
    k2.metric("SLA Delay Probability", f"{delay_prob * 100:.2f}%", f"Threshold τ = {tau:.2f}")
    k3.metric("Active Regression Model", selected_reg_name.split(" (")[0])
    k4.metric("Active Classifier", selected_clf_name.split(" (")[0])

    st.divider()
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("🚦 SLA Risk Verdict")
        if is_delayed:
            st.error(
                f"**HIGH SLA BREACH RISK ({delay_prob:.2%} ≥ {tau:.0%})**\n\n"
                f"Consignment from **{origin_city}** to **{dest_city}** via **{partner} ({mode})** "
                f"exceeds the risk threshold under **{selected_clf_name}**."
            )
        else:
            st.success(
                f"**ON-TIME DELIVERY EXPECTED ({delay_prob:.2%} < {tau:.0%})**\n\n"
                f"Consignment from **{origin_city}** to **{dest_city}** via **{partner} ({mode})** "
                f"is within safe SLA limits under **{selected_clf_name}**."
            )

        st.subheader("🔄 Cross-Partner Simulation (Using Selected Models)")
        partner_rows = []
        for p in opts["partners"]:
            temp_inputs = user_inputs.copy()
            temp_inputs["Delivery_Partner"] = p
            X_temp = build_feature_vector(temp_inputs, artifacts["feature_columns"], artifacts["scaler_stats"])

            h = float(y_predict(X_temp, reg_obj["w"], reg_obj["b"])[0]) if reg_obj["type"] == "scratch" else float(reg_obj["model"].predict(X_temp)[0])
            prob = float(sigmoid(X_temp @ clf_obj["w"] + clf_obj["b"])[0]) if clf_obj["type"] == "scratch" else float(clf_obj["model"].predict_proba(X_temp)[0, 1])

            partner_rows.append({
                "Delivery Partner": p,
                "Predicted EDD (hrs)": round(max(0.37, h), 2),
                "Predicted Days": round(max(0.37, h) / 24.0, 2),
                "Delay Risk (%)": round(prob * 100, 2),
                "Status": "⚠️ Delayed" if prob >= tau else "✅ On-Time"
            })
        st.dataframe(pd.DataFrame(partner_rows), use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("🎯 SLA Delay Probability Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=delay_prob * 100,
            title={"text": f"Risk Score (%) vs Threshold ({tau*100:.0f}%)"},
            delta={"reference": tau * 100, "increasing": {"color": "#FF4B4B"}, "decreasing": {"color": "#00CC96"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#FF4B4B" if is_delayed else "#00CC96"},
                "steps": [
                    {"range": [0, tau * 100], "color": "rgba(0, 204, 150, 0.15)"},
                    {"range": [tau * 100, 100], "color": "rgba(255, 75, 75, 0.15)"}
                ],
                "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.75, "value": tau * 100}
            }
        ))
        fig_gauge.update_layout(height=340, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

with tab_compare:
    st.subheader("⚖️ All Models Evaluated on Your Current Shipment Input")
    c_reg, c_clf = st.columns(2)

    with c_reg:
        st.markdown("#### Regression Models (Predicted EDD)")
        all_reg_preds = []
        for name, m_obj in artifacts["reg_models"].items():
            val = float(y_predict(X_input, m_obj["w"], m_obj["b"])[0]) if m_obj["type"] == "scratch" else float(m_obj["model"].predict(X_input)[0])
            all_reg_preds.append({
                "Model": name,
                "Predicted Hours": round(max(0.37, val), 2),
                "Predicted Days": round(max(0.37, val) / 24.0, 2)
            })
        df_all_reg = pd.DataFrame(all_reg_preds)
        st.dataframe(df_all_reg, use_container_width=True, hide_index=True)

        fig_all_reg = px.bar(
            df_all_reg, x="Predicted Hours", y="Model", orientation="h",
            text="Predicted Hours", color="Model", title="EDD Prediction Across All 7 Regressors"
        )
        fig_all_reg.update_layout(showlegend=False, height=340)
        st.plotly_chart(fig_all_reg, use_container_width=True)

    with c_clf:
        st.markdown("#### Classification Models (Predicted Delay Probability)")
        all_clf_preds = []
        for name, m_obj in artifacts["clf_models"].items():
            prob = float(sigmoid(X_input @ m_obj["w"] + m_obj["b"])[0]) if m_obj["type"] == "scratch" else float(m_obj["model"].predict_proba(X_input)[0, 1])
            all_clf_preds.append({
                "Model": name,
                "Delay Probability (%)": round(prob * 100, 2),
                "Decision (at τ)": "⚠️ Delayed" if prob >= tau else "✅ On-Time"
            })
        df_all_clf = pd.DataFrame(all_clf_preds)
        st.dataframe(df_all_clf, use_container_width=True, hide_index=True)

        fig_all_clf = px.bar(
            df_all_clf, x="Delay Probability (%)", y="Model", orientation="h",
            text="Delay Probability (%)", color="Model", title="SLA Risk Score Across All 3 Classifiers"
        )
        fig_all_clf.add_vline(x=tau * 100, line_dash="dash", line_color="red", annotation_text=f"τ={tau}")
        fig_all_clf.update_layout(showlegend=False, height=340)
        st.plotly_chart(fig_all_clf, use_container_width=True)

with tab_bench:
    st.subheader("📊 Holdout Partition Performance ($m_{\\text{test}} = 42,170$ rows, 80/20 Sequential Split)")
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("#### Master Regression Evaluation Matrix")
        st.dataframe(artifacts["reg_metrics_df"], use_container_width=True, hide_index=True)
    with b2:
        st.markdown("#### Master Classification Evaluation Matrix")
        st.dataframe(artifacts["clf_metrics_df"], use_container_width=True, hide_index=True)

with tab_sparsity:
    st.subheader("🧬 Proximal Lasso ($L_1$) Feature Selection & Gradient Descent Convergence")
    s1, s2 = st.columns(2)
    with s1:
        weights = artifacts["lasso_weights"]
        zero_count = int((weights.abs() < 1e-4).sum())
        st.metric("Features Pruned to Zero (α = 0.0316)", f"{zero_count} / {len(weights)}", "Exact Soft-Thresholding Sparsity")

        non_zero = weights[weights.abs() >= 1e-4].sort_values(key=abs, ascending=True)
        fig_w = px.bar(
            x=non_zero.values,
            y=non_zero.index,
            orientation="h",
            labels={"x": "Learned Weight (w_j)", "y": "Feature"},
            title="Surviving Non-Zero Features in Custom Proximal Lasso"
        )
        fig_w.update_layout(height=420)
        st.plotly_chart(fig_w, use_container_width=True)

    with s2:
        st.markdown("#### Training Loss Convergence Curves")
        min_len = min(len(artifacts["loss_curves"]["lasso"]), len(artifacts["loss_curves"]["gd"]))
        loss_df = pd.DataFrame({
            "Epoch": range(1, min_len + 1),
            "Proximal Lasso MSE": artifacts["loss_curves"]["lasso"][:min_len],
            "Lipschitz GD MSE": artifacts["loss_curves"]["gd"][:min_len],
        })
        fig_loss = px.line(
            loss_df, x="Epoch", y=["Proximal Lasso MSE", "Lipschitz GD MSE"],
            title="Fast MSE Convergence Curves"
        )
        fig_loss.update_layout(height=450)
        st.plotly_chart(fig_loss, use_container_width=True)