import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

# ==========================================
# 1. PAGE CONFIGURATION + THEME
# ==========================================

st.set_page_config(
    page_title="Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "features.pkl"

GREEN, RED, BLUE, AMBER, PURPLE = "#16C784", "#FF4B4B", "#00D4FF", "#FFB020", "#9B6BFF"
STATUS_COLORS = {"Benign": GREEN, "Malicious": RED}
SEVERITY_COLORS = {"Low": GREEN, "Medium": AMBER, "High": "#FF7A45", "Critical": RED}

st.markdown(
    """
    <style>
    .kpi {background: linear-gradient(145deg,#0f1b2d,#13243b);
          border: 1px solid #1f3554; border-radius: 14px;
          padding: 16px 18px; margin-bottom: 8px;}
    .kpi .label {color:#8aa4c8; font-size:.75rem; text-transform:uppercase;
                 letter-spacing:.08em;}
    .kpi .value {color:#ffffff; font-size:1.9rem; font-weight:700; line-height:1.3;}
    .kpi .sub {font-size:.8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi(col, label, value, sub="", color=GREEN):
    col.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="sub" style="color:{color}">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def style(fig, height=380):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
    return model, list(features)


def get_importance(model, features):
    """Feature importance (tree models) or |coefficients| (linear models)."""
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_)
    elif hasattr(est, "coef_"):
        imp = np.abs(np.asarray(est.coef_)).ravel()
    else:
        return None
    if len(imp) != len(features):
        return None
    return (
        pd.DataFrame({"Feature": features, "Importance": imp})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


st.title("🛡️ Network Intrusion Detection System")
st.caption("Machine Learning Based Cybersecurity Monitoring Dashboard")
st.divider()

try:
    model, features = load_model()
except Exception as e:
    st.error(f"Unable to load model files: {e}")
    st.stop()

importance_df = get_importance(model, features)


# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🛡️ IDS Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Attack Detection", "Model Performance", "About"],
)

threshold = 0.5
if page == "Attack Detection":
    st.sidebar.divider()
    threshold = st.sidebar.slider(
        "Detection threshold (attack probability)",
        0.05, 0.95, 0.50, 0.05,
        help="Records with attack probability at or above this value are "
             "flagged as Malicious. Lower = catch more attacks, more false alarms.",
    )


# ==========================================
# 4. HELPERS FOR THE ANALYSIS DASHBOARD
# ==========================================

def run_analysis(data):
    """Validate, predict, and store raw results in session_state."""
    missing = [c for c in features if c not in data.columns]
    if missing:
        st.error(f"Missing required features: {missing}")
        st.stop()

    X = data[features].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    if not hasattr(model, "predict_proba"):
        st.error("The saved model does not support probability prediction.")
        st.stop()

    try:
        proba = model.predict_proba(X)
    except ValueError:
        # Model has no imputer: fall back to filling gaps with 0
        X = X.fillna(0)
        proba = model.predict_proba(X)

    classes = list(model.classes_)
    if 1 not in classes:
        st.error("The trained model does not contain attack class 1.")
        st.stop()

    st.session_state["ids_data"] = data
    st.session_state["ids_X"] = X
    st.session_state["ids_prob"] = proba[:, classes.index(1)]


def build_results(data, prob, thr):
    res = data.copy()
    res["Prediction"] = (prob >= thr).astype(int)
    res["Status"] = np.where(res["Prediction"] == 1, "Malicious", "Benign")
    res["Attack Probability (%)"] = (prob * 100).round(2)
    res["Severity"] = pd.cut(
        prob * 100,
        bins=[-0.01, 50, 70, 90, 100.01],
        labels=["Low", "Medium", "High", "Critical"],
    ).astype(str)
    if "Label" in res.columns:
        res["Actual Label"] = res["Label"].astype(str).str.strip()
    return res


def build_timeline(res):
    """Benign vs malicious counts across 40 time (or record) windows."""
    tl = res[["Status"]].copy()
    xlabel = "Record window"
    if "Timestamp" in res.columns:
        ts = pd.to_datetime(res["Timestamp"], errors="coerce")
        if ts.notna().mean() > 0.5:
            tl["Time"] = ts
            tl = tl.dropna(subset=["Time"]).sort_values("Time")
            xlabel = "Time"
    n = len(tl)
    nb = max(1, min(40, n))
    tl["bin"] = np.arange(n) * nb // n
    counts = (
        tl.groupby(["bin", "Status"]).size().unstack(fill_value=0)
        .reindex(columns=["Benign", "Malicious"], fill_value=0)
    )
    if xlabel == "Time":
        counts.index = tl.groupby("bin")["Time"].min()
    else:
        counts.index = counts.index + 1
    return counts, xlabel


# ==========================================
# 5. DASHBOARD OVERVIEW
# ==========================================

if page == "Dashboard":

    st.header("Security Overview")
    st.success("Trained machine learning model is ready.")

    c1, c2, c3 = st.columns(3)
    kpi(c1, "Model Status", "Active", "Loaded from best_model.pkl")
    kpi(c2, "Input Features", len(features), "Used for every prediction", BLUE)
    kpi(c3, "Classification", "Binary", "Benign (0) / Attack (1)", AMBER)

    st.subheader("System Architecture")
    st.markdown(
        """
        **Network Traffic CSV** ➜ **Data Preprocessing** ➜
        **Trained ML Model** ➜ **Benign / Malicious Classification** ➜
        **Dashboard & Reports**
        """
    )
    st.info("Go to Attack Detection to upload a network traffic CSV.")


# ==========================================
# 6. ATTACK DETECTION (MAIN ANALYTICS DASHBOARD)
# ==========================================

elif page == "Attack Detection":

    st.header("Network Traffic Analysis")

    uploaded_file = st.file_uploader("Upload CIC-IDS2017 CSV File", type=["csv"])

    if uploaded_file is not None:

        # reset stored results when a different file is uploaded
        if st.session_state.get("ids_file") != uploaded_file.name:
            for k in ("ids_data", "ids_X", "ids_prob"):
                st.session_state.pop(k, None)
            st.session_state["ids_file"] = uploaded_file.name

        try:
            data = pd.read_csv(uploaded_file)
            data.columns = data.columns.str.strip()

            if data.empty:
                st.error("The uploaded CSV is empty.")
                st.stop()

            p1, p2 = st.columns(2)
            kpi(p1, "Total Records", f"{len(data):,}", "", BLUE)
            kpi(p2, "Total Columns", len(data.columns), "", BLUE)
            with st.expander("Dataset preview"):
                st.dataframe(data.head(10), use_container_width=True)

            if st.button("Analyse Network Traffic", type="primary"):
                with st.spinner("Running model..."):
                    run_analysis(data)

        except Exception as e:
            st.error(f"Analysis error: {e}")
            st.stop()

    # ---------- render dashboard from stored results ----------
    if "ids_prob" in st.session_state:

        try:
            data = st.session_state["ids_data"]
            X = st.session_state["ids_X"]
            prob = st.session_state["ids_prob"]

            res = build_results(data, prob, threshold)
            preds = res["Prediction"].to_numpy()
            has_label = "Label" in res.columns

            total = len(res)
            malicious = int(preds.sum())
            benign = total - malicious
            mal_pct = malicious / total * 100

            y_true = None
            if has_label:
                y_true = (res["Actual Label"].str.upper() != "BENIGN").astype(int).to_numpy()
                if len(np.unique(y_true)) < 2:
                    y_true = None

            st.divider()

            tab_over, tab_traffic, tab_attack, tab_eval, tab_report = st.tabs(
                ["📊 Overview", "📈 Traffic Analysis", "🎯 Attack Analysis",
                 "🧪 Model Evaluation", "📥 Report"]
            )

            # ======================================================
            # TAB 1: OVERVIEW
            # ======================================================
            with tab_over:

                k1, k2, k3, k4, k5 = st.columns(5)
                kpi(k1, "Total Traffic", f"{total:,}", "records analysed", BLUE)
                kpi(k2, "Benign", f"{benign:,}", f"{100 - mal_pct:.1f}% of traffic", GREEN)
                kpi(k3, "Malicious", f"{malicious:,}", f"{mal_pct:.1f}% of traffic", RED)
                kpi(k4, "Avg Attack Probability", f"{prob.mean() * 100:.1f}%",
                    "across all records", AMBER)
                critical = int((res["Severity"] == "Critical").sum())
                kpi(k5, "Critical Alerts", f"{critical:,}", "probability ≥ 90%", RED)

                # --- timeline + donut ---
                left, right = st.columns([2, 1])

                counts, xlabel = build_timeline(res)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=counts.index, y=counts["Benign"], name="Benign",
                    mode="lines+markers", line=dict(color=GREEN, width=2.5),
                    fill="tozeroy", fillcolor="rgba(22,199,132,0.10)"))
                fig.add_trace(go.Scatter(
                    x=counts.index, y=counts["Malicious"], name="Malicious",
                    mode="lines+markers", line=dict(color=RED, width=2.5),
                    fill="tozeroy", fillcolor="rgba(255,75,75,0.12)"))
                fig.update_layout(title="Traffic Overview (benign vs malicious over time)",
                                  xaxis_title=xlabel, yaxis_title="Flows per window")
                with left:
                    show(style(fig, 400))

                fig = px.pie(
                    pd.DataFrame({"Status": ["Benign", "Malicious"],
                                  "Count": [benign, malicious]}),
                    names="Status", values="Count", hole=0.6,
                    color="Status", color_discrete_map=STATUS_COLORS,
                    title="Traffic Distribution")
                fig.update_traces(textinfo="label+percent")
                fig.add_annotation(text=f"<b>{total:,}</b><br>flows", showarrow=False,
                                   font=dict(size=16))
                with right:
                    show(style(fig, 400))

                # --- gauge + severity + attack types ---
                g1, g2, g3 = st.columns(3)

                gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=mal_pct,
                    number={"suffix": "%"},
                    title={"text": "Threat Level (malicious share)"},
                    gauge={"axis": {"range": [0, 100]},
                           "bar": {"color": RED if mal_pct > 50 else AMBER if mal_pct > 20 else GREEN},
                           "steps": [{"range": [0, 20], "color": "rgba(22,199,132,0.25)"},
                                     {"range": [20, 50], "color": "rgba(255,176,32,0.25)"},
                                     {"range": [50, 100], "color": "rgba(255,75,75,0.25)"}]}))
                with g1:
                    show(style(gauge, 330))

                sev = (res.loc[res["Prediction"] == 1, "Severity"]
                       .value_counts()
                       .reindex(["Low", "Medium", "High", "Critical"], fill_value=0)
                       .reset_index())
                sev.columns = ["Severity", "Count"]
                fig = px.bar(sev, x="Severity", y="Count", color="Severity",
                             color_discrete_map=SEVERITY_COLORS,
                             title="Alert Severity (malicious flows)", text="Count")
                fig.update_layout(showlegend=False)
                with g2:
                    show(style(fig, 330))

                with g3:
                    if has_label:
                        lc = res["Actual Label"].value_counts().reset_index()
                        lc.columns = ["Attack Type", "Count"]
                        fig = px.pie(lc, names="Attack Type", values="Count", hole=0.55,
                                     title="Attack Types (actual labels)",
                                     color_discrete_sequence=px.colors.qualitative.Bold)
                        fig.update_traces(textinfo="percent")
                        show(style(fig, 330))
                    else:
                        st.info("Upload a file with a Label column to see attack types.")

                # --- recent alerts ---
                st.subheader("Recent Alerts (highest attack probability)")
                alert_cols = [c for c in ["Timestamp", "Source IP", "Destination IP",
                                          "Destination Port", "Protocol"] if c in res.columns]
                alert_cols += ["Status", "Severity", "Attack Probability (%)"]
                if has_label:
                    alert_cols.append("Actual Label")
                alerts = (res[res["Prediction"] == 1]
                          .sort_values("Attack Probability (%)", ascending=False)
                          .head(20)[alert_cols])
                if alerts.empty:
                    st.success("No malicious traffic detected at this threshold.")
                else:
                    st.dataframe(
                        alerts, use_container_width=True,
                        column_config={"Attack Probability (%)": st.column_config.ProgressColumn(
                            "Attack Probability (%)", min_value=0, max_value=100, format="%.1f%%")})

            # ======================================================
            # TAB 2: TRAFFIC ANALYSIS
            # ======================================================
            with tab_traffic:

                t1, t2 = st.columns(2)

                fig = px.histogram(
                    res, x="Attack Probability (%)", color="Status", nbins=40,
                    barmode="overlay", opacity=0.75, color_discrete_map=STATUS_COLORS,
                    title="Attack Probability Distribution")
                fig.update_layout(yaxis_title="Number of records", yaxis_type="log")
                with t1:
                    show(style(fig, 400))
                    st.caption("Log scale on Y. A good model shows two clear clusters near 0% and 100%.")

                with t2:
                    if "Destination Port" in res.columns:
                        pt = (res[res["Prediction"] == 1]["Destination Port"]
                              .value_counts().head(10).reset_index())
                        pt.columns = ["Destination Port", "Malicious Flows"]
                        pt["Destination Port"] = pt["Destination Port"].astype(str)
                        fig = px.bar(pt, x="Malicious Flows", y="Destination Port",
                                     orientation="h", title="Top 10 Targeted Ports",
                                     color="Malicious Flows", color_continuous_scale="Reds")
                        fig.update_layout(yaxis=dict(autorange="reversed"),
                                          coloraxis_showscale=False)
                        show(style(fig, 400))
                    else:
                        st.info("No Destination Port column found.")

                st.subheader("Feature Explorer")
                default_feats = (importance_df["Feature"].tolist() if importance_df is not None
                                 else features)
                f1, f2, f3 = st.columns([2, 2, 1])
                feat_a = f1.selectbox("Feature", default_feats, index=0)
                chart_type = f2.radio("Chart", ["Box", "Violin", "Histogram"], horizontal=True)
                use_log = f3.checkbox("Log scale", value=True)

                sample_idx = (X.sample(min(20000, len(X)), random_state=42).index
                              if len(X) > 20000 else X.index)
                plot_df = pd.DataFrame({"value": X.loc[sample_idx, feat_a],
                                        "Status": res.loc[sample_idx, "Status"]}).dropna()
                if use_log:
                    plot_df["value"] = np.log10(plot_df["value"].clip(lower=0) + 1)
                ylab = f"log10({feat_a} + 1)" if use_log else feat_a

                if chart_type == "Box":
                    fig = px.box(plot_df, x="Status", y="value", color="Status",
                                 color_discrete_map=STATUS_COLORS)
                elif chart_type == "Violin":
                    fig = px.violin(plot_df, x="Status", y="value", color="Status",
                                    box=True, color_discrete_map=STATUS_COLORS)
                else:
                    fig = px.histogram(plot_df, x="value", color="Status", nbins=50,
                                       barmode="overlay", opacity=0.7,
                                       color_discrete_map=STATUS_COLORS)
                fig.update_layout(title=f"{feat_a}: Benign vs Malicious",
                                  xaxis_title="Status" if chart_type != "Histogram" else ylab,
                                  yaxis_title=ylab if chart_type != "Histogram" else "Count")
                show(style(fig, 420))

                s1, s2 = st.columns(2)

                with s1:
                    st.markdown("**Feature scatter plot**")
                    sa, sb = st.columns(2)
                    fx = sa.selectbox("X axis", default_feats, index=0, key="sx")
                    fy = sb.selectbox("Y axis", default_feats,
                                      index=min(1, len(default_feats) - 1), key="sy")
                    n_s = min(5000, len(X))
                    sidx = X.sample(n_s, random_state=1).index
                    sc = pd.DataFrame({fx: X.loc[sidx, fx], fy: X.loc[sidx, fy],
                                       "Status": res.loc[sidx, "Status"]})
                    sc[fx] = np.log10(sc[fx].clip(lower=0) + 1)
                    sc[fy] = np.log10(sc[fy].clip(lower=0) + 1)
                    fig = px.scatter(sc, x=fx, y=fy, color="Status", opacity=0.6,
                                     color_discrete_map=STATUS_COLORS,
                                     title=f"{fx} vs {fy} (log10, {n_s:,} sampled flows)")
                    show(style(fig, 420))

                with s2:
                    top_feats = default_feats[:12]
                    csample = X[top_feats].sample(min(20000, len(X)), random_state=7)
                    corr = csample.corr().round(2)
                    fig = px.imshow(corr, text_auto=True, aspect="auto",
                                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                    title="Correlation Heatmap (top 12 features)")
                    fig.update_traces(textfont_size=9)
                    st.markdown("&nbsp;")
                    show(style(fig, 460))

            # ======================================================
            # TAB 3: ATTACK ANALYSIS
            # ======================================================
            with tab_attack:

                if has_label:
                    a1, a2 = st.columns(2)

                    lc = res["Actual Label"].value_counts().reset_index()
                    lc.columns = ["Attack Type", "Count"]
                    log_y = a1.checkbox("Log scale (helps with rare attacks)", value=True)
                    fig = px.bar(lc, x="Attack Type", y="Count", color="Attack Type",
                                 title="Attack Type Distribution (actual labels)",
                                 color_discrete_sequence=px.colors.qualitative.Bold)
                    fig.update_layout(xaxis_tickangle=-35, showlegend=False,
                                      yaxis_type="log" if log_y else "linear")
                    with a1:
                        show(style(fig, 450))

                    ct = (res.groupby(["Actual Label", "Status"]).size()
                          .reset_index(name="Count"))
                    fig = px.sunburst(ct, path=["Actual Label", "Status"], values="Count",
                                      color="Status", color_discrete_map=STATUS_COLORS,
                                      title="Actual Attack Type → Model Verdict")
                    with a2:
                        st.markdown("&nbsp;")
                        show(style(fig, 450))

                    # per-type detection rate
                    rate = res.assign(
                        Correct=np.where(res["Actual Label"].str.upper() == "BENIGN",
                                         res["Prediction"] == 0, res["Prediction"] == 1)
                    ).groupby("Actual Label")["Correct"].agg(["mean", "size"]).reset_index()
                    rate.columns = ["Traffic Type", "Correct %", "Records"]
                    rate["Correct %"] = (rate["Correct %"] * 100).round(2)
                    rate = rate.sort_values("Correct %")
                    fig = px.bar(rate, x="Correct %", y="Traffic Type", orientation="h",
                                 color="Correct %", color_continuous_scale="RdYlGn",
                                 range_color=[0, 100], text="Correct %", hover_data=["Records"],
                                 title="Detection Rate per Traffic Type "
                                       "(attack types: recall, benign: specificity)")
                    fig.update_layout(coloraxis_showscale=False)
                    show(style(fig, 450))

                    stack = res.groupby(["Actual Label", "Status"]).size().reset_index(name="Count")
                    fig = px.bar(stack, x="Actual Label", y="Count", color="Status",
                                 color_discrete_map=STATUS_COLORS, barmode="stack",
                                 title="Predicted Status for Each Actual Traffic Type")
                    fig.update_layout(xaxis_tickangle=-35, yaxis_type="log")
                    show(style(fig, 420))
                    st.caption("The model is binary, so attack types come from the uploaded "
                               "Label column, not from the model's prediction.")
                else:
                    st.info("Upload a labelled dataset (with a Label column) to see "
                            "attack-type analysis.")

                if importance_df is not None:
                    top = importance_df.head(15)
                    fig = px.bar(top, x="Importance", y="Feature", orientation="h",
                                 color="Importance", color_continuous_scale="Tealgrn",
                                 title="Top 15 Features Driving the Model")
                    fig.update_layout(yaxis=dict(autorange="reversed"),
                                      coloraxis_showscale=False)
                    show(style(fig, 480))

            # ======================================================
            # TAB 4: MODEL EVALUATION
            # ======================================================
            with tab_eval:

                if y_true is None:
                    st.info("Upload a labelled dataset containing both BENIGN and attack "
                            "records to see accuracy, confusion matrix, ROC and PR curves.")
                else:
                    acc = accuracy_score(y_true, preds)
                    prec = precision_score(y_true, preds, zero_division=0)
                    rec = recall_score(y_true, preds, zero_division=0)
                    f1 = f1_score(y_true, preds, zero_division=0)
                    auc = roc_auc_score(y_true, prob)
                    ap = average_precision_score(y_true, prob)

                    m = st.columns(6)
                    kpi(m[0], "Accuracy", f"{acc:.4f}", "", BLUE)
                    kpi(m[1], "Precision", f"{prec:.4f}", "", GREEN)
                    kpi(m[2], "Recall", f"{rec:.4f}", "", AMBER)
                    kpi(m[3], "F1 Score", f"{f1:.4f}", "", PURPLE)
                    kpi(m[4], "ROC-AUC", f"{auc:.4f}", "", BLUE)
                    kpi(m[5], "Avg Precision", f"{ap:.4f}", "", GREEN)

                    cm = confusion_matrix(y_true, preds, labels=[0, 1])
                    tn, fp, fn, tp = cm.ravel()
                    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

                    e1, e2 = st.columns(2)
                    fig = px.imshow(cm, text_auto=True, x=["Benign", "Attack"],
                                    y=["Benign", "Attack"], color_continuous_scale="Blues",
                                    title="Confusion Matrix (counts)")
                    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
                    with e1:
                        show(style(fig, 400))

                    fig = px.imshow(cm_norm, text_auto=".2%", x=["Benign", "Attack"],
                                    y=["Benign", "Attack"], color_continuous_scale="Greens",
                                    zmin=0, zmax=1, title="Confusion Matrix (row-normalised)")
                    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
                    with e2:
                        show(style(fig, 400))

                    q1, q2, q3, q4 = st.columns(4)
                    kpi(q1, "True Negatives", f"{int(tn):,}", "benign → benign", GREEN)
                    kpi(q2, "False Positives", f"{int(fp):,}", "false alarms", AMBER)
                    kpi(q3, "False Negatives", f"{int(fn):,}", "missed attacks", RED)
                    kpi(q4, "True Positives", f"{int(tp):,}", "attacks caught", GREEN)
                    if fn > 0:
                        st.warning(f"{int(fn):,} malicious records were missed. "
                                   "Try lowering the threshold in the sidebar.")

                    r1, r2 = st.columns(2)
                    fpr, tpr, _ = roc_curve(y_true, prob)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                             name=f"Model (AUC = {auc:.4f})",
                                             line=dict(color=BLUE, width=3),
                                             fill="tozeroy", fillcolor="rgba(0,212,255,0.10)"))
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                             name="Random classifier",
                                             line=dict(color=RED, dash="dash")))
                    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                                      yaxis_title="True Positive Rate")
                    with r1:
                        show(style(fig, 420))

                    pr_p, pr_r, _ = precision_recall_curve(y_true, prob)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=pr_r, y=pr_p, mode="lines",
                                             name=f"Model (AP = {ap:.4f})",
                                             line=dict(color=GREEN, width=3),
                                             fill="tozeroy", fillcolor="rgba(22,199,132,0.10)"))
                    fig.update_layout(title="Precision-Recall Curve",
                                      xaxis_title="Recall", yaxis_title="Precision")
                    with r2:
                        show(style(fig, 420))

                    # threshold sweep
                    sweep = []
                    for t in np.arange(0.05, 1.0, 0.05):
                        p_t = (prob >= t).astype(int)
                        sweep.append({
                            "Threshold": round(t, 2),
                            "Precision": precision_score(y_true, p_t, zero_division=0),
                            "Recall": recall_score(y_true, p_t, zero_division=0),
                            "F1": f1_score(y_true, p_t, zero_division=0)})
                    sweep = pd.DataFrame(sweep).melt("Threshold", var_name="Metric",
                                                     value_name="Score")
                    fig = px.line(sweep, x="Threshold", y="Score", color="Metric",
                                  markers=True, title="Threshold Analysis",
                                  color_discrete_map={"Precision": GREEN, "Recall": AMBER,
                                                      "F1": PURPLE})
                    fig.add_vline(x=threshold, line_dash="dot", line_color="white",
                                  annotation_text="current")
                    show(style(fig, 400))

                    bars = pd.DataFrame({"Metric": ["Accuracy", "Precision", "Recall",
                                                    "F1", "ROC-AUC"],
                                         "Score": [acc, prec, rec, f1, auc]})
                    fig = px.bar(bars, x="Metric", y="Score", color="Metric", text_auto=".4f",
                                 range_y=[0, 1.05], title="Metric Summary",
                                 color_discrete_sequence=[BLUE, GREEN, AMBER, PURPLE, "#00B8A9"])
                    fig.update_layout(showlegend=False)
                    show(style(fig, 360))

            # ======================================================
            # TAB 5: REPORT
            # ======================================================
            with tab_report:
                st.subheader("Download Detection Report")
                st.write(f"**{total:,}** records analysed · **{malicious:,}** flagged malicious "
                         f"({mal_pct:.2f}%) · threshold **{threshold:.2f}**")

                d1, d2 = st.columns(2)
                d1.download_button(
                    "📥 Full detection report (CSV)",
                    res.to_csv(index=False).encode("utf-8"),
                    file_name="network_detection_report.csv", mime="text/csv")
                d2.download_button(
                    "🚨 Malicious flows only (CSV)",
                    res[res["Prediction"] == 1].to_csv(index=False).encode("utf-8"),
                    file_name="malicious_traffic.csv", mime="text/csv")

                st.dataframe(
                    res[["Status", "Severity", "Attack Probability (%)"]].head(100),
                    use_container_width=True)

        except Exception as e:
            st.error(f"Dashboard error: {e}")


# ==========================================
# 7. MODEL PERFORMANCE
# ==========================================

elif page == "Model Performance":

    st.header("Model Information")

    c1, c2 = st.columns(2)
    kpi(c1, "Input Features", len(features), "", BLUE)
    kpi(c2, "Model Type", type(model).__name__, "", PURPLE)

    if importance_df is not None:
        fig = px.bar(importance_df.head(20), x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Tealgrn",
                     title="Top 20 Feature Importances")
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        show(style(fig, 560))

    st.subheader("Feature Names")
    st.dataframe(pd.DataFrame({"Feature": features}), use_container_width=True)

    st.info("For accuracy, ROC and confusion matrix, upload a labelled dataset "
            "in the Attack Detection section.")


# ==========================================
# 8. ABOUT
# ==========================================

elif page == "About":

    st.header("About This Project")

    st.markdown(
        """
        ### Network Intrusion Detection System

        This project uses machine learning to identify suspicious network traffic.

        **Dataset:** CIC-IDS2017

        **Algorithms evaluated:** Logistic Regression, Decision Tree, Random Forest

        **Selected model:** Loaded from best_model.pkl

        **Technologies:** Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit

        **Classification:** BENIGN (0) · ATTACK (1)
        """
    )

    st.warning("This is a dataset-based detection prototype, "
               "not a live network monitoring or prevention system.")