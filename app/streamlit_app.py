import streamlit as st
import requests
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Churn Risk Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Section headers */
    .section-header {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 12px;
        margin-top: 24px;
    }

    /* Result cards */
    .result-card {
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }
    .card-high {
        background: #1f0a0a;
        border: 1px solid #ef4444;
    }
    .card-medium {
        background: #1a1400;
        border: 1px solid #f59e0b;
    }
    .card-low {
        background: #041a0a;
        border: 1px solid #22c55e;
    }
    .risk-label-high   { color: #ef4444; font-size: 1.5rem; font-weight: 800; }
    .risk-label-medium { color: #f59e0b; font-size: 1.5rem; font-weight: 800; }
    .risk-label-low    { color: #22c55e; font-size: 1.5rem; font-weight: 800; }
    .prob-text { color: #e5e7eb; font-size: 1rem; margin-top: 4px; }
    .pred-text { color: #9ca3af; font-size: 0.9rem; margin-top: 4px; }

    /* Factor tags */
    .factor-tag {
        display: inline-block;
        background: #1e293b;
        border: 1px solid #334155;
        color: #94a3b8;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        margin: 3px;
    }

    /* Profile summary table */
    .profile-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #1e293b;
        font-size: 0.9rem;
    }
    .profile-key   { color: #6b7280; }
    .profile-value { color: #e5e7eb; font-weight: 500; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0d1117; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label { color: #9ca3af !important; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Customer Profile")
    st.markdown("---")

    st.markdown('<div class="section-header">Demographics</div>', unsafe_allow_html=True)
    gender     = st.selectbox("Gender", ["Male", "Female"])
    senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner    = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])

    st.markdown('<div class="section-header">Account</div>', unsafe_allow_html=True)
    tenure   = st.slider("Tenure (months)", 0, 72, 12)
    monthly  = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total    = st.number_input("Total Charges ($)", min_value=0.0,
                               value=float(round(monthly * tenure, 2)), step=10.0)
    contract = st.selectbox("Contract Type",
                            ["Month-to-month", "One year", "Two year"])
    payment  = st.selectbox("Payment Method",
                            ["Electronic check", "Mailed check",
                             "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    st.markdown('<div class="section-header">Services</div>', unsafe_allow_html=True)
    phone         = st.selectbox("Phone Service", ["Yes", "No"])
    multi_lines   = st.selectbox("Multiple Lines", ["No", "Yes"])
    internet      = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    online_sec    = st.selectbox("Online Security", ["No", "Yes"])
    online_bkp    = st.selectbox("Online Backup", ["No", "Yes"])
    device_prot   = st.selectbox("Device Protection", ["No", "Yes"])
    tech_sup      = st.selectbox("Tech Support", ["No", "Yes"])
    streaming_tv  = st.selectbox("Streaming TV", ["No", "Yes"])
    streaming_mov = st.selectbox("Streaming Movies", ["No", "Yes"])


# ── Build payload ─────────────────────────────────────────────────────────────
def yn(v): return 1 if v == "Yes" else 0

payload = {
    "gender": 1 if gender == "Male" else 0,
    "SeniorCitizen": yn(senior),
    "Partner": yn(partner),
    "Dependents": yn(dependents),
    "tenure": tenure,
    "PhoneService": yn(phone),
    "MultipleLines": yn(multi_lines),
    "OnlineSecurity": yn(online_sec),
    "OnlineBackup": yn(online_bkp),
    "DeviceProtection": yn(device_prot),
    "TechSupport": yn(tech_sup),
    "StreamingTV": yn(streaming_tv),
    "StreamingMovies": yn(streaming_mov),
    "PaperlessBilling": yn(paperless),
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "InternetService_Fiber_optic": 1 if internet == "Fiber optic" else 0,
    "InternetService_No": 1 if internet == "No" else 0,
    "Contract_One_year": 1 if contract == "One year" else 0,
    "Contract_Two_year": 1 if contract == "Two year" else 0,
    "PaymentMethod_Credit_card__automatic_": 1 if payment == "Credit card (automatic)" else 0,
    "PaymentMethod_Electronic_check": 1 if payment == "Electronic check" else 0,
    "PaymentMethod_Mailed_check": 1 if payment == "Mailed check" else 0,
}


# ── Gauge chart ───────────────────────────────────────────────────────────────
def gauge_chart(prob):
    color = "#ef4444" if prob >= 0.70 else "#f59e0b" if prob >= 0.40 else "#22c55e"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": "#e5e7eb"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4b5563",
                     "tickfont": {"color": "#6b7280"}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#1e293b",
            "bordercolor": "#1e293b",
            "steps": [
                {"range": [0,  40], "color": "#0a1a0f"},
                {"range": [40, 70], "color": "#1a1400"},
                {"range": [70, 100], "color": "#1f0a0a"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": prob * 100
            }
        }
    ))
    fig.update_layout(
        height=220,
        margin=dict(t=20, b=0, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5e7eb"}
    )
    return fig


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("## 📊 Churn Risk Analyzer")
st.markdown("Adjust the customer profile in the sidebar, then click **Analyze**.")
st.markdown("---")

left, right = st.columns([1, 1], gap="large")

# Left — Profile summary
with left:
    st.markdown("### 🧾 Customer Profile")

    profile_items = [
        ("Gender",          gender),
        ("Senior Citizen",  senior),
        ("Partner",         partner),
        ("Dependents",      dependents),
        ("Tenure",          f"{tenure} months"),
        ("Monthly Charges", f"${monthly:.2f}"),
        ("Total Charges",   f"${total:.2f}"),
        ("Contract",        contract),
        ("Payment Method",  payment),
        ("Internet",        internet),
        ("Phone Service",   phone),
        ("Online Security", online_sec),
        ("Tech Support",    tech_sup),
        ("Streaming TV",    streaming_tv),
        ("Streaming Movies",streaming_mov),
    ]

    rows_html = ""
    for k, v in profile_items:
        rows_html += f"""
        <div class="profile-row">
            <span class="profile-key">{k}</span>
            <span class="profile-value">{v}</span>
        </div>"""
    st.markdown(rows_html, unsafe_allow_html=True)

# Right — Prediction
with right:
    st.markdown("### 🔮 Risk Assessment")
    analyze = st.button("▶ Analyze Churn Risk", type="primary", use_container_width=True)

    if analyze:
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if resp.status_code == 200:
                r    = resp.json()
                prob = r["churn_probability"]
                risk = r["risk_level"]
                pred = r["churn_prediction"]
                expl = r["explanation"]

                # Gauge
                st.plotly_chart(gauge_chart(prob), use_container_width=True)

                # Risk card
                card_cls  = f"card-{risk.lower()}"
                label_cls = f"risk-label-{risk.lower()}"
                emoji     = "🔴" if risk=="HIGH" else "🟡" if risk=="MEDIUM" else "🟢"
                verdict   = "Will Churn" if pred == 1 else "Will Stay"

                st.markdown(f"""
                <div class="result-card {card_cls}">
                    <div class="{label_cls}">{emoji} {risk} RISK</div>
                    <div class="prob-text">Churn Probability: <strong>{prob*100:.1f}%</strong></div>
                    <div class="pred-text">Verdict: <strong>{verdict}</strong> &nbsp;|&nbsp; Model: {r['model_used']}</div>
                </div>
                """, unsafe_allow_html=True)

                # Risk factors
                if expl and "No major" not in expl:
                    factors = expl.replace("Risk factors: ", "").replace(".", "").split(", ")
                    tags = "".join([f'<span class="factor-tag">⚠ {f}</span>' for f in factors])
                    st.markdown(f"""
                    <div style="margin-top:12px">
                        <div class="section-header">Risk Factors Detected</div>
                        {tags}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color:#22c55e;margin-top:12px">✅ No major risk factors detected.</div>',
                                unsafe_allow_html=True)

            else:
                st.error(f"API error {resp.status_code}: {resp.text}")

        except requests.exceptions.ConnectionError:
            st.error("❌ API not running. Start it with:\n`venv/bin/python3 -m uvicorn api.main:app --reload`")
        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.markdown("""
        <div style="color:#4b5563; text-align:center; padding:60px 20px; border:1px dashed #1e293b; border-radius:10px; margin-top:16px">
            <div style="font-size:2.5rem">📈</div>
            <div style="margin-top:8px; font-size:0.95rem">Configure the customer profile<br>and click <strong style="color:#6b7280">Analyze</strong> to see results</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="color:#374151;font-size:0.8rem;text-align:center">'
    'Built with Scikit-learn · FastAPI · Streamlit &nbsp;|&nbsp; Telco Customer Churn Dataset'
    '</div>',
    unsafe_allow_html=True
)
