
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "GLP-1 Sales Intelligence | California 2023",
    page_icon  = "💊",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #F8F9FA; }
    [data-testid="stSidebar"] { background-color: #0A2342; padding-top: 1rem; }
    [data-testid="stSidebar"] * { color: #E8EDF2 !important; }
    [data-testid="stSidebar"] hr { border-color: #1E3A5F; }
    .header-bar {
        background: linear-gradient(135deg, #0A2342 0%, #1B4F8A 100%);
        padding: 20px 32px; border-radius: 8px; margin-bottom: 24px;
        display: flex; justify-content: space-between; align-items: center;
    }
    .header-title { color: white; font-size: 22px; font-weight: 700;
                    margin: 0; letter-spacing: -0.3px; }
    .header-sub   { color: #8FB3D9; font-size: 13px;
                    margin-top: 4px; font-weight: 400; }
    .header-badge {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2);
        color: white; padding: 6px 14px; border-radius: 20px;
        font-size: 12px; font-weight: 500;
    }
    .metric-row { display: flex; gap: 16px; margin-bottom: 24px; }
    .metric-card {
        background: white; border: 1px solid #E2E8F0;
        border-radius: 8px; padding: 16px 20px; flex: 1;
        border-left: 4px solid #1B4F8A;
    }
    .metric-card.green { border-left-color: #1A7F5A; }
    .metric-card.amber { border-left-color: #D97706; }
    .metric-label {
        font-size: 11px; font-weight: 600; color: #64748B;
        text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px;
    }
    .metric-value { font-size: 28px; font-weight: 700;
                    color: #0F172A; line-height: 1; margin-bottom: 4px; }
    .metric-sub   { font-size: 12px; color: #94A3B8; }
    .section-header {
        font-size: 15px; font-weight: 600; color: #0A2342;
        margin: 0 0 12px 0; padding-bottom: 8px;
        border-bottom: 2px solid #E2E8F0;
    }
    .physician-card {
        background: white; border: 1px solid #E2E8F0;
        border-radius: 8px; padding: 20px 24px; margin-bottom: 16px;
    }
    .physician-name  { font-size: 20px; font-weight: 700;
                       color: #0A2342; margin-bottom: 4px; }
    .physician-meta  { font-size: 13px; color: #64748B; margin-bottom: 16px; }
    .score-bar-bg    { background: #E2E8F0; border-radius: 4px;
                       height: 8px; width: 100%; margin-top: 6px; }
    .score-bar-fill  { border-radius: 4px; height: 8px; }
    .insight-box {
        background: #EFF6FF; border: 1px solid #BFDBFE;
        border-radius: 6px; padding: 12px 16px;
        margin: 8px 0; font-size: 13px; color: #1E3A5F;
    }
    .insight-box.green { background: #F0FDF4;
                         border-color: #BBF7D0; color: #14532D; }
    .ab-card { background: white; border: 1px solid #E2E8F0;
               border-radius: 8px; padding: 24px; text-align: center; }
    .ab-number { font-size: 48px; font-weight: 700;
                 color: #1B4F8A; line-height: 1; }
    .ab-label  { font-size: 13px; color: #64748B; margin-top: 6px; }
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
    header    {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — works on both Streamlit Cloud and Colab
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    # Try repo-relative path first (Streamlit Cloud)
    # Fall back to Drive path (Colab)
    candidates = [
        Path("data/processed"),
        Path("/content/drive/MyDrive/pharma_rec/data/processed"),
    ]
    PROC = next((p for p in candidates if p.exists()), None)

    if PROC is None:
        st.error("Data folder not found. Check that data/processed/ exists.")
        st.stop()

    targets      = pd.read_parquet(PROC / "sales_targets_final.parquet")
    physician_df = pd.read_parquet(PROC / "physician_features_glp1_ca_2023.parquet")
    interactions = pd.read_parquet(PROC / "interactions_glp1_ca_2023.parquet")
    ab_results   = pd.read_parquet(PROC / "ab_test_results.parquet")
    return targets, physician_df, interactions, ab_results

targets, physician_df, interactions, ab_results = load_data()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
    <div style="background:#0A2342; padding:10px 24px; border-radius:8px;
                margin-bottom:16px; display:flex; align-items:center;
                justify-content:space-between;">
        <div style="color:white; font-size:15px; font-weight:600;">
            💊 GLP-1 Sales Intelligence Platform
        </div>
        <div style="font-size:12px; color:#8FB3D9;">
            ← Use sidebar to navigate · click arrow to reopen
        </div>
    </div>
""", unsafe_allow_html=True)
with st.sidebar:
    st.markdown("""
        <div style="padding: 8px 0 24px 0;">
            <div style="font-size:11px; font-weight:600; color:#4A7FA5;
                        letter-spacing:1.5px; text-transform:uppercase;">
                Professional Style
            </div>
            <div style="font-size:20px; font-weight:700; color:white;
                        margin-top:4px; line-height:1.2;">
                Sales Intelligence<br>Platform
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Target List", "Physician Profile", "Model Validation"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
        <div style="font-size:11px; color:#4A7FA5; font-weight:600;
                    text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;">
            Campaign Details
        </div>
        <div style="font-size:13px; color:#B8C4CE; line-height:2;">
            <b style="color:white;">Drug</b><br>Tirzepatide (Mounjaro)<br><br>
            <b style="color:white;">Market</b><br>California, 2023<br><br>
            <b style="color:white;">Data Source</b><br>CMS Medicare Part D<br><br>
            <b style="color:white;">Model</b><br>Volume-weighted<br>cosine similarity
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
        <div style="font-size:11px; color:#4A7FA5; text-align:center;">
            Built with real CMS data<br>15,289 California physicians
        </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TARGET LIST
# ══════════════════════════════════════════════════════════════════════════════
if page == "Target List":

    st.markdown("""
        <div class="header-bar">
            <div>
                <div class="header-title">
                    Physician Target List — Tirzepatide
                </div>
                <div class="header-sub">
                    Ranked by affinity score · California Medicare · 2023
                </div>
            </div>
            <div class="header-badge">🟢 Model Active</div>
        </div>
    """, unsafe_allow_html=True)

    # Filters
    f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
    with f1:
        cities = ["All Cities"] + sorted(targets["city"].unique().tolist())
        city_filter = st.selectbox("City", cities)
    with f2:
        specs = ["All Specialties"] + sorted(targets["specialty"].unique().tolist())
        spec_filter = st.selectbox("Specialty", specs)
    with f3:
        tier_filter = st.multiselect(
            "Priority Tier", ["A","B","C"], default=["A","B"]
        )
    with f4:
        top_n = st.selectbox("Show top", [50, 100, 250, 500], index=0)

    # Apply filters
    filtered = targets.copy()
    if city_filter != "All Cities":
        filtered = filtered[filtered["city"] == city_filter]
    if spec_filter != "All Specialties":
        filtered = filtered[filtered["specialty"] == spec_filter]
    if tier_filter:
        filtered = filtered[filtered["priority_tier"].isin(tier_filter)]

    # Metric cards
    total     = len(filtered)
    tier_a    = (filtered["priority_tier"] == "A").sum()
    avg_score = filtered["affinity_score"].mean()
    avg_vol   = filtered["total_claims"].mean()

    st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Physicians in View</div>
                <div class="metric-value">{total:,}</div>
                <div class="metric-sub">of 13,027 total targets</div>
            </div>
            <div class="metric-card green">
                <div class="metric-label">Tier A Targets</div>
                <div class="metric-value">{tier_a:,}</div>
                <div class="metric-sub">highest priority</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Affinity Score</div>
                <div class="metric-value">{avg_score:.3f}</div>
                <div class="metric-sub">0 = low · 1 = high</div>
            </div>
            <div class="metric-card amber">
                <div class="metric-label">Avg GLP-1 Volume</div>
                <div class="metric-value">{avg_vol:.0f}</div>
                <div class="metric-sub">claims per physician</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Table
    st.markdown(
        '<div class="section-header">Ranked Physician Targets</div>',
        unsafe_allow_html=True
    )
    display = filtered.head(top_n)[[
        "target_rank","first_name","last_name","specialty",
        "city","total_claims","priority_tier",
        "affinity_score","similarity_score","volume_score"
    ]].rename(columns={
        "target_rank":      "Rank",
        "first_name":       "First",
        "last_name":        "Last",
        "specialty":        "Specialty",
        "city":             "City",
        "total_claims":     "GLP-1 Claims",
        "priority_tier":    "Tier",
        "affinity_score":   "Affinity",
        "similarity_score": "Similarity",
        "volume_score":     "Volume",
    })

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=480,
        column_config={
            "Affinity": st.column_config.ProgressColumn(
                "Affinity Score", min_value=0, max_value=1, format="%.3f"
            ),
            "Similarity": st.column_config.ProgressColumn(
                "Drug Similarity", min_value=0, max_value=1, format="%.3f"
            ),
            "Volume": st.column_config.ProgressColumn(
                "Volume Score", min_value=0, max_value=1, format="%.3f"
            ),
            "GLP-1 Claims": st.column_config.NumberColumn(
                "GLP-1 Claims", format="%d"
            ),
            "Rank": st.column_config.NumberColumn("Rank", format="%d"),
        }
    )

    # Charts
    st.markdown("---")
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown(
            '<div class="section-header">Top Cities by Target Count</div>',
            unsafe_allow_html=True
        )
        city_data = (
            filtered.groupby("city")["npi"].count()
            .sort_values(ascending=True).tail(10)
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("white")
        bars = ax.barh(city_data.index, city_data.values,
                       color="#1B4F8A", alpha=0.85, height=0.6)
        ax.set_xlabel("Number of Targets", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="x", alpha=0.3, ls="--")
        for bar, val in zip(bars, city_data.values):
            ax.text(bar.get_width() + 0.5,
                    bar.get_y() + bar.get_height()/2,
                    str(val), va="center", fontsize=9, color="#64748B")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with ch2:
        st.markdown(
            '<div class="section-header">Target Mix by Specialty</div>',
            unsafe_allow_html=True
        )
        spec_data = (
            filtered.groupby("specialty")["npi"].count()
            .sort_values(ascending=True).tail(10)
        )
        RELEVANT = {"Endocrinology","Internal Medicine","Family Practice",
                    "Obesity Medicine","Geriatric Medicine","Cardiology"}
        colors = ["#1A7F5A" if s in RELEVANT else "#94A3B8"
                  for s in spec_data.index]
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("white")
        ax.barh(spec_data.index, spec_data.values,
                color=colors, alpha=0.85, height=0.6)
        ax.set_xlabel("Number of Targets", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="x", alpha=0.3, ls="--")
        legend = [
            mpatches.Patch(color="#1A7F5A", label="Relevant specialty"),
            mpatches.Patch(color="#94A3B8", label="Other")
        ]
        ax.legend(handles=legend, fontsize=8, loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PHYSICIAN PROFILE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Physician Profile":

    st.markdown("""
        <div class="header-bar">
            <div>
                <div class="header-title">Physician Profile</div>
                <div class="header-sub">
                    Targeting rationale and prescribing detail
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    search = st.text_input(
        "Search by last name",
        placeholder="e.g. Bhatt, Jimenez, Shaw..."
    )

    results = (
        targets[targets["last_name"].str.contains(search, case=False, na=False)]
        if search else targets.head(50)
    )

    if len(results) == 0:
        st.warning("No physicians found.")
        st.stop()

    options = [
        f"#{int(r.target_rank):,}  ·  {r.first_name} {r.last_name}"
        f"  ·  {r.specialty}  ·  {r.city}"
        for _, r in results.head(20).iterrows()
    ]
    selected_idx = st.selectbox(
        "Select physician", range(len(options)),
        format_func=lambda i: options[i]
    )
    physician = results.iloc[selected_idx]

    st.markdown("---")

    TIER_COLOR = {"A": "#166534", "B": "#854D0E", "C": "#991B1B"}
    TIER_BG    = {"A": "#DCFCE7", "B": "#FEF9C3", "C": "#FEE2E2"}

    st.markdown(f"""
        <div class="physician-card">
            <div style="display:flex; justify-content:space-between;
                        align-items:flex-start;">
                <div>
                    <div class="physician-name">
                        {physician.first_name} {physician.last_name}
                    </div>
                    <div class="physician-meta">
                        {physician.specialty} · {physician.city}, CA
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="background:{TIER_BG[physician.priority_tier]};
                                color:{TIER_COLOR[physician.priority_tier]};
                                padding:4px 14px; border-radius:16px;
                                font-size:13px; font-weight:600;
                                display:inline-block;">
                        Priority Tier {physician.priority_tier}
                    </div>
                    <div style="font-size:13px; color:#64748B; margin-top:8px;">
                        Rank <b style="color:#0A2342;">
                            #{int(physician.target_rank):,}
                        </b> of 13,027
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3)

    def score_card(col, label, value, description, color):
        pct = int(value * 100)
        col.markdown(f"""
            <div style="background:white; border:1px solid #E2E8F0;
                        border-radius:8px; padding:16px 20px;">
                <div style="font-size:11px; font-weight:600; color:#64748B;
                            text-transform:uppercase; letter-spacing:0.8px;">
                    {label}
                </div>
                <div style="font-size:32px; font-weight:700; color:#0A2342;
                            margin:6px 0 4px;">{value:.3f}</div>
                <div class="score-bar-bg">
                    <div class="score-bar-fill"
                         style="width:{pct}%; background:{color};"></div>
                </div>
                <div style="font-size:11px; color:#94A3B8; margin-top:6px;">
                    {description}
                </div>
            </div>
        """, unsafe_allow_html=True)

    score_card(sc1, "Overall Affinity",    physician.affinity_score,
               "Combined targeting score", "#1B4F8A")
    score_card(sc2, "Drug Mix Similarity", physician.similarity_score,
               "vs high-volume Tirzepatide prescribers", "#1A7F5A")
    score_card(sc3, "Volume Score",        physician.volume_score,
               "GLP-1 prescribing volume percentile", "#D97706")

    st.markdown("<br>", unsafe_allow_html=True)

    d1, d2 = st.columns([3, 2])

    with d1:
        st.markdown(
            '<div class="section-header">GLP-1 Prescribing Breakdown</div>',
            unsafe_allow_html=True
        )
        doc_rx = interactions[
            interactions["npi"] == physician["npi"]
        ].sort_values("total_claims", ascending=True)

        if len(doc_rx) > 0:
            fig, ax = plt.subplots(
                figsize=(7, max(3, len(doc_rx) * 0.7))
            )
            fig.patch.set_facecolor("white")
            colors = ["#EF9F27" if d == "TIRZEPATIDE" else "#1B4F8A"
                      for d in doc_rx["drug_generic"]]
            bars = ax.barh(doc_rx["drug_generic"], doc_rx["total_claims"],
                           color=colors, alpha=0.9, height=0.5)
            ax.set_xlabel("Total Claims (2023)", fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.grid(axis="x", alpha=0.3, ls="--")
            for bar, val in zip(bars, doc_rx["total_claims"]):
                ax.text(bar.get_width() + 0.5,
                        bar.get_y() + bar.get_height()/2,
                        f"{int(val):,}", va="center", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with d2:
        st.markdown(
            '<div class="section-header">Why Target This Physician?</div>',
            unsafe_allow_html=True
        )
        RELEVANT_SPECS = {
            "Endocrinology","Internal Medicine","Family Practice",
            "Obesity Medicine","Geriatric Medicine","Cardiology"
        }
        insights = []
        if physician.affinity_score > 0.45:
            insights.append("🎯 High affinity — strong match to "
                            "Tirzepatide adopter profile")
        if physician.total_claims > physician_df["total_claims"].quantile(0.75):
            insights.append(f"📈 Top-quartile prescriber — "
                            f"{int(physician.total_claims):,} GLP-1 claims")
        if physician.specialty in RELEVANT_SPECS:
            insights.append(f"✓ {physician.specialty} — primary "
                            f"target specialty")
        if physician.similarity_score > 0.75:
            insights.append("🔬 Drug mix mirrors high-volume "
                            "Tirzepatide prescribers")
        if not insights:
            insights.append("ℹ️ Moderate affinity — secondary "
                            "outreach recommended")

        for text in insights:
            st.markdown(
                f'<div class="insight-box green">{text}</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">Quick Stats</div>',
            unsafe_allow_html=True
        )
        st.markdown(f"""
            <table style="width:100%; font-size:13px; border-collapse:collapse;">
                <tr style="border-bottom:1px solid #E2E8F0;">
                    <td style="padding:8px 0; color:#64748B;">Total GLP-1 Claims</td>
                    <td style="text-align:right; font-weight:600; color:#0A2342;">
                        {int(physician.total_claims):,}
                    </td>
                </tr>
                <tr style="border-bottom:1px solid #E2E8F0;">
                    <td style="padding:8px 0; color:#64748B;">Drugs Prescribed</td>
                    <td style="text-align:right; font-weight:600; color:#0A2342;">
                        {len(doc_rx)}
                    </td>
                </tr>
                <tr style="border-bottom:1px solid #E2E8F0;">
                    <td style="padding:8px 0; color:#64748B;">On Tirzepatide?</td>
                    <td style="text-align:right; font-weight:600; color:#DC2626;">
                        No — opportunity
                    </td>
                </tr>
                <tr>
                    <td style="padding:8px 0; color:#64748B;">Priority Tier</td>
                    <td style="text-align:right; font-weight:600;
                                color:{TIER_COLOR[physician.priority_tier]};">
                        Tier {physician.priority_tier}
                    </td>
                </tr>
            </table>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Validation":

    st.markdown("""
        <div class="header-bar">
            <div>
                <div class="header-title">
                    Model Validation & A/B Test Results
                </div>
                <div class="header-sub">
                    Retrospective hypothesis test · 500 physicians per group
                </div>
            </div>
            <div class="header-badge">✓ p &lt; 0.000001</div>
        </div>
    """, unsafe_allow_html=True)

    ab = ab_results.iloc[0]

    st.markdown(
        '<div class="section-header">A/B Test Summary</div>',
        unsafe_allow_html=True
    )

    r1, r2, r3, r4, r5 = st.columns(5)
    for col, number, label in [
        (r1, "82%",  "Absolute lift<br>over random"),
        (r2, "455%", "Relative lift<br>vs control"),
        (r3, "26.4", "Z-statistic"),
        (r4, "8.6x", "Top decile<br>lift"),
        (r5, "$76M", "Est. revenue<br>impact (CA)"),
    ]:
        col.markdown(f"""
            <div class="ab-card">
                <div class="ab-number">{number}</div>
                <div class="ab-label">{label}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.success(
        "✓  Statistically significant at p < 0.000001  ·  "
        "Effect size (Cohen\'s h): 3.02  ·  "
        "Study is 162x overpowered — result is robust"
    )

    st.markdown("---")
    st.markdown(
        '<div class="section-header">Cumulative Gain & Decile Lift</div>',
        unsafe_allow_html=True
    )

    # Rebuild gain curve from data
    ADOPTION_THRESH = physician_df["total_claims"].quantile(0.75)
    all_phys = physician_df.copy()
    score_map = targets.set_index("npi")["affinity_score"].to_dict()
    all_phys["affinity_score"] = all_phys["npi"].map(score_map).fillna(0)
    all_phys["adopted"] = (
        all_phys["total_claims"] >= ADOPTION_THRESH
    ).astype(int)
    total_adopters = all_phys["adopted"].sum()

    model_sorted  = all_phys.sort_values("affinity_score", ascending=False)
    model_cumsum  = model_sorted["adopted"].cumsum().values
    pct_pop       = np.arange(1, len(model_sorted)+1) / len(model_sorted) * 100
    model_pct_cap = model_cumsum / total_adopters * 100
    random_cap    = pct_pop.copy()
    perfect_cap   = np.minimum(
        pct_pop / (total_adopters / len(all_phys) * 100) * 100, 100
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.plot(pct_pop, model_pct_cap,  color="#1B4F8A", lw=2.5, label="Our model")
    ax.plot(pct_pop, random_cap,     color="#94A3B8", lw=1.5, ls="--",
            label="Random targeting")
    ax.plot(pct_pop, perfect_cap,    color="#1A7F5A", lw=1.5, ls=":",
            label="Perfect model")
    ax.fill_between(pct_pop, random_cap, model_pct_cap, alpha=0.1, color="#1B4F8A")
    idx20 = int(0.20 * len(model_sorted))
    ax.annotate(
        "Top 20% captures\n{:.0f}% of adopters".format(model_pct_cap[idx20]),
        xy=(20, model_pct_cap[idx20]),
        xytext=(35, model_pct_cap[idx20] - 12),
        fontsize=9, color="#1B4F8A", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#1B4F8A")
    )
    ax.set_xlabel("% of Physicians Called On")
    ax.set_ylabel("% of High-Value Adopters Captured")
    ax.set_title("Cumulative Gain Curve", fontweight="600")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3, ls="--")

    ax = axes[1]
    decile_model  = []
    decile_random = []
    for d in range(1, 11):
        lo = int((d-1) * 0.10 * len(model_sorted))
        hi = int(d     * 0.10 * len(model_sorted))
        decile_model.append(model_sorted.iloc[lo:hi]["adopted"].mean() * 100)
        decile_random.append(all_phys["adopted"].mean() * 100)

    x = np.arange(10)
    ax.bar(x - 0.2, decile_model,  0.4, color="#1B4F8A", alpha=0.85,
           label="Model decile")
    ax.bar(x + 0.2, decile_random, 0.4, color="#94A3B8", alpha=0.6,
           label="Random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{i}" for i in range(1, 11)])
    ax.set_xlabel("Physician Decile (D1 = highest affinity)")
    ax.set_ylabel("% High-Value Adopters in Decile")
    ax.set_title("Adoption Rate by Model Decile", fontweight="600")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3, ls="--", axis="y")
    lift_d1 = decile_model[0] / decile_random[0]
    ax.text(0 - 0.2, decile_model[0] + 1,
            f"{lift_d1:.1f}x", ha="center",
            fontsize=9, color="#1B4F8A", fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown(
        '<div class="section-header">Methodology</div>',
        unsafe_allow_html=True
    )
    st.markdown("""
        <div style="background:white; border:1px solid #E2E8F0;
                    border-radius:8px; padding:24px; font-size:13px;
                    color:#334155; line-height:1.8;">
            <b style="color:#0A2342;">Data</b><br>
            CMS Medicare Part D 2023 — California GLP-1 prescribers.
            27,781 prescription records across 15,289 physicians.<br><br>
            <b style="color:#0A2342;">Model</b><br>
            Volume-weighted cosine similarity. Identified top 25% of existing
            Tirzepatide prescribers by volume (n=570), computed a centroid
            vector in drug-feature space, scored all 13,027 non-prescribers
            by cosine similarity weighted 50/50 with overall GLP-1 volume.<br><br>
            <b style="color:#0A2342;">Validation</b><br>
            Retrospective A/B test. Treatment = top 500 model-ranked physicians.
            Control = 500 random from same pool. Two-proportion z-test,
            one-sided alternative. Adoption threshold: top-quartile GLP-1
            prescribing (77+ claims).<br><br>
            <b style="color:#0A2342;">Limitations</b><br>
            Medicare only — excludes commercial insurance (~40% of market).
            18-month CMS data lag. Single-year snapshot cannot capture
            physician switching behaviour over time.
        </div>
    """, unsafe_allow_html=True)
