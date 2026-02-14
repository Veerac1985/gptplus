# %% [markdown]
# # A/B Test Analysis: Free First Month of ChatGPT Plus
# **Data Science Take-Home Assessment**
#
# This notebook contains the complete, reproducible analysis for an A/B test experiment
# offering users a "free first month" promotion to improve paid plan signups.
#
# **Experiment Design:**
# - Users who haven't signed up for a paid plan after 3 months of use are randomized
# - **Control:** No offer (business as usual)
# - **Treatment:** "First month free" promotion
# - **Split:** 2:1 control/treatment
#
# **Questions Addressed:**
# 0. Data Overview & Descriptive Statistics
# 1. Impact on paid plan signups
# 2. Impact on paid plan retention
# 3. Statistical reliability of conclusions
# 4. Day-of-week heterogeneity in treatment effect
# 5. Experiment duration adequacy
# 6. ROI model & break-even analysis
# 7. Ship / no-ship recommendation
# 8. Follow-up targeting strategy

# %% [markdown]
# ---
# ## Setup & Dependencies
# Run this cell first to install required packages (if in Google Colab).

# %%
# !pip install pandas numpy scipy statsmodels plotly -q

import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import NormalIndPower
import warnings
warnings.filterwarnings("ignore")
import os

IS_NOTEBOOK = False
try:
    get_ipython()
    IS_NOTEBOOK = True
except NameError:
    pass

if not IS_NOTEBOOK:
    import plotly.io as pio
    pio.renderers.default = "png"

def show_fig(fig):
    """Display figure — works in Colab/Jupyter and as a plain script."""
    if IS_NOTEBOOK:
        fig.show()
    else:
        print("[Chart rendered — open in Colab/Jupyter to see interactive plots]")

print("All packages loaded successfully.")

# %% [markdown]
# ---
# ## Load & Prepare Data
# Upload `experiment_data.csv` to your Colab/Jupyter environment, or adjust the path below.

# %%
# ── Load the dataset ──
# If running in Google Colab, uncomment the next two lines to upload the file:
# from google.colab import files
# uploaded = files.upload()

df = pd.read_csv("experiment_data.csv")

# ── Feature engineering ──
df["assignment_date"] = pd.to_datetime(df["assignment_date"])
df["paid_signup_date"] = pd.to_datetime(df["paid_signup_date"])
df["signed_up"] = df["paid_signup_date"].notna()
df["day_of_week"] = df["assignment_date"].dt.day_name()
df["days_to_signup"] = (df["paid_signup_date"] - df["assignment_date"]).dt.days
df["group"] = df["treatment"].map({0: "control", 1: "treatment"})

# ── Split into groups ──
control = df[df["treatment"] == 0]
treatment = df[df["treatment"] == 1]

print(f"Dataset loaded: {len(df):,} users")
print(f"  Control:   {len(control):,} users")
print(f"  Treatment: {len(treatment):,} users")
print(f"  Columns:   {list(df.columns)}")

# %% [markdown]
# ---
# # Q0: Data Overview & Descriptive Statistics
# Before diving into the experiment results, we thoroughly examine the dataset
# to understand its structure, distributions, and quality.

# %%
# ── Dataset summary ──
n_total = len(df)
n_ctrl = len(control)
n_treat = len(treatment)
n_signed = int(df["signed_up"].sum())
n_canceled = int(df["paid_plan_canceled"].sum())
date_min = df["assignment_date"].min().strftime("%Y-%m-%d")
date_max = df["assignment_date"].max().strftime("%Y-%m-%d")
exp_dur = (df["assignment_date"].max() - df["assignment_date"].min()).days

print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(f"Total users:         {n_total:,}")
print(f"Control group:       {n_ctrl:,} ({n_ctrl/n_total:.1%})")
print(f"Treatment group:     {n_treat:,} ({n_treat/n_total:.1%})")
print(f"Experiment window:   {date_min} to {date_max} ({exp_dur} days)")
print(f"Total signups:       {n_signed:,} ({n_signed/n_total:.1%})")
print(f"Total cancellations: {n_canceled:,}")
print(f"Platforms:           {df['assigned_on_platform'].nunique()} — {sorted(df['assigned_on_platform'].unique())}")
print(f"Time zones:          {df['time_zone'].nunique()} — {sorted(df['time_zone'].unique())}")
print(f"Missing values:      {df.isnull().sum().sum():,} (only in paid_signup_date — expected for non-signers)")
print()
print("Column types:")
print(df.dtypes.to_string())

# %%
# ── Treatment vs Control split ──
fig_split = go.Figure(data=[go.Pie(
    labels=["Control", "Treatment"],
    values=[n_ctrl, n_treat],
    hole=0.45,
    marker_colors=["#636EFA", "#EF553B"],
    textinfo="label+value+percent",
    textposition="outside",
)])
fig_split.update_layout(title="Group Assignment Distribution", height=400, showlegend=False,
                        margin=dict(t=60, b=60))
show_fig(fig_split)

print(f"\n2:1 ratio: {n_ctrl:,} control ({n_ctrl/n_total:.1%}) vs {n_treat:,} treatment ({n_treat/n_total:.1%})")

# %%
# ── Platform distribution ──
print("Platform Distribution:")
print(df.groupby("assigned_on_platform").agg(
    total=("signed_up", "count"),
    signups=("signed_up", "sum"),
    signup_rate=("signed_up", "mean"),
    cancel_rate=("paid_plan_canceled", "mean"),
).to_string())

plat_counts = df["assigned_on_platform"].value_counts()
fig_plat = go.Figure(data=[go.Pie(
    labels=plat_counts.index, values=plat_counts.values,
    hole=0.45, marker_colors=["#636EFA", "#EF553B"],
    textinfo="label+value+percent", textposition="outside",
)])
fig_plat.update_layout(title="Users by Platform", height=380, showlegend=False,
                       margin=dict(t=60, b=60))
show_fig(fig_plat)

# %%
# ── Time zone distribution ──
print("\nTime Zone Distribution:")
tz_summary = df.groupby("time_zone").agg(
    total=("signed_up", "count"),
    signup_rate=("signed_up", "mean"),
    cancel_rate=("paid_plan_canceled", "mean"),
).sort_values("total", ascending=False)
print(tz_summary.to_string())

tz_counts = df["time_zone"].value_counts().sort_values(ascending=True)
fig_tz = go.Figure(go.Bar(
    x=tz_counts.values, y=tz_counts.index, orientation="h",
    marker_color=px.colors.qualitative.Set2[:len(tz_counts)],
    text=[f"{v:,} ({v/n_total:.1%})" for v in tz_counts.values],
    textposition="outside",
))
fig_tz.update_layout(title="Users by Time Zone", height=380,
                     xaxis_title="Number of Users", yaxis_title="Time Zone",
                     margin=dict(r=120))
show_fig(fig_tz)

# %%
# ── Daily assignment pattern ──
daily_assignments = df.groupby([df["assignment_date"].dt.date, "group"]).size().reset_index(name="count")
daily_assignments.columns = ["date", "group", "count"]
fig_daily_assign = px.line(
    daily_assignments, x="date", y="count", color="group",
    title="Daily User Assignments Over Time",
    color_discrete_map={"control": "#636EFA", "treatment": "#EF553B"},
)
fig_daily_assign.update_layout(height=350)
show_fig(fig_daily_assign)

daily_total = df.groupby(df["assignment_date"].dt.date).size()
print(f"\nAvg daily assignment: {daily_total.mean():.0f} users/day (range: {daily_total.min()} – {daily_total.max()})")

# %%
# ── Day-of-week assignment ──
dow_order_eda = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow_counts = df["day_of_week"].value_counts().reindex(dow_order_eda)
fig_dow_dist = go.Figure(go.Bar(
    x=dow_counts.index, y=dow_counts.values,
    marker_color=["#ff9800" if d in ["Saturday", "Sunday"] else "#636EFA" for d in dow_counts.index],
    text=[f"{v:,}" for v in dow_counts.values], textposition="outside",
))
fig_dow_dist.update_layout(title="Users Assigned by Day of Week", height=380,
                           xaxis_title="Day", yaxis_title="Users", margin=dict(t=60))
show_fig(fig_dow_dist)

weekend_pct = df["day_of_week"].isin(["Saturday", "Sunday"]).mean()
print(f"\nWeekends account for ~{weekend_pct:.0%} of assignments")

# %%
# ── Outcome distributions ──
print("Outcome Summary:")
print(f"  Overall signup rate:      {n_signed/n_total:.2%}")
cancel_among_signups = df[df["signed_up"]]
n_cancel_signups = int(cancel_among_signups["paid_plan_canceled"].sum())
print(f"  Cancel rate (among signups): {n_cancel_signups/n_signed:.1%}")

# ── Days to signup descriptive stats ──
signups_eda = df[df["signed_up"]].copy()
print("\nDays to Signup — Descriptive Statistics:")
print(signups_eda.groupby("group")["days_to_signup"].describe().to_string())

fig_dts = px.histogram(
    signups_eda, x="days_to_signup", color="group",
    barmode="overlay", nbins=40,
    title="Days from Assignment to Signup",
    color_discrete_map={"control": "#636EFA", "treatment": "#EF553B"},
    opacity=0.7,
)
fig_dts.update_layout(height=350)
show_fig(fig_dts)

# %%
# ── Cross-tabulations ──
print("\nCross-Tabulation: Group × Platform")
cross_plat = df.groupby(["group", "assigned_on_platform"]).agg(
    users=("signed_up", "count"),
    signups=("signed_up", "sum"),
    signup_rate=("signed_up", "mean"),
    cancel_rate=("paid_plan_canceled", "mean"),
).reset_index()
print(cross_plat.to_string(index=False))

print("\nCross-Tabulation: Group × Time Zone")
cross_tz = df.groupby(["group", "time_zone"]).agg(
    users=("signed_up", "count"),
    signups=("signed_up", "sum"),
    signup_rate=("signed_up", "mean"),
    cancel_rate=("paid_plan_canceled", "mean"),
).reset_index()
print(cross_tz.to_string(index=False))

# %%
# ── Raw data sample ──
print("\nFirst 10 rows of the dataset:")
df.head(10)

# %% [markdown]
# ---
# # Q1: Impact on Paid Plan Signups
#
# **Method:** Two-proportion z-test comparing signup rates between control and treatment groups.
#
# **Formulas:**
# - $\hat{p}_T = \frac{\text{Treatment signups}}{n_T}$, $\hat{p}_C = \frac{\text{Control signups}}{n_C}$
# - $\Delta = \hat{p}_T - \hat{p}_C$ (absolute lift)
# - $\hat{p}_{\text{pool}} = \frac{x_T + x_C}{n_T + n_C}$
# - $z = \frac{\hat{p}_T - \hat{p}_C}{\sqrt{\hat{p}_{\text{pool}}(1-\hat{p}_{\text{pool}})\left(\frac{1}{n_T}+\frac{1}{n_C}\right)}}$
# - $SE = \sqrt{\frac{\hat{p}_T(1-\hat{p}_T)}{n_T} + \frac{\hat{p}_C(1-\hat{p}_C)}{n_C}}$ (unpooled, for CI)
# - $CI_{95\%} = \Delta \pm 1.96 \times SE$

# %%
# ── Signup rates ──
ctrl_signups = control["signed_up"].sum()
treat_signups = treatment["signed_up"].sum()
ctrl_rate = control["signed_up"].mean()
treat_rate = treatment["signed_up"].mean()
abs_lift = treat_rate - ctrl_rate
rel_lift = abs_lift / ctrl_rate if ctrl_rate > 0 else 0

# ── Two-proportion z-test ──
count = np.array([treat_signups, ctrl_signups])
nobs = np.array([len(treatment), len(control)])
z_stat, p_val = proportions_ztest(count, nobs, alternative="two-sided")

# ── Confidence interval (unpooled SE) ──
se_diff = np.sqrt(treat_rate*(1-treat_rate)/len(treatment) + ctrl_rate*(1-ctrl_rate)/len(control))
ci_diff = 1.96 * se_diff

# ── Pooled proportion ──
pooled_p_val = (treat_signups + ctrl_signups) / (len(treatment) + len(control))

print("=" * 60)
print("Q1: IMPACT ON PAID PLAN SIGNUPS")
print("=" * 60)
print(f"Control signup rate:  {ctrl_rate:.4f} ({ctrl_rate:.2%})  [{int(ctrl_signups):,} / {len(control):,}]")
print(f"Treatment signup rate:{treat_rate:.4f} ({treat_rate:.2%})  [{int(treat_signups):,} / {len(treatment):,}]")
print(f"Absolute lift:        {abs_lift:.4f} ({abs_lift:.2%})")
print(f"Relative lift:        {rel_lift:.1%}")
print()
print(f"Pooled proportion:    {pooled_p_val:.4f}")
print(f"Z-statistic:          {z_stat:.4f}")
print(f"P-value:              {p_val:.6f}")
print(f"SE (unpooled):        {se_diff:.6f}")
print(f"95% CI for diff:      [{abs_lift - ci_diff:.4f}, {abs_lift + ci_diff:.4f}]")
print(f"                      [{abs_lift - ci_diff:.2%}, {abs_lift + ci_diff:.2%}]")
print()
sig_label = "STATISTICALLY SIGNIFICANT" if p_val < 0.05 else "NOT statistically significant"
print(f"Result: {sig_label} at alpha = 0.05")

# %%
# ── Signup rate bar chart ──
fig_signup = go.Figure()
fig_signup.add_trace(go.Bar(
    x=["Control", "Treatment"],
    y=[ctrl_rate * 100, treat_rate * 100],
    text=[f"{ctrl_rate:.2%}", f"{treat_rate:.2%}"],
    textposition="outside",
    marker_color=["#636EFA", "#EF553B"],
    width=0.5,
))
fig_signup.update_layout(
    title="Paid Plan Signup Rate by Group",
    yaxis_title="Signup Rate (%)",
    yaxis=dict(range=[0, max(ctrl_rate, treat_rate) * 100 * 1.5]),
    showlegend=False, height=400, margin=dict(t=60),
)
fig_signup.update_traces(cliponaxis=False)
show_fig(fig_signup)

# %%
# ── Days to signup histogram ──
signups_only = df[df["signed_up"]]
fig_time = px.histogram(
    signups_only, x="days_to_signup", color="group",
    barmode="overlay", nbins=30,
    title="Days from Assignment to Signup",
    color_discrete_map={"control": "#636EFA", "treatment": "#EF553B"},
    opacity=0.7,
)
fig_time.update_layout(height=350)
show_fig(fig_time)

# %%
# ── Daily signup rate over time ──
daily_signups = df.groupby([df["assignment_date"].dt.date, "group"])["signed_up"].mean().reset_index()
daily_signups.columns = ["date", "group", "signup_rate"]
fig_daily = px.line(
    daily_signups, x="date", y="signup_rate", color="group",
    title="Daily Signup Rate Over Time",
    color_discrete_map={"control": "#636EFA", "treatment": "#EF553B"},
)
fig_daily.update_layout(height=350)
show_fig(fig_daily)

# %% [markdown]
# ---
# # Q2: Impact on Paid Plan Retention
#
# **Method:** Among users who signed up, compare cancellation rates between groups.
#
# **Formulas:**
# - $\hat{c}_T = \frac{\text{Treatment cancels}}{\text{Treatment signups}}$
# - $\hat{c}_C = \frac{\text{Control cancels}}{\text{Control signups}}$
# - Same z-test framework as Q1, applied to cancellation proportions

# %%
ctrl_subs = control[control["signed_up"]].copy()
treat_subs = treatment[treatment["signed_up"]].copy()

ctrl_cancel_rate = ctrl_subs["paid_plan_canceled"].mean()
treat_cancel_rate = treat_subs["paid_plan_canceled"].mean()
ctrl_retain_rate = 1 - ctrl_cancel_rate
treat_retain_rate = 1 - treat_cancel_rate

cancel_count = np.array([int(treat_subs["paid_plan_canceled"].sum()), int(ctrl_subs["paid_plan_canceled"].sum())])
cancel_nobs = np.array([len(treat_subs), len(ctrl_subs)])
z_cancel, p_cancel = proportions_ztest(cancel_count, cancel_nobs, alternative="two-sided")

cancel_diff = treat_cancel_rate - ctrl_cancel_rate
se_cancel = np.sqrt(
    treat_cancel_rate*(1-treat_cancel_rate)/max(len(treat_subs),1) +
    ctrl_cancel_rate*(1-ctrl_cancel_rate)/max(len(ctrl_subs),1)
)
cancel_ci = 1.96 * se_cancel

print("=" * 60)
print("Q2: IMPACT ON PAID PLAN RETENTION")
print("=" * 60)
print(f"Control subscribers:    {len(ctrl_subs):,}")
print(f"Treatment subscribers:  {len(treat_subs):,}")
print()
print(f"Control cancel rate:    {ctrl_cancel_rate:.4f} ({ctrl_cancel_rate:.1%})")
print(f"Treatment cancel rate:  {treat_cancel_rate:.4f} ({treat_cancel_rate:.1%})")
print(f"Control retention:      {ctrl_retain_rate:.1%}")
print(f"Treatment retention:    {treat_retain_rate:.1%}")
print()
print(f"Cancel rate difference: {cancel_diff:+.4f} ({cancel_diff:+.1%})")
print(f"Z-statistic:            {z_cancel:.4f}")
print(f"P-value:                {p_cancel:.4f}")
print(f"95% CI for diff:        [{cancel_diff - cancel_ci:.4f}, {cancel_diff + cancel_ci:.4f}]")
cancel_sig = "STATISTICALLY SIGNIFICANT" if p_cancel < 0.05 else "NOT statistically significant"
print(f"Result: {cancel_sig}")

# %%
fig_ret = make_subplots(rows=1, cols=2,
                        subplot_titles=["Cancellation Rate by Group", "Retention Breakdown"])
fig_ret.add_trace(go.Bar(
    x=["Control", "Treatment"],
    y=[ctrl_cancel_rate * 100, treat_cancel_rate * 100],
    text=[f"{ctrl_cancel_rate:.1%}", f"{treat_cancel_rate:.1%}"],
    textposition="outside",
    marker_color=["#636EFA", "#EF553B"], width=0.5, showlegend=False,
), row=1, col=1)
fig_ret.add_trace(go.Bar(
    name="Retained", x=["Control", "Treatment"],
    y=[ctrl_retain_rate * 100, treat_retain_rate * 100], marker_color="#4CAF50",
), row=1, col=2)
fig_ret.add_trace(go.Bar(
    name="Canceled", x=["Control", "Treatment"],
    y=[ctrl_cancel_rate * 100, treat_cancel_rate * 100], marker_color="#f44336",
), row=1, col=2)
fig_ret.update_layout(barmode="stack", height=420, margin=dict(t=60))
fig_ret.update_yaxes(title_text="Cancellation Rate (%)",
                     range=[0, max(ctrl_cancel_rate, treat_cancel_rate) * 100 * 1.35], row=1, col=1)
fig_ret.update_yaxes(title_text="% of Signups", row=1, col=2)
fig_ret.update_traces(cliponaxis=False)
show_fig(fig_ret)

# %% [markdown]
# ---
# # Q3: Statistical Reliability of Our Conclusions
#
# **Method:** Power analysis, effect size calculation, and covariate balance check.
#
# **Formulas:**
# - Cohen's h: $h = \frac{\Delta}{\sqrt{\hat{p}_{\text{pool}}(1-\hat{p}_{\text{pool}})}}$
# - Power: $P(\text{Reject } H_0 \mid H_1 \text{ true}) = 1 - \beta$
# - Type I error: $\alpha = 0.05$
# - Type II error: $\beta = 1 - \text{Power}$

# %%
power_analysis = NormalIndPower()
pooled_p = df["signed_up"].mean()
effect_size = abs_lift / np.sqrt(pooled_p * (1 - pooled_p))
power = power_analysis.solve_power(
    effect_size=effect_size,
    nobs1=len(treatment),
    alpha=0.05,
    ratio=len(control)/len(treatment),
    alternative="two-sided",
)

try:
    mde_h = power_analysis.solve_power(
        effect_size=None, nobs1=len(treatment), alpha=0.05,
        ratio=len(control)/len(treatment), power=0.8, alternative="two-sided",
    )
    mde_abs = mde_h * np.sqrt(pooled_p * (1 - pooled_p))
except Exception:
    mde_h, mde_abs = None, None

print("=" * 60)
print("Q3: STATISTICAL RELIABILITY")
print("=" * 60)
print(f"Cohen's h (effect size):   {effect_size:.4f}")
print(f"Statistical power:         {power:.4f} ({power:.1%})")
print(f"P-value:                   {p_val:.6f}")
print(f"95% CI width:              ±{ci_diff:.4f} (±{ci_diff:.2%})")
print(f"Type I error (alpha):      0.05")
print(f"Type II error (beta):      {1-power:.4f}")
if mde_abs is not None:
    print(f"MDE at 80% power:          {mde_abs:.4f} ({mde_abs:.2%})")
print()
print(f"Assessment: {'Well-powered' if power >= 0.8 else 'Potentially underpowered'} for observed effect")

# %%
# ── Balance check ──
print("\nBALANCE CHECK: Pre-Treatment Covariates")
print("-" * 50)
balance_data = []
for col_name, label in [("assigned_on_platform", "Platform"), ("time_zone", "Time Zone")]:
    for val in sorted(df[col_name].unique()):
        c_frac = (control[col_name] == val).mean()
        t_frac = (treatment[col_name] == val).mean()
        balance_data.append({
            "Covariate": f"{label}: {val}",
            "Control": c_frac,
            "Treatment": t_frac,
            "Difference": t_frac - c_frac,
        })

bal_df = pd.DataFrame(balance_data)
bal_df["Control"] = bal_df["Control"].apply(lambda x: f"{x:.2%}")
bal_df["Treatment"] = bal_df["Treatment"].apply(lambda x: f"{x:.2%}")
bal_df["Difference"] = bal_df["Difference"].apply(lambda x: f"{x:+.2%}")
print(bal_df.to_string(index=False))

fig_bal = go.Figure()
ctrl_vals = [bd["Control"] for bd in balance_data]  # raw floats
treat_vals = [bd["Treatment"] for bd in balance_data]
fig_bal.add_trace(go.Bar(name="Control", x=[bd["Covariate"] for bd in balance_data],
                         y=[bd["Control"] for bd in balance_data], marker_color="#636EFA"))
fig_bal.add_trace(go.Bar(name="Treatment", x=[bd["Covariate"] for bd in balance_data],
                         y=[bd["Treatment"] for bd in balance_data], marker_color="#EF553B"))
fig_bal.update_layout(barmode="group", title="Covariate Balance Between Groups",
                      yaxis_title="Proportion", height=400, xaxis_tickangle=-30)
show_fig(fig_bal)

# %% [markdown]
# ---
# # Q4: Treatment Effect by Day of Week
#
# **Method:** Linear Probability Model (LPM) with interaction terms + joint F-test.
#
# **Model:**
# $$Y_i = \beta_0 + \beta_1 T_i + \sum_{d=2}^{7} \gamma_d D_{id} + \sum_{d=2}^{7} \delta_d (T_i \times D_{id}) + \varepsilon_i$$
#
# **Hypothesis:**
# $$H_0: \delta_2 = \delta_3 = \cdots = \delta_7 = 0$$
#
# We use HC1 (heteroscedasticity-robust) standard errors since the outcome is binary.

# %%
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow_stats = []
for dow in dow_order:
    c_dow = control[control["day_of_week"] == dow]
    t_dow = treatment[treatment["day_of_week"] == dow]
    if len(c_dow) == 0 or len(t_dow) == 0:
        continue
    c_rate_dow = c_dow["signed_up"].mean()
    t_rate_dow = t_dow["signed_up"].mean()
    effect = t_rate_dow - c_rate_dow
    se = np.sqrt(t_rate_dow*(1-t_rate_dow)/len(t_dow) + c_rate_dow*(1-c_rate_dow)/len(c_dow))
    dow_stats.append({
        "Day": dow,
        "Control Rate": c_rate_dow,
        "Treatment Rate": t_rate_dow,
        "Treatment Effect": effect,
        "SE": se,
        "CI_low": effect - 1.96*se,
        "CI_high": effect + 1.96*se,
        "N_control": len(c_dow),
        "N_treatment": len(t_dow),
    })

dow_df = pd.DataFrame(dow_stats)

# ── Interaction regression & F-test ──
X_dow = pd.get_dummies(df["day_of_week"], drop_first=True).astype(float)
X_dow["treatment"] = df["treatment"].astype(float)
dow_cols = [c for c in X_dow.columns if c != "treatment"]
for dc in dow_cols:
    X_dow[f"treatment_x_{dc}"] = X_dow["treatment"] * X_dow[dc]
X_dow = sm.add_constant(X_dow)
y_dow = df["signed_up"].astype(float)
model_dow = sm.OLS(y_dow, X_dow).fit(cov_type="HC1")

interaction_terms = [c for c in model_dow.params.index if c.startswith("treatment_x_")]
constraint_matrix = np.zeros((len(interaction_terms), len(model_dow.params)))
for i, term in enumerate(interaction_terms):
    constraint_matrix[i, list(model_dow.params.index).index(term)] = 1
f_test = model_dow.f_test(constraint_matrix)
f_pvalue = float(f_test.pvalue)
f_stat = float(f_test.fvalue)

print("=" * 60)
print("Q4: DAY-OF-WEEK HETEROGENEITY")
print("=" * 60)
print(f"\nF-statistic: {f_stat:.4f}")
print(f"P-value:     {f_pvalue:.4f}")
print(f"Restrictions:{len(interaction_terms)}")
print()
if f_pvalue < 0.05:
    print("Result: SIGNIFICANT — treatment effect genuinely varies by day of week")
else:
    print("Result: NOT significant — day-to-day variation is likely noise")

print("\nDay-of-Week Details:")
display_dow = dow_df.copy()
display_dow["Control Rate"] = display_dow["Control Rate"].apply(lambda x: f"{x:.2%}")
display_dow["Treatment Rate"] = display_dow["Treatment Rate"].apply(lambda x: f"{x:.2%}")
display_dow["Treatment Effect"] = display_dow["Treatment Effect"].apply(lambda x: f"{x:+.2%}")
display_dow["95% CI"] = display_dow.apply(lambda r: f"[{r['CI_low']:.2%}, {r['CI_high']:.2%}]", axis=1)
print(display_dow[["Day", "N_control", "N_treatment", "Control Rate",
                    "Treatment Rate", "Treatment Effect", "95% CI"]].to_string(index=False))

# %%
fig_dow = go.Figure()
is_weekend = [d in ["Saturday", "Sunday"] for d in dow_df["Day"]]
fig_dow.add_trace(go.Bar(
    x=dow_df["Day"],
    y=dow_df["Treatment Effect"] * 100,
    error_y=dict(
        type="data", symmetric=False,
        array=(dow_df["CI_high"] - dow_df["Treatment Effect"]) * 100,
        arrayminus=(dow_df["Treatment Effect"] - dow_df["CI_low"]) * 100,
    ),
    marker_color=["#ff9800" if w else "#636EFA" for w in is_weekend],
    text=[f"{e:.2%}" for e in dow_df["Treatment Effect"]],
    textposition="outside",
))
fig_dow.add_hline(y=abs_lift * 100, line_dash="dash", line_color="gray",
                  annotation_text=f"Overall effect: {abs_lift:.2%}")
fig_dow.update_layout(
    title="Treatment Effect (Signup Rate Lift) by Day of Week",
    yaxis_title="Treatment Effect (pp)", xaxis_title="Day of Week",
    height=450, showlegend=False, margin=dict(t=60),
)
fig_dow.update_traces(cliponaxis=False)
show_fig(fig_dow)

# %% [markdown]
# ---
# # Q5: Have We Run This Experiment Long Enough?
#
# **Method:** Track the cumulative treatment effect and its 95% CI over time.
# If the estimate has stabilized and the CI is narrow enough, the experiment has converged.
#
# **Formulas:**
# - $\hat{\Delta}(d) = \hat{p}_T(d) - \hat{p}_C(d)$ (cumulative effect at date d)
# - $SE(d) = \sqrt{\frac{\hat{p}_T(d)(1-\hat{p}_T(d))}{n_T(d)} + \frac{\hat{p}_C(d)(1-\hat{p}_C(d))}{n_C(d)}}$
# - CI width shrinks as $\propto \frac{1}{\sqrt{n}}$

# %%
df_sorted = df.sort_values("assignment_date")
cumulative_data = []
unique_dates = sorted(df_sorted["assignment_date"].dt.date.unique())

for i, date in enumerate(unique_dates):
    if i % 2 != 0 and i != len(unique_dates) - 1:
        continue
    mask = df_sorted["assignment_date"].dt.date <= date
    subset = df_sorted[mask]
    c = subset[subset["treatment"] == 0]
    t = subset[subset["treatment"] == 1]
    if len(c) < 50 or len(t) < 50:
        continue
    c_r = c["signed_up"].mean()
    t_r = t["signed_up"].mean()
    eff = t_r - c_r
    se = np.sqrt(t_r*(1-t_r)/len(t) + c_r*(1-c_r)/len(c))
    cumulative_data.append({
        "Date": pd.Timestamp(date),
        "Effect": eff,
        "CI_low": eff - 1.96*se,
        "CI_high": eff + 1.96*se,
        "N": len(subset),
        "p_val": 2*(1 - stats.norm.cdf(abs(eff/se))) if se > 0 else 1,
    })

cum_df = pd.DataFrame(cumulative_data)

last_5 = cum_df.tail(5)
effect_range = last_5["Effect"].max() - last_5["Effect"].min()
effect_cv = last_5["Effect"].std() / abs(last_5["Effect"].mean()) if last_5["Effect"].mean() != 0 else float("inf")
early_effect = cum_df.iloc[len(cum_df)//4]["Effect"] if len(cum_df) > 4 else abs_lift
late_effect = cum_df.iloc[-1]["Effect"]
effect_stable = abs(late_effect - early_effect) < ci_diff

print("=" * 60)
print("Q5: EXPERIMENT DURATION ASSESSMENT")
print("=" * 60)
print(f"Effect stable:       {'Yes' if effect_stable else 'No — may still be evolving'}")
print(f"Final CI width:      ±{ci_diff:.2%}")
print(f"Effect range (last 5): {effect_range:.3%}")
print(f"Effect CV (last 5):    {effect_cv:.4f}")
print(f"Power at observed:     {power:.0%}")
print()
print("For retention: We likely need MORE TIME")
print("  - Treatment users get month 1 free, so cancellation timing differs")
print("  - Users assigned late have little observation time")
print("  - Cannot yet measure 3/6/12 month retention")

# %%
fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(
    x=cum_df["Date"], y=cum_df["CI_high"] * 100,
    mode="lines", line=dict(width=0), showlegend=False,
))
fig_cum.add_trace(go.Scatter(
    x=cum_df["Date"], y=cum_df["CI_low"] * 100,
    mode="lines", line=dict(width=0), fill="tonexty",
    fillcolor="rgba(99, 110, 250, 0.2)", showlegend=False,
))
fig_cum.add_trace(go.Scatter(
    x=cum_df["Date"], y=cum_df["Effect"] * 100,
    mode="lines+markers", name="Cumulative Treatment Effect",
    line=dict(color="#636EFA", width=3), marker=dict(size=4),
))
fig_cum.add_hline(y=0, line_dash="dash", line_color="gray")
fig_cum.update_layout(
    title="Cumulative Treatment Effect Over Time (with 95% CI)",
    xaxis_title="Date", yaxis_title="Treatment Effect (pp)", height=450,
)
show_fig(fig_cum)

# %%
fig_pval = go.Figure()
fig_pval.add_trace(go.Scatter(
    x=cum_df["Date"], y=cum_df["p_val"],
    mode="lines+markers", line=dict(color="#EF553B", width=2),
    marker=dict(size=4), name="p-value",
))
fig_pval.add_hline(y=0.05, line_dash="dash", line_color="green",
                   annotation_text="alpha = 0.05")
fig_pval.update_layout(
    title="Cumulative p-value Over Time",
    xaxis_title="Date", yaxis_title="p-value",
    yaxis=dict(type="log"), height=350,
)
show_fig(fig_pval)

# %% [markdown]
# ---
# # Q6: ROI Model & Break-Even Analysis
#
# **Formulas:**
# - Incremental signups: $\Delta_{\text{signups}} = (\hat{p}_T - \hat{p}_C) \times N$
# - Cannibalized users: $N_{\text{cannib}} = \hat{p}_C \times N$
# - Total cost: $C = \hat{p}_T \times N \times P$
# - Revenue at lifetime L: $R(L) = \Delta_{\text{signups}} \times P \times L$
# - Break-even: $L^* = \frac{C}{\Delta_{\text{signups}} \times P}$
# - ROI: $\text{ROI}(L) = \frac{R(L) - C}{C}$
# - Net profit: $\Pi(L) = R(L) - C$

# %%
MONTHLY_PRICE = 20.0
N_TARGET = 100_000

incremental_signup_rate = treat_rate - ctrl_rate
incremental_users = incremental_signup_rate * N_TARGET
total_treated_signups = treat_rate * N_TARGET
cannibalized_users = ctrl_rate * N_TARGET
cost_free_month = total_treated_signups * MONTHLY_PRICE
revenue_per_month_incremental = incremental_users * MONTHLY_PRICE

if incremental_users > 0:
    breakeven_months = cost_free_month / (incremental_users * MONTHLY_PRICE)
else:
    breakeven_months = float("inf")

print("=" * 60)
print("Q6: ROI MODEL & BREAK-EVEN ANALYSIS")
print("=" * 60)
print(f"Monthly price:          ${MONTHLY_PRICE:.0f}")
print(f"Target population:      {N_TARGET:,}")
print()
print("COST BREAKDOWN:")
print(f"  Total treatment signups:  {total_treated_signups:,.0f} ({treat_rate:.2%} × {N_TARGET:,})")
print(f"  Cost of free month:       ${cost_free_month:,.0f}")
print(f"  ├─ Cannibalized users:    {cannibalized_users:,.0f} → ${cannibalized_users * MONTHLY_PRICE:,.0f}")
print(f"  └─ Incremental users:     {incremental_users:,.0f} → ${incremental_users * MONTHLY_PRICE:,.0f}")
print()
print("BREAK-EVEN:")
print(f"  Revenue/month from incremental users: ${revenue_per_month_incremental:,.0f}")
print(f"  Break-even lifetime:                  {breakeven_months:.1f} paid months")
print()
print("ROI AT SAMPLE LIFETIMES:")
print(f"  {'Lifetime':>10} {'Revenue':>12} {'Net Profit':>12} {'ROI':>8}")
print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*8}")
for L in [3, 6, 12, 24, 36]:
    rev = incremental_users * MONTHLY_PRICE * L
    net = rev - cost_free_month
    roi = net / cost_free_month * 100
    print(f"  {L:>8}mo ${rev:>11,.0f} ${net:>11,.0f} {roi:>+7.0f}%")

# %%
lifetimes = np.arange(1, 37)
roi_values = [(incremental_users * MONTHLY_PRICE * lt - cost_free_month) / cost_free_month * 100 for lt in lifetimes]
net_profit_values = [incremental_users * MONTHLY_PRICE * lt - cost_free_month for lt in lifetimes]

fig_roi = make_subplots(rows=1, cols=2, subplot_titles=["ROI (%)", "Net Profit ($)"])
fig_roi.add_trace(go.Scatter(
    x=lifetimes, y=roi_values, mode="lines+markers",
    line=dict(color="#636EFA", width=3), name="ROI %", marker=dict(size=4),
), row=1, col=1)
fig_roi.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
fig_roi.add_vline(x=breakeven_months, line_dash="dot", line_color="orange", row=1, col=1,
                  annotation_text=f"Break-even: {breakeven_months:.1f}mo")

fig_roi.add_trace(go.Scatter(
    x=lifetimes, y=net_profit_values, mode="lines+markers",
    line=dict(color="#4CAF50", width=3), name="Net Profit", marker=dict(size=4),
), row=1, col=2)
fig_roi.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

fig_roi.update_layout(height=450, showlegend=False)
fig_roi.update_xaxes(title_text="Avg Paid Months per Incremental User", row=1, col=1)
fig_roi.update_xaxes(title_text="Avg Paid Months per Incremental User", row=1, col=2)
fig_roi.update_yaxes(title_text="ROI (%)", row=1, col=1)
fig_roi.update_yaxes(title_text="Net Profit ($)", row=1, col=2)
show_fig(fig_roi)

# %% [markdown]
# ---
# # Q7: Ship / No-Ship Recommendation
#
# **Decision Framework:**
# $$\text{Ship if: } \underbrace{p < 0.05 \text{ and } \Delta > 0}_{\text{Significant positive lift}} \;\wedge\; \underbrace{|\hat{c}_T - \hat{c}_C| < 0.05}_{\text{No retention degradation}} \;\wedge\; \underbrace{L^* < 12}_{\text{Reasonable break-even}}$$

# %%
sig_positive = p_val < 0.05 and abs_lift > 0
retention_ok = abs(treat_cancel_rate - ctrl_cancel_rate) < 0.05
reasonable_breakeven = breakeven_months < 12

if sig_positive and retention_ok and reasonable_breakeven:
    recommendation = "SHIP"
elif sig_positive:
    recommendation = "SHIP WITH CAUTION"
else:
    recommendation = "DO NOT SHIP YET"

print("=" * 60)
print("Q7: SHIP / NO-SHIP RECOMMENDATION")
print("=" * 60)
print(f"\n>>> RECOMMENDATION: {recommendation} <<<\n")
print("DECISION CRITERIA:")
print(f"  [{'✓' if sig_positive else '✗'}] Significant positive lift (p < 0.05 & Δ > 0)")
print(f"      p = {p_val:.4f}, Δ = {abs_lift:.4f}")
print(f"  [{'✓' if retention_ok else '✗'}] No retention degradation (|c_T - c_C| < 0.05)")
print(f"      |{treat_cancel_rate:.4f} - {ctrl_cancel_rate:.4f}| = {abs(treat_cancel_rate - ctrl_cancel_rate):.4f}")
print(f"  [{'✓' if reasonable_breakeven else '✗'}] Reasonable break-even (L* < 12 months)")
print(f"      L* = {breakeven_months:.1f} months")

print("\nEVIDENCE SUMMARY:")
evidence = [
    ("Signup lift", f"+{abs_lift:.2%} ({rel_lift:+.1%} relative)", "Positive" if abs_lift > 0 else "No lift"),
    ("Significance", f"p = {p_val:.4f}", "Significant" if p_val < 0.05 else "Not significant"),
    ("95% CI", f"[{abs_lift - ci_diff:.2%}, {abs_lift + ci_diff:.2%}]", "Excludes zero" if abs_lift - ci_diff > 0 else "Includes zero"),
    ("Cancel diff", f"{cancel_diff:+.1%}", "Similar" if abs(cancel_diff) < 0.05 else "Different"),
    ("Break-even", f"{breakeven_months:.1f} months", "Reasonable" if breakeven_months < 12 else "Long"),
]
for metric, val, assess in evidence:
    print(f"  {metric:<20} {val:<30} {assess}")

print("\nRECOMMENDED MODIFICATIONS IF SHIPPING:")
print("  • Target high-value segments to reduce cannibalization costs")
print("  • Improve onboarding during free month to increase stickiness")
print("  • Test shorter free periods (1-2 weeks) for similar conversion at lower cost")
print("  • Monitor long-term retention to validate ROI assumptions")
print("  • A/B test messaging to optimize conversion further")

# %% [markdown]
# ---
# # Q8: Follow-Up Targeting Strategy
#
# **Method:** Calculate treatment effects within each segment (platform, time zone),
# compute segment-level break-even, and identify the best segments to target.
#
# **Formulas:**
# - $\hat{\Delta}_s = \hat{p}_{T,s} - \hat{p}_{C,s}$ (segment treatment effect)
# - $SE_s = \sqrt{\frac{\hat{p}_{T,s}(1-\hat{p}_{T,s})}{n_{T,s}} + \frac{\hat{p}_{C,s}(1-\hat{p}_{C,s})}{n_{C,s}}}$
# - $L^*_s = \frac{\hat{p}_{T,s}}{\hat{\Delta}_s}$ (segment break-even)
# - Target if: $\hat{\Delta}_s > 0$ and $L^*_s < L_{\text{expected}}$

# %%
segment_configs = [
    ("assigned_on_platform", "Platform"),
    ("time_zone", "Time Zone"),
]

segment_results = []
for seg_col, seg_label in segment_configs:
    for val in sorted(df[seg_col].unique()):
        c_seg = control[control[seg_col] == val]
        t_seg = treatment[treatment[seg_col] == val]
        if len(c_seg) < 10 or len(t_seg) < 10:
            continue
        c_r = c_seg["signed_up"].mean()
        t_r = t_seg["signed_up"].mean()
        eff = t_r - c_r
        se = np.sqrt(t_r*(1-t_r)/len(t_seg) + c_r*(1-c_r)/len(c_seg))

        c_cancel = c_seg[c_seg["signed_up"]]["paid_plan_canceled"].mean() if c_seg["signed_up"].sum() > 0 else np.nan
        t_cancel = t_seg[t_seg["signed_up"]]["paid_plan_canceled"].mean() if t_seg["signed_up"].sum() > 0 else np.nan

        inc_users_seg = eff * len(t_seg)
        cost_seg = t_r * len(t_seg) * MONTHLY_PRICE
        rev_per_month_seg = inc_users_seg * MONTHLY_PRICE
        be_months_seg = cost_seg / rev_per_month_seg if rev_per_month_seg > 0 else float("inf")

        segment_results.append({
            "Segment": f"{seg_label}: {val}",
            "N_control": len(c_seg),
            "N_treatment": len(t_seg),
            "Control Signup": c_r,
            "Treatment Signup": t_r,
            "Lift": eff,
            "SE": se,
            "p-value": 2 * (1 - stats.norm.cdf(abs(eff / se))) if se > 0 else 1.0,
            "Control Cancel": c_cancel,
            "Treatment Cancel": t_cancel,
            "Break-Even Months": be_months_seg,
        })

seg_df = pd.DataFrame(segment_results)

print("=" * 60)
print("Q8: FOLLOW-UP TARGETING STRATEGY")
print("=" * 60)
print("\nSegment Analysis:")
display_seg = seg_df.copy()
display_seg["Control Signup"] = display_seg["Control Signup"].apply(lambda x: f"{x:.2%}")
display_seg["Treatment Signup"] = display_seg["Treatment Signup"].apply(lambda x: f"{x:.2%}")
display_seg["Lift"] = display_seg["Lift"].apply(lambda x: f"{x:+.2%}")
display_seg["p-value"] = display_seg["p-value"].apply(lambda x: f"{x:.4f}")
display_seg["Control Cancel"] = display_seg["Control Cancel"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
display_seg["Treatment Cancel"] = display_seg["Treatment Cancel"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
display_seg["Break-Even Months"] = display_seg["Break-Even Months"].apply(lambda x: f"{x:.1f}" if x < 100 else "N/A")
print(display_seg[["Segment", "N_control", "N_treatment", "Control Signup",
                    "Treatment Signup", "Lift", "p-value", "Break-Even Months"]].to_string(index=False))

positive_lift = seg_df[seg_df["Lift"] > 0].sort_values("Break-Even Months")
best_segments = positive_lift.head(3)
best_names = ", ".join(best_segments["Segment"].tolist()) if len(best_segments) > 0 else "N/A"
print(f"\nBest segments (highest lift + best ROI): {best_names}")

# %%
fig_seg = make_subplots(rows=1, cols=2,
                        subplot_titles=["Treatment Effect (Lift) by Segment", "Break-Even Months"])

colors = px.colors.qualitative.Set2
fig_seg.add_trace(go.Bar(
    x=seg_df["Segment"], y=seg_df["Lift"] * 100,
    error_y=dict(type="data", array=seg_df["SE"] * 1.96 * 100),
    marker_color=colors[:len(seg_df)], name="Lift (pp)",
), row=1, col=1)
fig_seg.add_hline(y=abs_lift * 100, line_dash="dash", line_color="gray", row=1, col=1,
                  annotation_text=f"Overall: {abs_lift:.2%}")

be_display = seg_df["Break-Even Months"].clip(upper=50)
fig_seg.add_trace(go.Bar(
    x=seg_df["Segment"], y=be_display,
    marker_color=[colors[i % len(colors)] for i in range(len(seg_df))],
    name="Break-Even", showlegend=False,
), row=1, col=2)
fig_seg.add_hline(y=breakeven_months, line_dash="dash", line_color="gray", row=1, col=2,
                  annotation_text=f"Overall: {breakeven_months:.1f}mo")

fig_seg.update_layout(height=500, showlegend=False)
fig_seg.update_yaxes(title_text="Lift (pp)", row=1, col=1)
fig_seg.update_yaxes(title_text="Months to Break-Even", row=1, col=2)
fig_seg.update_xaxes(tickangle=-30)
show_fig(fig_seg)

# %% [markdown]
# ---
# # Summary
#
# | Question | Key Finding |
# |---|---|
# | Q1: Signup Impact | Treatment increases signups — statistically significant |
# | Q2: Retention | Cancellation rates similar between groups |
# | Q3: Reliability | Well-powered test with balanced randomization |
# | Q4: Day-of-Week | No significant heterogeneity (F-test) |
# | Q5: Duration | Signup effect has converged; retention needs more time |
# | Q6: ROI | Reasonable break-even achievable |
# | Q7: Recommendation | Ship (with monitoring) |
# | Q8: Targeting | Focus on highest-lift, lowest break-even segments |

# %%
print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nFinal recommendation: {recommendation}")
print(f"Signup lift: +{abs_lift:.2%} (p = {p_val:.4f})")
print(f"Break-even: {breakeven_months:.1f} months")
print(f"Best target segments: {best_names}")
