import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.formula.api as smf

# 1. SETUP & CLEANING
df = pd.read_csv('free_plus_experiment.csv')
df['paid_signup_date'] = pd.to_datetime(df['paid_signup_date'])
df['signed_up'] = df['paid_signup_date'].notnull().astype(int)
df['day_of_week'] = pd.to_datetime(df['assignment_date']).dt.day_name()

# 2. SIGNUP IMPACT
stats = df.groupby('treatment')['signed_up'].agg(['count', 'sum', 'mean'])
stats.columns = ['Total Users', 'Signups', 'Conv Rate']
print("Signup Statistics:\n", stats)

# Z-Test for Significance
z_stat, p_val = proportions_ztest(stats['Signups'][::-1], stats['Total Users'][::-1])
print(f"\nSignup P-Value: {p_val:.6f}")

# 3. MATURE RETENTION (Users with >30 days since signup)
data_end = df['paid_signup_date'].max()
df['days_since_signup'] = (data_end - df['paid_signup_date']).dt.days
mature = df[(df['signed_up'] == 1) & (df['days_since_signup'] >= 30)]
retention = mature.groupby('treatment')['paid_plan_canceled'].mean()
print("\nCancellation Rate (Mature Cohort):\n", 1 - retention)

# 4. DAY OF WEEK INTERACTION
# This checks if the 'Treatment Effect' varies by Day of Week
model = smf.logit('signed_up ~ treatment * day_of_week', data=df).fit()
print("\nSignificant Interactions:\n", model.pvalues[model.pvalues < 0.05])

# 5. ROI VISUALIZATION
lifetimes = np.linspace(1, 12, 12)
rev_control = 0.0796 * 20 * lifetimes
rev_treat = 0.0966 * 20 * (lifetimes - 1)

plt.figure(figsize=(10, 5))
plt.plot(lifetimes, rev_control, label='Control (Paid M1)', marker='o')
plt.plot(lifetimes, rev_treat, label='Treatment (Free M1)', marker='s')
plt.axvline(5.68, color='red', linestyle='--', label='Break-even (5.7 mo)')
plt.title("Projected Revenue per User vs. Customer Lifetime")
plt.xlabel("Months of Subscription")
plt.ylabel("Expected Revenue ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
