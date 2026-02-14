import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest

# 1. Load Data
df = pd.read_csv('free_plus_experiment.csv')
df['signed_up'] = pd.to_datetime(df['paid_signup_date']).notnull().astype(int)

# 2. Statistical Significance Test
successes = df.groupby('treatment')['signed_up'].sum().values[::-1]
totals = df.groupby('treatment')['signed_up'].count().values[::-1]
stat, pval = proportions_ztest(successes, totals)

print(f"Treatment Signup Rate: {df[df['treatment']==1]['signed_up'].mean():.2%}")
print(f"Control Signup Rate: {df[df['treatment']==0]['signed_up'].mean():.2%}")
print(f"P-Value: {pval:.4f}")

# 3. ROI Break-Even Calculation
# Formula: P_control * 20 * L = P_treatment * 20 * (L - 1)
p_c, p_t = 0.0796, 0.0966
l_break_even = p_t / (p_t - p_c)

# 4. Visualization: ROI Model
lifetimes = np.linspace(1, 10, 100)
rev_c = p_c * 20 * lifetimes
rev_t = p_t * 20 * (lifetimes - 1)

plt.figure(figsize=(10, 6))
plt.plot(lifetimes, rev_c, label='Control (Paid M1)')
plt.plot(lifetimes, rev_t, label='Treatment (Free M1)', color='orange')
plt.axvline(l_break_even, color='red', linestyle='--', label=f'Break-even ({l_break_even:.1f} mo)')
plt.title('Expected Revenue per User vs. Lifetime')
plt.xlabel('Paid Months (L)')
plt.ylabel('Expected Revenue ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
