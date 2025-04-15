import pandas as pd

df = pd.read_csv('/Users/martina/Desktop/Causal Fair Gen/data/no_a_y/data/confounder_0_1.csv')

import statsmodels.api as sm

# Assuming df is your DataFrame containing the variables A, Y, and C
model = sm.OLS(df['Y'], sm.add_constant(df[['A', 'C']])).fit()

# Extract coefficients
coef_A = model.params['A']

# Calculate adjusted total effect
TE_adj = coef_A
print("Adjusted Total Effect:", TE_adj)
