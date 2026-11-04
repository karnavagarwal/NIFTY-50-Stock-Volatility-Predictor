import pandas as pd
import numpy as np
from naive_persistence import NaivePersistence

#There is no training for naive persistence as it simply answers the volatility of the past 5 days
print("Loading Raw test data...")
test_df = pd.read_csv("test_raw.csv", parse_dates=['Date'])

print("Generating predictions...")
model = NaivePersistence()
model.fit(test_df)
predictions = model.predict(test_df)

output_df = test_df[['Date', 'Ticker', 'Target_Vol_5d']].copy()
output_df.rename(columns = {'Target_Vol_5d': 'Actual_Volatility'}, inplace = True)
output_df['Predicted_Volatility'] = predictions

output_df.to_csv("naive_persistence_output.csv", index = False)
print("Predictions successfully generated and saved to naive_persistence_output.csv")