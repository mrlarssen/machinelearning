import pandas as pd
import numpy as np

df = pd.read_csv('/Users/user/Development/Python/deeplearning/churn.csv')

col_names = df.columns.tolist()

#print(col_names)

# Isolate target data
churn_result = df['Churn?']
y = np.where(churn_result == 'True', 1, 0)

#print(y)

# We don't need these columns
to_drop = ['State', 'Area Code', 'Phone', 'Churn?']
churn_feature_space = df.drop(to_drop, axis=1)

yes_no_cols = ["Int'l Plan", "VMail Plan"]
churn_feature_space[yes_no_cols] = churn_feature_space[yes_no_cols] == 'yes'

#Pull out features for future use
features = churn_feature_space.columns

X = churn_feature_space.as_matrix().astype(np.float)
print(X)
