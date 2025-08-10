
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load and train model
df = pd.read_csv('employee_data.csv')
X = df[['experience', 'education_level', 'evaluation_score']]
y = df['performance']

model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
