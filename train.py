import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("Social_media_impact_on_life.csv")


df.drop("Student_ID", axis=1, inplace=True)


categorical_cols = [
    "Gender",
    "Academic_Level",
    "Country",
    "Most_Used_Platform",
    "Affects_Academic_Performance",
    "Overall_Impact"
]


encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # 🔥 FIX HERE
    encoders[col] = le


X = df.drop("Overall_Impact", axis=1)
y = df["Overall_Impact"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)


joblib.dump(model, open("model.pkl", "wb"))
joblib.dump(encoders, open("encoders.pkl", "wb"))

print(" Model & encoders saved")