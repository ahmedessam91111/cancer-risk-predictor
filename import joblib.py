import joblib
from sklearn.preprocessing import LabelEncoder

model = joblib.load(r"model_xgb_new.pkl")

# Feature names: prefer whatever sklearn stored from training; else the notebook's list
feats = getattr(model, "feature_names_in_", None)
if feats is None:
    feats = ["Age", "Gender", "Smoking", "Alcohol_Use", "Obesity", "Family_History",
             "Diet_Red_Meat", "Diet_Salted_Processed", "Fruit_Veg_Intake",
             "Physical_Activity", "Air_Pollution", "Occupational_Hazards",
             "BRCA_Mutation", "H_Pylori_Infection", "Calcium_Intake",
             "Overall_Risk_Score", "BMI", "Physical_Activity_Level"]
feats = list(feats)

# Training encoded Low/Medium/High alphabetically -> High=0, Low=1, Medium=2
le = LabelEncoder().fit(["Low", "Medium", "High"])

joblib.dump(le, "label_encoder.pkl")
joblib.dump(feats, "feature_names.pkl")

print("label encoder classes:", le.classes_)
print("saved features:", len(feats))