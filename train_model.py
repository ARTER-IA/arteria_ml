import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# Carga del dataset
data = pd.read_excel("datasets/dataset.xlsx")

# Corregimos el valor "Fmale" por "Female" de la variable Sex
data["Sex"] = data["Sex"].replace({"Fmale": "Female"})

# Seleccionamos los parámetros de entrada
selected_variables = [
    "Age",
    "Weight",
    "Length",
    "Sex",
    "BMI",
    "DM",
    "HTN",
    "Current Smoker",
    "EX-Smoker",
    "FH",
    "Obesity",
    "CVA",
    "Thyroid Disease",
    "BP",
    "PR",
    "Weak Peripheral Pulse",
    "Q Wave",
    "St Elevation",
    "St Depression",
    "Tinversion",
    "LVH",
    "Poor R Progression",
    "TG",
    "LDL",
    "HDL",
    "HB",
    "Cath",
]

data = data[selected_variables]

# Renombramos las columnas
data = data.rename(
    columns={
        "Age": "age",
        "Weight": "weight",
        "Length": "length",
        "Sex": "sex",
        "BMI": "bmi",
        "DM": "dm",
        "HTN": "htn",
        "Current Smoker": "current_Smoker",
        "EX-Smoker": "ex_Smoker",
        "FH": "fh",
        "Obesity": "obesity",
        "CVA": "cva",
        "Thyroid Disease": "thyroid_Disease",
        "BP": "bp",
        "PR": "pr",
        "Weak Peripheral Pulse": "weak_Peripheral_Pulse",
        "Q Wave": "q_Wave",
        "St Elevation": "st_Elevation",
        "St Depression": "st_Depression",
        "Tinversion": "tinversion",
        "LVH": "lvh",
        "Poor R Progression": "poor_R_Progression",
        "TG": "tg",
        "LDL": "ldl",
        "HDL": "hdl",
        "HB": "hb",
    }
)

# Conversión de valores categóricos a valores binarios
binary_cols = [
    "obesity",
    "cva",
    "thyroid_Disease",
    "weak_Peripheral_Pulse",
    "lvh",
    "poor_R_Progression",
]

for col in binary_cols:
    data[col] = data[col].map({"Y": 1, "N": 0})

# Separar variables dependientes (X) y variable independiente (y)
X = data.drop("Cath", axis=1)  # 'Cath' es la variable objetivo
y = data["Cath"].apply(lambda x: 1 if x == "CAD" else 0)

# Identificar columnas numéricas y categóricas
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns

# Se crea un ColumnTransformer para preprocesar las columnas numéricas y categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
)

# Se crea un pipeline que incluye preprocesamiento, balanceo de datos con SMOTE y el modelo de regresión logística.
pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", LogisticRegression(random_state=42)),
    ]
)

# División de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
pipeline.fit(X_train, y_train)

# Guardar el modelo entrenado
joblib.dump(pipeline, "model.pkl")
