import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
import numpy as np
import joblib

spark = SparkSession.builder.config("spark.driver.host", "localhost").config("spark.python.worker.reuse", "false").getOrCreate()
file_path = 'file:///C:/Users/seanz/VSCode_WS/BigData/NFL_PBP_V1.csv'
df = spark.read.csv(file_path, header=True, inferSchema=True)

df = df.withColumn(
    'play_success', 
    F.when(
        (F.col('play_type') == 'run') & (F.col('rush_attempt') == 1) & (F.col('yards_gained') >= 4), 1
    ).when(
        (F.col('play_type') == 'pass') & (F.col('pass_attempt') == 1) & (F.col('yards_gained') >= 4), 1
    ).otherwise(0)
)

window_spec = Window.partitionBy('game_id', 'posteam').orderBy('play_id')
df = df.withColumn(
    'cumulative_rush_attempts', 
    F.sum(F.when((F.col('play_type') == 'run') & (F.col('rush_attempt') == 1), 1).otherwise(0)).over(window_spec)
)

df = df.withColumn(
    'cumulative_pass_attempts', 
    F.sum(F.when((F.col('play_type') == 'pass') & (F.col('pass_attempt') == 1), 1).otherwise(0)).over(window_spec)
)

df = df.withColumn(
    'cumulative_rush_successes', 
    F.sum(F.when(
        (F.col('play_type') == 'run') & (F.col('rush_attempt') == 1) & (F.col('yards_gained') >= 4), 1).otherwise(0)
    ).over(window_spec)
)

df = df.withColumn(
    'cumulative_pass_successes', 
    F.sum(F.when(
        (F.col('play_type') == 'pass') & (F.col('pass_attempt') == 1) & (F.col('yards_gained') >= 4), 1).otherwise(0)
    ).over(window_spec)
)

df = df.withColumn(
    'rush_success_rate', 
    F.when(F.col('cumulative_rush_attempts') > 0, F.col('cumulative_rush_successes') / F.col('cumulative_rush_attempts')).otherwise(0)
)

df = df.withColumn(
    'pass_success_rate', 
    F.when(F.col('cumulative_pass_attempts') > 0, F.col('cumulative_pass_successes') / F.col('cumulative_pass_attempts')).otherwise(0)
)

df = df.withColumn('posteam_leading', F.col('score_differential_post') > 0)
df = df.withColumn('posteam_trailing', F.col('score_differential_post') < 0)
df = df.withColumn('yards_gained', F.col('yards_gained').cast('float'))
df = df.withColumn('shotgun', F.col('shotgun').cast('float'))
df = df.withColumn('no_huddle', F.col('no_huddle').cast('float'))
df = df.withColumn('timeout', F.col('timeout').cast('float'))
df = df.withColumn('posteam_timeouts_remaining', F.col('posteam_timeouts_remaining').cast('float'))
df = df.withColumn('defteam_timeouts_remaining', F.col('defteam_timeouts_remaining').cast('float'))
offensive_playtypes = ['field_goal', 'run', 'punt', 'pass']
df = df.filter(df.play_type.isin(offensive_playtypes))

df_indexed = df.withColumn('play_type_index', 
                           F.when(F.col('play_type') == 'pass', 0)
                            .when(F.col('play_type') == 'run', 1)
                            .when(F.col('play_type') == 'punt', 2)
                            .when(F.col('play_type') == 'field_goal', 3)
                            .otherwise(-1))  

feature_columns = [
    # Field Position & Time
    'yardline_100', 'game_seconds_remaining', 'qtr', 'down', 'goal_to_go', 'ydstogo', 
    'score_differential_post', 'posteam_score_post',

    # Game Context & Situational Awareness
    'quarter_seconds_remaining', 'half_seconds_remaining', 'drive', 'score_differential',

    # Play History (Momentum)
    'play_success', 'cumulative_rush_attempts', 'cumulative_pass_attempts', 
    'cumulative_rush_successes', 'cumulative_pass_successes', 'rush_success_rate', 
    'pass_success_rate', 'posteam_leading', 'posteam_trailing',

    # Critical 3rd/4th Down Context
    'third_down_converted', 'third_down_failed', 'fourth_down_converted', 'fourth_down_failed',

    # Play Consequences
    'incomplete_pass', 'sack', 'touchback', 'interception',

    # Special Teams Indicators
    'punt_blocked', 'punt_inside_twenty', 'punt_in_endzone', 'punt_out_of_bounds'
]


assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
df_vector = assembler.transform(df_indexed)

games = df_vector.select("game_id").distinct().collect()  
train_games = games[:int(0.8 * len(games))] 
test_games = games[int(0.8 * len(games)):]
test_df = df_vector.filter(col("game_id").isin([game["game_id"] for game in test_games]))
test_pandas = test_df.toPandas()
X_test = np.array(test_pandas["features"].tolist())
y_test = np.array(test_pandas["play_type_index"].tolist())

# === 1. LOAD SAVED XGBOOST MODEL === #
model_path = "xgboost_football_model.pkl"

try:
    xgb = joblib.load(model_path)
    print("✅ Successfully loaded XGBoost model!")
except FileNotFoundError:
    st.error("❌ Model file not found! Please train and save the model first.")
    st.stop()  # Stop execution if the model is missing
    
loss_path = "xgboost_loss_results.pkl"

try:
    results = joblib.load(loss_path)
    train_loss = results["train_loss"]  # ✅ FIXED
    test_loss = results["test_loss"]
except FileNotFoundError:
    st.warning("⚠️ Training loss data not found. Retrain the model with `eval_set` to enable loss visualization.")
    train_loss, test_loss = None, None


# === 2. LOAD TEST DATA === #
games = df_vector.select("game_id").distinct().collect()  
train_games = games[:int(0.8 * len(games))] 
test_games = games[int(0.8 * len(games)):]
test_df = df_vector.filter(col("game_id").isin([game["game_id"] for game in test_games]))

test_pandas = test_df.toPandas()
X_test = np.array(test_pandas["features"].tolist())
y_test = np.array(test_pandas["play_type_index"].tolist())

y_pred = xgb.predict(X_test)

# === 4. DASHBOARD VISUALIZATIONS === #
st.title("XGBoost Football Play Prediction Dashboard")

# Define class labels
class_labels = ["Pass", "Run", "Punt", "Field Goal"]

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# === 5. Display Model Performance === #
st.subheader("Model Performance Metrics")
accuracy = (y_test == y_pred).mean()
precision = classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["precision"]
recall = classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["recall"]
f1 = classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["f1-score"]

st.write(f"**Accuracy:** {accuracy:.4f}")
st.write(f"**Precision:** {precision:.4f}")
st.write(f"**Recall:** {recall:.4f}")
st.write(f"**F1 Score:** {f1:.4f}")

if train_loss and test_loss:
    st.subheader("Training vs. Test Set Loss")
    
    st.write(f"🔹 **Training Set Log Loss:** {train_loss:.4f}")
    st.write(f"🔹 **Testing Set Log Loss:** {test_loss:.4f}")


# === 6. INDIVIDUAL CLASS METRICS === #
st.subheader("Individual Class Metrics")
class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
class_metrics_df = pd.DataFrame(class_report).transpose()

class_metrics_df = class_metrics_df.rename(index={  
    "0": "Pass",  
    "1": "Run",  
    "2": "Punt",  
    "3": "Field Goal"  
})

# Show precision, recall, and F1-score for each class
st.dataframe(class_metrics_df.loc[class_labels, ["precision", "recall", "f1-score"]])

# === 8. Play Type Distribution === #
st.subheader("Play Type Distribution")
play_counts = pd.Series(y_pred).value_counts().sort_index()
fig, ax = plt.subplots()
ax.bar(class_labels, play_counts, color=["blue", "green", "red", "purple"])
ax.set_ylabel("Count")
st.pyplot(fig)

# === 7. Confusion Matrix Heatmap === #
st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
st.pyplot(fig)

st.subheader("🔑 Play Type Key")
st.markdown("""
**Play Type Mapping:**
- **Pass** 🏈 → 0  
- **Run** 🏃‍♂️ → 1  
- **Punt** 🦵 → 2  
- **Field Goal** 🏹 → 3  
""")


# === 9. Misclassified Plays with Prediction Probabilities === #
st.subheader("Misclassified Plays with Prediction Probabilities")
y_pred_proba = xgb.predict_proba(X_test)
proba_df = pd.DataFrame(y_pred_proba, columns=["predicted_0", "predicted_1", "predicted_2", "predicted_3"])

# Attach probabilities to test data
test_pandas = test_pandas.reset_index(drop=True)
test_pandas = pd.concat([test_pandas, proba_df], axis=1)

# Filter misclassified plays
test_pandas["predicted_play"] = y_pred
misclassified = test_pandas[test_pandas["predicted_play"] != test_pandas["play_type_index"]]

st.dataframe(misclassified[["down", "ydstogo", "yardline_100", "play_type_index", "predicted_play", "predicted_0", "predicted_1", "predicted_2", "predicted_3"]])


# === 11. KDE Density Plot for Play Type Distribution === #
st.subheader("Play Type Density Map")
test_pandas["yardline_100"] = pd.to_numeric(test_pandas["yardline_100"], errors="coerce")
test_pandas = test_pandas.dropna(subset=["yardline_100"])
test_pandas = test_pandas[(test_pandas["yardline_100"] >= 0) & (test_pandas["yardline_100"] <= 99)]

fig, ax = plt.subplots(figsize=(10, 6))
play_colors = {0: "blue", 1: "green", 2: "red", 3: "purple"}

for play_type in [0, 1, 2, 3]:
    sns.kdeplot(
        test_pandas[test_pandas["predicted_play"] == play_type]["yardline_100"],
        label=f"{class_labels[play_type]}",
        fill=True, color=play_colors[play_type], alpha=0.4,
        bw_adjust=0.5
    )

plt.xlabel("Yardline (Closer to 0 = Near End Zone)")
plt.ylabel("Density")
plt.title("Play Type Density Distribution by Field Position")
plt.legend(["Pass", "Run", "Punt", "Field Goal"])
plt.xlim(0, 99)

st.pyplot(fig)