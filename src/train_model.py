import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_model():
    """
    Train a Linear Regression model on the processed student dataset.

    Returns:
        model: Trained Linear Regression model.
    """
    # ---------------------------------------------------------
    # Step 2: Load Processed Data
    # ---------------------------------------------------------
    
    # Adjust the path if your CSV name is different
    data_path = "../data/processed/processed_dataset.csv" 
    print("Current Working Directory:", os.getcwd())
    print("Trying to load:", data_path)
    df = pd.read_csv(data_path)
    print(f"Loading dataset from: {data_path}")

    # ---------------------------------------------------------
    # Setup for Step 3: Split Features and Target
    # ---------------------------------------------------------
    # Assuming 'Exam_Score' is your target variable
    X = df.drop("Exam_Score", axis=1)
    y = df["Exam_Score"]

    # Train-Test Split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---------------------------------------------------------
    # Step 3: Train the Model
    # ---------------------------------------------------------
    print("Training Linear Regression model...")
    model = LinearRegression()
    
    # .fit() is where the model learns the relationship (the weights/coefficients) 
    # between your features (X_train) and the target scores (y_train).
    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # Step 4: Make Predictions
    # ---------------------------------------------------------
    print("Making predictions on the test set...")
    y_pred = model.predict(X_test)

    # ---------------------------------------------------------
    # Step 5: Compare Predictions (Updated with Task 2)
    # ---------------------------------------------------------
    comparison_df = pd.DataFrame({
        'Actual Score': y_test.values, # .values resets the index for a clean view
        'Predicted Score': y_pred
    })
    print("\n--- Prediction Comparison (First 5 rows) ---")
    print(comparison_df.head())

    # --- NEW CODE FOR TASK 2 ---
    # Make sure the results folder exists (creates it if you forgot Step 1!)
    os.makedirs("../results", exist_ok=True) 
    
    # Save the dataframe to a CSV file without the index numbers
    predictions_path = "../results/predictions.csv"
    comparison_df.to_csv(predictions_path, index=False)
    print(f"✅ Predictions successfully saved to: {predictions_path}")
    # ---------------------------

    # ---------------------------------------------------------
    # Step 6: Evaluate the Model
    # ---------------------------------------------------------
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation Metrics ---")
    print(f"MAE (Mean Absolute Error): {mae:.2f}")
    print(f"MSE (Mean Squared Error):  {mse:.2f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")

    # ---------------------------------------------------------
    # Step 7: Feature Importance
    # ---------------------------------------------------------
    # Inspecting coefficients to see which feature has the biggest impact
    importance_df = pd.DataFrame({
        'Feature': X.columns, 
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)
    
    print("\n--- Feature Importance (Coefficients) ---")
    print(importance_df)

    # ---------------------------------------------------------
    # Step 8: Save the Model
    # ---------------------------------------------------------
    model_save_path = "../models/student_score_model.pkl"
    joblib.dump(model, model_save_path)
    print(f"\nModel successfully saved to: {model_save_path}")
    return model



if __name__ == "__main__":
    train_model()