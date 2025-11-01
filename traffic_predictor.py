import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor             
from sklearn.ensemble import GradientBoostingRegressor     
from sklearn.neural_network import MLPRegressor            
import joblib 

# ==========================================================
# 🥇 ALGORITHM 1: SIMPLE LINEAR REGRESSION (FROM SCRATCH)
# ==========================================================

class SimpleLinearRegressionScratch:
    """Linear Regression using Ordinary Least Squares (Matrix Form)."""
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        try:
            X_T_X = X_b.T @ X_b
            X_T_X_inv = np.linalg.inv(X_T_X)
            X_T_y = X_b.T @ y
            self.weights = X_T_X_inv @ X_T_y
        except np.linalg.LinAlgError:
            self.weights = np.linalg.lstsq(X_b, y, rcond=None)[0]

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.weights


# ==========================================================
# 🥈 ALGORITHM 2: K-NEAREST NEIGHBORS (KNN) (FROM SCRATCH)
# ==========================================================

class KNNRegressorScratch:
    """K-Nearest Neighbors Regressor using Euclidean Distance."""
    def __init__(self, n_neighbors=5):
        self.k = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _predict_single(self, x_test):
        distances = np.sqrt(np.sum((self.X_train - x_test)**2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_targets = self.y_train[k_indices]
        return np.mean(k_nearest_targets)

    def predict(self, X_test):
        predictions = np.array([self._predict_single(x) for x in X_test])
        return predictions


# ==========================================================
# 🚀 DATA PREPROCESSING AND MAIN EXECUTION
# ==========================================================

def load_and_preprocess_data(file_path):
    """Loads, cleans, and prepares the data for model training."""
    df = pd.read_csv(file_path)
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    # Feature Engineering and cleanup
    df['hour'] = df['date_time'].dt.hour
    df['dayofweek'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    df = df.drop(['date_time', 'year'], axis=1, errors='ignore')
    
    # Encode categorical features
    for column in ['holiday', 'weather_main', 'weather_description']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])

    # Convert temperature
    df['temp'] = df['temp'] - 273.15 
    
    # Scale features for better performance
    X = df.drop('traffic_volume', axis=1)
    y = df['traffic_volume'].values # Convert target to numpy array
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return pd.DataFrame(X_scaled, columns=X.columns), y # Return X as DataFrame for splits and y as numpy array

def train_and_select_best_model(X_train, X_test, y_train, y_test):
    """Initializes, trains, and evaluates the five regression models."""
    
    models = {
        '1. Linear Reg (Scratch)': SimpleLinearRegressionScratch(),
        '2. KNN Reg (Scratch)': KNNRegressorScratch(n_neighbors=5),
        '3. Decision Tree (Sklearn)': DecisionTreeRegressor(random_state=42, max_depth=10),
        '4. Gradient Boosting (Sklearn)': GradientBoostingRegressor(n_estimators=100, random_state=42),
        '5. Simple Neural Network (Sklearn)': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=200, random_state=42)
    }
    
    best_r2 = -float('inf')
    best_model_name = ""
    best_model = None
    results = []
    
    print("\n--- 5 Algorithm Comparison (Traffic Volume Prediction) ---")

    # Train and evaluate each model
    for name, model in models.items():
        try:
            # Use .values for scratch models, default for sklearn
            model.fit(X_train.values, y_train) 
            predictions = model.predict(X_test.values) 
            
            # Calculate Evaluation Metrics
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            
            print(f"| {name:<30} | R2: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f} |")
            
            results.append({
                'Model': name,
                'R2 Score': r2,
                'MAE': mae,
                'RMSE': rmse
            })
            
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
                best_model = model
                
        except Exception as e:
            print(f"| {name:<30} | FAILED - Error: {e} |")

    print("\n----------------------------------------------------------------")
    print(f"🥇 BEST MODEL SELECTED: {best_model_name} (R2 Score: {best_r2:.4f})")
    print("----------------------------------------------------------------")
    
    joblib.dump(best_model, 'best_traffic_model.pkl')
    print("✅ Best model saved as 'best_traffic_model.pkl'")
    
    return best_model, pd.DataFrame(results), X_test.iloc[0].values

def generate_comparison_charts(results_df):
    """Generates comparison bar charts for R2, MAE, and RMSE."""
    
    # Sort by R2 for better visualization order
    results_df = results_df.sort_values(by='R2 Score', ascending=False)
    
    metrics = ['R2 Score', 'MAE', 'RMSE']
    titles = ['R-squared Score (Higher is Better)', 
              'Mean Absolute Error (Lower is Better)', 
              'Root Mean Squared Error (Lower is Better)']
    colors = ['#4CAF50', '#2196F3', '#FF9800'] # Green, Blue, Orange

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle('Model Performance Comparison for Traffic Volume Prediction', fontsize=16, y=1.02)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Determine colors based on performance
        is_best = results_df[metric] == (results_df[metric].max() if metric == 'R2 Score' else results_df[metric].min())
        bar_colors = np.where(is_best, 'red', colors[i])
        
        ax.bar(results_df['Model'], results_df[metric], color=bar_colors)
        ax.set_title(titles[i])
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', linestyle='--')
        
        # Add labels on bars
        for bar in ax.patches:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.05 * bar.get_height() if bar.get_height() > 0 else 0.1),
                f'{bar.get_height():.3f}',
                ha='center',
                fontsize=9
            )

    plt.tight_layout()
    plt.savefig('model_comparison_charts.png')
    print("✅ Model comparison charts saved as 'model_comparison_charts.png'")
    

# --- Main Execution Block ---
if __name__ == "__main__":
    FILE_PATH = 'Metro_Interstate_Traffic_Volume.csv'
    
    # 1. Load and preprocess data
    X, y = load_and_preprocess_data(FILE_PATH)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Train, evaluate, and select the best model
    trained_model, results_df, sample_features = train_and_select_best_model(
        X_train, X_test, y_train, y_test
    )
    
    # 3. Generate Comparison Charts
    generate_comparison_charts(results_df)

    # 4. Use the best model for a sample prediction
    if trained_model:
        features_array = sample_features.reshape(1, -1)
        sample_prediction = int(trained_model.predict(features_array)[0])
        
        print("\n--- Sample Prediction Demo ---")
        print(f"Predicted Traffic Volume: {sample_prediction}")