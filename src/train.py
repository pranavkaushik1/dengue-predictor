from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

from data_loader import load_data


def get_data():
    df = load_data()
    df = df.drop(columns=["city", "year", "weekofyear", "week_start_date"])

    X = df.drop("cases", axis=1)
    y = df["cases"]

    return X, y


if __name__ == "__main__":
    X, y = get_data()

    tscv = TimeSeriesSplit(n_splits=5)

    rf_mae, rf_rmse = [], []
    gbr_mae, gbr_rmse = [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n=== Fold {fold} ===")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # --- Random Forest ---
        rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_val)

        rf_mae.append(mean_absolute_error(y_val, rf_preds))
        rf_rmse.append(np.sqrt(mean_squared_error(y_val, rf_preds)))

        # --- Gradient Boosting ---
        gbr = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3
        )

        gbr.fit(X_train, y_train)
        gbr_preds = gbr.predict(X_val)

        gbr_mae.append(mean_absolute_error(y_val, gbr_preds))
        gbr_rmse.append(np.sqrt(mean_squared_error(y_val, gbr_preds)))

        print(f"RF MAE: {rf_mae[-1]:.2f}, RMSE: {rf_rmse[-1]:.2f}")
        print(f"GBR MAE: {gbr_mae[-1]:.2f}, RMSE: {gbr_rmse[-1]:.2f}")

    print("\n=== FINAL RESULTS ===")
    print(f"RF Avg MAE: {np.mean(rf_mae):.2f}")
    print(f"RF Avg RMSE: {np.mean(rf_rmse):.2f}")

    print(f"GBR Avg MAE: {np.mean(gbr_mae):.2f}")
    print(f"GBR Avg RMSE: {np.mean(gbr_rmse):.2f}")
