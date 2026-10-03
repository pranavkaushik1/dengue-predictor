from data_loader import load_data
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd


def stream_data(df):
    """
    Simulates real-time weekly data arrival
    """
    for i in range(50, len(df)):
        train_window = df.iloc[:i]
        next_point = df.iloc[i : i + 1]
        yield train_window, next_point


if __name__ == "__main__":
    df = load_data()

    # drop non-feature columns
    df = df.drop(columns=["city", "year", "weekofyear", "week_start_date"])

    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3)

    WINDOW = 100  # sliding window size (key fix)

    predictions = []

    # start after enough history exists
    for i in range(WINDOW, len(df)):
        train_window = df.iloc[i - WINDOW : i]
        next_point = df.iloc[i : i + 1]

        # split features/target
        X_train = train_window.drop("cases", axis=1)
        y_train = train_window["cases"]

        # train on recent window only
        model.fit(X_train, y_train)

        # predict next step
        X_input = next_point.drop("cases", axis=1)
        pred = model.predict(X_input)[0]

        predictions.append(pred)

        print(f"{i}: Predicted cases = {pred:.2f}")
