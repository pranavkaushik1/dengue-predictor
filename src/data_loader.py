from features import preprocess, create_features
import pandas as pd


def load_data():
    features = pd.read_csv("data/raw/dengue_features_train.csv")
    labels = pd.read_csv("data/raw/dengue_labels_train.csv")

    df = features.merge(labels, on=["city", "year", "weekofyear"])
    df = df.rename(columns={"total_cases": "cases"})

    df = preprocess(df)
    df = create_features(df)

    return df


if __name__ == "__main__":
    df = load_data()

    print(df.head())
    print("\nShape AFTER features:", df.shape)
