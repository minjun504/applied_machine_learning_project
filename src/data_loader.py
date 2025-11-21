import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from src.config import RAW_DIR, PROCESSED_DIR

def load_and_split_data(random_state, file_name, test_size=0.2, save_csv=False):
    features = ["sex", "length", "diameter", "height", "whole_weight", 
            "shucked_weight", "viscera_weight", "shell_weight", "rings"]
    data = pd.read_csv(RAW_DIR/file_name, sep=",", names=features)

    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded = encoder.fit_transform(data[["sex"]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["sex"]))
    data = pd.concat([data.drop("sex", axis=1), encoded_df], axis=1)

    conditions = [
        data["rings"].between(0, 7),
        data["rings"].between(8, 10),
        data["rings"].between(11, 15), 
        data["rings"] > 15
    ]
    choices = [1, 2, 3, 4]
    data["age_class"] = np.select(conditions, choices)
    data = data.drop("rings", axis=1)

    X = data.drop("age_class", axis=1)
    y = data["age_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    if save_csv:
        X_train.to_csv(PROCESSED_DIR/f"X_train_{random_state}.csv", index=False)
        X_test.to_csv(PROCESSED_DIR/f"X_test_{random_state}.csv", index=False)
        y_train.to_csv(PROCESSED_DIR/f"y_train_{random_state}.csv", index=False)
        y_test.to_csv(PROCESSED_DIR/f"y_test_{random_state}.csv", index=False)
        print(f"Data is processed and saved to {PROCESSED_DIR}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    pass
