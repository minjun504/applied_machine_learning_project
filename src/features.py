from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class Preprocessor:
    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def build_preprocessor(self, n_components=None, scaled=None, pca=False):
        if scaled == "standard":
            scaler = StandardScaler()
        elif scaled == "minmax":
            scaler = MinMaxScaler()
        elif scaled is None:
            scaler = "passthrough"
        else:
            raise ValueError(f"scaler type unknown: {scaled}")

        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(drop="first"), self.categorical_features),
            ("num", scaler, self.numeric_features),
        ])
        if pca:
            return Pipeline([
                ("preprocessor", preprocessor),
                ("pca", PCA(n_components=n_components))
            ])
        else:
            return preprocessor
        
    def feature_selection(self, X, y, method, k):
        if method == "anova-f":
            selector = SelectKBest(f_classif, k)
        elif method == "mutual_info":
            selector = SelectKBest(mutual_info_classif, k)
        else:
            raise ValueError(f"Unknown selection method, {method}")
        
        selector.fit_transform(X, y)
        mask = selector.get_support()
        return X.columns[mask]
    
if __name__ == "__main__":
    preprocessor = Preprocessor(numerical_features, categorical_features)
    preprocessor.feature_selection()
    