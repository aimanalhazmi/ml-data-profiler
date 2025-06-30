from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import KNNImputer
from scipy.stats.mstats import winsorize
import pandas as pd
import numpy as np
#import utils.preprocessing_utils as utils
from pandas.api.types import is_numeric_dtype
from preprocessing_dict import SENSITIVE_KEYWORDS
from typing import Tuple

class Preprocessor:
    def __init__(self, data: pd.DataFrame, target_column: str):
        """
        Base preprocessor initializer.

        Args:
            data (pd.DataFrame): The input DataFrame to preprocess.
        """
        self.data = data
        self.target_column = target_column

    # categorize columns
    def categorize_columns(self) -> Tuple[list, list, list]:
        """
        Categorize DataFrame columns into numeric, text, and categorical based on dtype and uniqueness.

        Args:
            None

        Returns:
            Tuple[List[str], List[str], List[str]]: 
                numeric_columns: list of numeric column names
                text_columns: list of text column names (high unique ratio)
                categorical_columns: list of categorical column names (low unique ratio)
        """
        numeric_columns = self.data.select_dtypes([np.number]).columns.tolist()
        numeric_columns = [column for column in numeric_columns if column != self.target_column]
        object_columns = self.data.select_dtypes(['object','string']).columns.tolist()
        object_columns = [column for column in object_columns if column != self.target_column]
        text_columns, categorical_columns = [], []
        for column in object_columns:
            unique_value_ratio = self.data[column].nunique() / self.data.shape[0]
            if unique_value_ratio > 0.1: 
                text_columns.append(column)
            else:
                categorical_columns.append(column)

        for column in numeric_columns:
            unique_ratio = self.data[column].nunique() / self.data.shape[0]
            if unique_ratio > 0.1:
                pass
            else:
                categorical_columns.append(column)
                numeric_columns.remove(column)


        return numeric_columns, text_columns, categorical_columns
    
    def receive_categorized_columns(self) -> Tuple[list, list]:
        numeric_columns, text_columns, categorical_columns = self.categorize_columns()
        sensitive_columns = self.find_sensitive_columns(self.data, text_columns)
        numeric_columns = [column for column in numeric_columns if column not in sensitive_columns]
        categorical_columns.extend(
            [column for column in sensitive_columns if column not in categorical_columns]
        )

        return numeric_columns, categorical_columns


    
    def encode_target_column(self) -> pd.DataFrame:
        """
        Pass through numeric columns excluding sensitive ones.

        Args:
            numeric_columns (List[str]): List of numeric column names.
            sensitive_columns (List[str]): List of sensitive column names to exclude.

        Returns:
            pd.DataFrame: DataFrame containing only non-sensitive numeric columns.
        """
        transformers = []


        transformers.append((
            "target", 
            "passthrough", 
            [self.target_column]
        ))

        preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")

        target_preprocessed = preprocessing.fit_transform(self.data)

        target_df = pd.DataFrame(target_preprocessed, columns=[self.target_column], index=self.data.index)

        return target_df

    def encode_numeric_columns(self, numeric_columns: list, sensitive_columns: list) -> pd.DataFrame:
        """
        Pass through numeric columns excluding sensitive ones.

        Args:
            numeric_columns (List[str]): List of numeric column names.
            sensitive_columns (List[str]): List of sensitive column names to exclude.

        Returns:
            pd.DataFrame: DataFrame containing only non-sensitive numeric columns.
        """
        numeric_columns = [column for column in numeric_columns if column not in sensitive_columns]
        transformers = []

        if numeric_columns:
            transformers.append((
                "numeric", 
                "passthrough", 
                numeric_columns
            ))

        preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")

        numeric_preprocessed = preprocessing.fit_transform(self.data)

        numeric_df = pd.DataFrame(numeric_preprocessed, columns=numeric_columns, index=self.data.index)

        return numeric_df
    
    def get_textual_feature_names(self, preprocessing_pipeline: Pipeline, text_columns: list) -> list:
        """
        Retrieve feature names from text pipelines.

        Args:
            preprocessing_pipeline (Pipeline): The fitted ColumnTransformer pipeline.
            text_columns (List[str]): Original text column names.

        Returns:
            List[str]: Generated feature names for text embeddings.
        """
        feature_names = []

        for col in text_columns:
            transformer_name = f"tfidf_{col}"
            if transformer_name in preprocessing_pipeline.named_transformers_:
                svd = preprocessing_pipeline.named_transformers_[transformer_name] \
                                            .named_steps['svd']
                svd_names = [f"{col}_svd_{i}" for i in range(svd.n_components)]
                feature_names.extend(svd_names)

        return feature_names


    def encode_text_columns(self, text_columns: list, sensitive_columns: list) -> Tuple[pd.DataFrame, list]:
        """
        Transform text columns into embeddings using TF-IDF and SVD.

        Args:
            text_columns (List[str]): Text column names to encode.
            sensitive_columns (List[str]): Sensitive columns to exclude.

        Returns:
            Tuple[pd.DataFrame, List[str]]: 
                DataFrame of text embeddings,
                List of embedding feature names.
        """
        text_columns = [column for column in text_columns if column not in sensitive_columns]
        transformers = []
        
        text_pipeline = Pipeline([
            ("vectorize", TfidfVectorizer(max_features=3000)),
            ("svd", TruncatedSVD(n_components=2))
        ])

        for column in text_columns:
            transformers.append((
                f"tfidf_{column}",
                text_pipeline,
                column 
            ))

        preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")

        text_preprocessed = preprocessing.fit_transform(self.data)

        text_column_names = self.get_textual_feature_names(preprocessing, text_columns)

        text_df = pd.DataFrame(text_preprocessed, columns=text_column_names, index=self.data.index)

        return text_df, text_column_names
    

    def encode_categorical_columns_ohe(self, categorical_columns: list, sensitive_columns: list) -> Tuple[pd.DataFrame, list]:
        """
        One-hot encode categorical columns.

        Args:
            categorical_columns (List[str]): Categorical column names to encode.
            sensitive_columns (List[str]): Sensitive columns to exclude.

        Returns:
            Tuple[pd.DataFrame, List[str]]: 
                DataFrame of one-hot encoded features,
                List of generated feature names.
        """
        categorical_columns = [column for column in categorical_columns if column not in sensitive_columns]

        transformers = []
        categorical_pipeline = Pipeline([
            ("encode", OneHotEncoder())
        ])

        if categorical_columns:
            transformers.append(("categorical", categorical_pipeline, categorical_columns))

        preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")

        if len(categorical_columns) > 0:
            categorical_preprocessed = preprocessing.fit_transform(self.data).toarray()
        else:
            categorical_preprocessed = preprocessing.fit_transform(self.data)

        categorical_columns_names = preprocessing.get_feature_names_out()

        categorical_df = pd.DataFrame(categorical_preprocessed, columns=categorical_columns_names, index=self.data.index)

        return categorical_df, list(categorical_columns_names)
    
    def encode_categorical_columns_non_ohe(self, categorical_columns: list, sensitive_columns: list) -> pd.DataFrame:
        """
        Pass through categorical columns without encoding.

        Args:
            categorical_columns (List[str]): Categorical column names.
            sensitive_columns (List[str]): Sensitive columns to exclude.

        Returns:
            pd.DataFrame: DataFrame containing original categorical columns.
        """
        categorical_columns = [column for column in categorical_columns if column not in sensitive_columns]
        transformers = []

        if categorical_columns:
            transformers.append(("categorical", "passthrough", categorical_columns))

        preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")

        categorical_preprocessed = preprocessing.fit_transform(self.data)

        categorical_df = pd.DataFrame(categorical_preprocessed, columns=categorical_columns, index=self.data.index)

        return categorical_df
    
    def encode_sensitive_columns(self, sensitive_columns: list, numeric_columns: list) -> pd.DataFrame:
        """
        Encode sensitive columns (e.g., age binning) and passthrough others.

        Args:
            sensitive_columns (List[str]): Sensitive column names.
            numeric_columns (List[str]): Numeric column names.

        Returns:
            pd.DataFrame: DataFrame with encoded sensitive columns.
        """
        transformers = []

        n_bins = 10
        age_min, age_max = 0, 100     
        bin_edges = np.linspace(age_min, age_max, n_bins+1)
        age_labels = [
            f"{int(bin_edges[i])}-{int(bin_edges[i+1]) - 1}"
            for i in range(n_bins)
        ]

        age_pipeline = Pipeline([

            ("impute", KNNImputer(
                n_neighbors=5,
                weights="distance",
                metric="nan_euclidean"
            )),

            ("clip", FunctionTransformer(
                lambda X: np.clip(X, age_min, age_max), validate=False
            )),
            
            ("ordinal", KBinsDiscretizer(
                n_bins=n_bins,
                encode="ordinal",
                strategy="uniform"
            )),
            
            ("to_label", FunctionTransformer(
                lambda X: np.array(age_labels)[X.astype(int).ravel()]
                        .reshape(-1, 1),
                validate=False
            ))
        ])

        for column in sensitive_columns:
            column_low = column.lower()

            if any(age_keyword in column_low and column in numeric_columns for age_keyword in SENSITIVE_KEYWORDS.get("age")):
                transformers.append((
                    f"sensitive_{column}", 
                    age_pipeline, 
                    [column]
                ))
            else:
                transformers.append((
                    f"sensitive_{column}", 
                    "passthrough", 
                    [column]
                ))

        preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")

        sensitive_preprocessed = preprocessing.fit_transform(self.data)

        sensitive_df = pd.DataFrame(sensitive_preprocessed, columns=sensitive_columns, index=self.data.index)

        for column in sensitive_columns:
            column_low = column.lower()
            if any(socioeconomic_keyword in column_low and column in numeric_columns for socioeconomic_keyword in SENSITIVE_KEYWORDS.get("socioeconomic")):
                cats, bins = pd.qcut(self.data[column], q=10, retbins=True, precision=0, duplicates="drop")

                interval_labels = [
                    f"{int(bins[i]):,}–{int(bins[i+1]) - 1:,}"
                    for i in range(len(bins)-1)
                ]

                sensitive_df[column] = pd.Categorical.from_codes(
                    cats.cat.codes, categories=interval_labels
                )

        for column in sensitive_df.columns:
            sensitive_df[column] = sensitive_df[column].astype("category")

        return sensitive_df
    
    def find_sensitive_columns(self, data: pd.DataFrame, text_columns: list) -> list:
        """
        Identify sensitive columns by keywords in column names.

        Args:
            data (pd.DataFrame): DataFrame to search.

        Returns:
            List[str]: List of sensitive column names.
        """
        sensitive_columns = [
            column
            for column in data.columns
            if column != self.target_column 
            and column not in text_columns
            and any(
                keyword in column.lower() 
                for keywords in SENSITIVE_KEYWORDS.values()
                for keyword in keywords
            )
        ]

        return sensitive_columns

class PreprocessorFactory:
    def __init__(self, data, method, target_column: str):
        """
        Factory for creating a Preprocessor.

        Args:
            data (pd.DataFrame): Input data.
            method (str): Preprocessing method ('data quality' or 'fairness').
        """
        self.data = data
        self.method = method
        self.target_column = target_column
        

    def create(self) -> Preprocessor:
        """
        Instantiate the appropriate Preprocessor based on method.

        Returns:
            Preprocessor: Instance of chosen preprocessor.

        Raises:
            ValueError: If method is unknown.
        """
        if self.method == "data quality":
            return DataQualityPreprocessor(self.data, self.target_column)
        elif self.method == "fairness":
            return FairnessPreprocessor(self.data, self.target_column)
        else:
            raise ValueError("Unknown Preprocessor.")

class DataQualityPreprocessor(Preprocessor):
    def __init__(self, data, target_column):
        """
        Preprocessor for data quality tasks.

        Args:
            data (pd.DataFrame): Input data.
        """
        super().__init__(data, target_column)
    
    def process_data(self, ohe:bool = False) -> Tuple[pd.DataFrame, list, list, list, list, str]:
        """
        Run full preprocessing: categorize and encode columns.

        Args:
            ohe (bool): Whether to one-hot encode categorical columns.

        Returns:
            Tuple containing:
                transformed_data (pd.DataFrame): Final preprocessed DataFrame.
                numeric_columns (List[str]): List of numeric column names.
                categorical_columns (List[str]): List of categorical feature names.
                text_columns_transformed (List[str]): List of text embedding names.
                sensitive_columns (List[str]): List of sensitive column names.
                target_column (str): Placeholder for target column.
        """
        numeric_columns, text_columns, categorical_columns = self.categorize_columns()
        sensitive_columns = self.find_sensitive_columns(self.data, text_columns)

        numeric_columns_df = self.encode_numeric_columns(numeric_columns, sensitive_columns)

        text_columns_df, text_columns_transformed = self.encode_text_columns(text_columns, sensitive_columns)

        if ohe:
            categorical_columns_df, categorical_columns_names = self.encode_categorical_columns_ohe(categorical_columns, sensitive_columns)
        else:
            categorical_columns_df = self.encode_categorical_columns_non_ohe(categorical_columns, sensitive_columns)
            categorical_columns_names = list(categorical_columns)

        sensitive_columns_df = self.encode_sensitive_columns(sensitive_columns, numeric_columns)
        numeric_columns = [column for column in numeric_columns if column not in sensitive_columns]
        categorical_columns_names.extend(
            [column for column in sensitive_columns if column not in categorical_columns_names]
        )

        target_column_df = self.encode_target_column()

        transformed_data = pd.concat([sensitive_columns_df, categorical_columns_df, numeric_columns_df, text_columns_df, target_column_df], axis=1) 

        return transformed_data, numeric_columns, categorical_columns_names, text_columns_transformed, sensitive_columns, self.target_column

class FairnessPreprocessor(Preprocessor):
    def __init__(self, data, target_column):
        super().__init__(data, target_column)

    def encode_numeric_columns(self, numeric_columns: list, sensitive_columns: list) -> pd.DataFrame:
        numeric_columns = [column for column in numeric_columns if column not in sensitive_columns]


        numeric_pipeline = Pipeline([
            ("impute", KNNImputer(n_neighbors=5,
                       weights="distance",
                       metric="nan_euclidean")),
            ("winsorize",FunctionTransformer(
                lambda X: np.vstack([winsorize(X[:, j], limits=(0.01, 0.01)) for j in range(X.shape[1])]).T,
                validate=False)),
            ("scale", StandardScaler())
        ])

        transformers = []

        if numeric_columns:
            transformers.append((
                "numeric", 
                numeric_pipeline, 
                numeric_columns
            ))

        preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")

        numeric_preprocessed = preprocessing.fit_transform(self.data)

        numeric_df = pd.DataFrame(numeric_preprocessed, columns=numeric_columns, index=self.data.index)

        return numeric_df

    def process_data(self, ohe:bool = False):
        numeric_columns, text_columns, categorical_columns = self.categorize_columns()
        sensitive_columns = self.find_sensitive_columns(self.data, text_columns)

        text_columns_df, text_columns_transformed = self.encode_text_columns(text_columns, sensitive_columns)

        if ohe:
            categorical_columns_df, categorical_columns_names = self.encode_categorical_columns_ohe(categorical_columns, sensitive_columns)
        else:
            categorical_columns_df = self.encode_categorical_columns_non_ohe(categorical_columns, sensitive_columns)
            categorical_columns_names = list(categorical_columns)

        numeric_columns_df = self.encode_numeric_columns(numeric_columns, sensitive_columns)

        sensitive_columns_df = self.encode_sensitive_columns(sensitive_columns, numeric_columns)
        numeric_columns = [column for column in numeric_columns if column not in sensitive_columns]
        categorical_columns_names.extend(
            [column for column in sensitive_columns if column not in categorical_columns_names]
        )

        target_column_df = self.encode_target_column()
        
        transformed_data = pd.concat([sensitive_columns_df, categorical_columns_df, numeric_columns_df, text_columns_df, target_column_df], axis=1) 

        return transformed_data, numeric_columns, categorical_columns_names, text_columns_transformed, sensitive_columns, self.target_column