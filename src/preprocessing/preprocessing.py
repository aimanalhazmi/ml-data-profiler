from abc import abstractmethod

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    KBinsDiscretizer,
    FunctionTransformer,
    StandardScaler,
)
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import KNNImputer, SimpleImputer
from scipy.stats.mstats import winsorize
import pandas as pd
import numpy as np
from pandas.api.types import is_float_dtype, is_numeric_dtype
from src.preprocessing.preprocessing_dict import SENSITIVE_KEYWORDS
from typing import Tuple, Dict


class Preprocessor:
    def __init__(self, data: pd.DataFrame, target_column: str):
        """
        Base preprocessor initializer.

        Args:
            data (pd.DataFrame): The input DataFrame to preprocess.
            target_column (str): Name of the column we consider the prediction target.
        """
        self.data = data
        self.target_column = target_column

    @abstractmethod
    def process_data(self, ohe: OneHotEncoder):
        pass

    def categorize_columns(self) -> Tuple[list, list, list]:
        """
        Categorize DataFrame columns into numeric, text, and categorical based on dtype and uniqueness.

        Returns:
            Tuple[List[str], List[str], List[str]]:
                numeric_columns: list of numeric column names
                text_columns: list of text column names (high unique ratio)
                categorical_columns: list of categorical column names (low unique ratio)
        """
        drop_columns = [
            column
            for column in self.data.columns
            if self.data[column].isna().all()
            or self.data[column].dropna().nunique() <= 1
        ]
        numeric_columns = []
        text_columns, categorical_columns = [], []
        for column in self.data.columns:
            series = self.data[column]
            if pd.api.types.is_numeric_dtype(series) is False:
                try:
                    series = series.astype(float)
                except (ValueError, TypeError):
                    pass
            if pd.api.types.is_numeric_dtype(series):
                if column not in drop_columns:
                    unique_values = series.nunique()
                    if column != self.target_column and unique_values>10:
                        numeric_columns.append(column)
                    elif column != self.target_column and unique_values<=10:
                        categorical_columns.append(column)
        
        """
        numeric_columns = [
            column
            for column in numeric_columns
            if column != self.target_column and column not in drop_columns
        ]
        """
        object_columns = self.data.select_dtypes(["object", "string"]).columns.tolist()
        object_columns = [
            column
            for column in object_columns
            if column != self.target_column 
            and column not in drop_columns 
            and column not in numeric_columns
        ]
        
        for column in object_columns:
            unique_value_ratio = self.data[column].nunique() / self.data.shape[0]
            if unique_value_ratio > 0.1:
                text_columns.append(column)
            else:
                categorical_columns.append(column)

        """
        print(numeric_columns)
        for column in numeric_columns:
            # unique_ratio = self.data[column].nunique() / self.data.shape[0]
            unique_values = self.data[column].nunique()
            if column == "bar_passed":
                print(unique_values)
            # if unique_ratio > 0.1:
            if unique_values > 10:
                pass
            else:
                categorical_columns.append(column)
                numeric_columns.remove(column)
        """
        return numeric_columns, text_columns, categorical_columns

    def receive_categorized_columns(self) -> Tuple[list, list]:
        """
        Get numeric and categorical column lists after removing any detected sensitive columns.

        Returns:
            Tuple[
                List[str],  # numeric_columns (sans sensitive)
                List[str]   # categorical_columns (including sensitive)
            ]
        """
        numeric_columns, text_columns, categorical_columns = self.categorize_columns()
        sensitive_columns = self.find_sensitive_columns(self.data, text_columns)
        numeric_columns = [
            column for column in numeric_columns if column not in sensitive_columns
        ]
        categorical_columns.extend(
            [
                column
                for column in sensitive_columns
                if column not in categorical_columns
            ]
        )

        if self.target_column in self.data.columns:
            if (self.target_column not in numeric_columns
            and self.target_column not in categorical_columns):
                series = self.data[self.target_column]
                unique_values = series.nunique(dropna=True)
                if is_numeric_dtype(series) and unique_values > 10:
                    numeric_columns.append(self.target_column)
                else:
                    categorical_columns.append(self.target_column)


        return numeric_columns, categorical_columns

    def receive_number_of_columns(self) -> Dict[str, int]:
        """
        Count how many numeric, categorical, and text columns there are (after sensitive removal).

        Returns:
            Dict[str,int]: counts under keys 'numeric_columns', 'categorical_columns', 'text_columns'
        """
        numeric_columns, categorical_columns = self.receive_categorized_columns()
        _, text_columns, _ = self.categorize_columns()

        column_dict = {
            "numeric_columns": len(numeric_columns),
            "categorical_columns": len(categorical_columns),
            "text_columns": len(text_columns),
        }

        return column_dict

    def encode_target_column(self) -> pd.DataFrame:
        """
        Extract the target column as a 1-column DataFrame (passthrough). Encoding every target as a binary target.

        Returns:
            pd.DataFrame: A single-column DF containing the target.
        """
        unique_ratio = self.data[self.target_column].nunique() / self.data.shape[0]
        unique_values = self.data[self.target_column].nunique()
        series = self.data[self.target_column]

        if pd.api.types.is_numeric_dtype(series) is False:
            try:
                series = series.astype(float)
            except (ValueError, TypeError):
                pass

        if pd.api.types.is_numeric_dtype(series) and unique_values > 10:
            arr = series.to_frame().values
            imputer = KNNImputer(n_neighbors=5, weights="distance", metric="nan_euclidean")
            imputed = imputer.fit_transform(arr).ravel()
            series = pd.Series(imputed, index=self.data.index)

            #n = series.shape[0] #series.dropna().shape[0]
            #n_bins = min(10, max(3, int(np.sqrt(n))))
            cats, bins = pd.qcut(series, q=2, retbins=True, duplicates="drop")
            if is_float_dtype(series):
                edges = np.round(bins, 2)
                labels = [
                    f"{bins[i]:.2f}–{bins[i+1]:.2f}" for i in range(len(bins) - 1)
                ]
            else:
                edges = bins.astype(int)
                labels = [
                    f"{edges[i]:,}–{edges[i+1]-1:,}" for i in range(len(edges) - 1)
                ]

            cat = pd.Categorical.from_codes(
                cats.cat.codes, categories=labels, ordered=True
            )

            # Overwriting the "nice" labels

            numeric_labels = list(range(len(labels)))
            cat_numeric = pd.Categorical.from_codes(
                cats.cat.codes, categories=numeric_labels, ordered=True
            )

            df = pd.DataFrame({self.target_column: cat_numeric}, index=self.data.index)
            return df

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
        """
        # If the target column is categorical
        df = self.data[[self.target_column]].copy()
        pipe = Pipeline([
            ("to_str", FunctionTransformer(
                lambda X: X.fillna("").astype(str), validate=False
            )),
            ("impute", SimpleImputer(
                strategy="most_frequent",
                missing_values=""       
            ))
        ])
        df[self.target_column] = pipe.fit_transform(df[[self.target_column]]).ravel()

        series = df[self.target_column]
        top = series.mode()[0]
        binary = series.eq(top).astype(int)

        #cat = df[self.target_column].astype("category")
        #codes = cat.cat.codes
        target_df = pd.DataFrame({self.target_column: binary}, index=self.data.index)

        return target_df

    def encode_numeric_columns(
        self, numeric_columns: list, sensitive_columns: list
    ) -> pd.DataFrame:
        """
        Pass through numeric columns excluding sensitive ones.

        Args:
            numeric_columns (List[str]): List of numeric column names.
            sensitive_columns (List[str]): List of sensitive column names to exclude.

        Returns:
            pd.DataFrame: DataFrame containing only non-sensitive numeric columns.
        """
        numeric_columns = [
            column for column in numeric_columns if column not in sensitive_columns
        ]

        for column in numeric_columns:
            median = np.nanmedian(self.data[column].values.astype(float))
            self.data[column] = self.data[column].fillna(median)
        
        transformers = []

        numeric_pipeline = Pipeline([
            ("imputer", KNNImputer(
                n_neighbors=5,
                weights="distance",        
                metric="nan_euclidean",
                missing_values=np.nan   
            ))])

        if numeric_columns:
            transformers.append(("numeric", numeric_pipeline, numeric_columns))

        preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")

        numeric_preprocessed = preprocessing.fit_transform(self.data)

        numeric_df = pd.DataFrame(
            numeric_preprocessed, columns=numeric_columns, index=self.data.index
        )

        return numeric_df

    def get_textual_feature_names(
        self, preprocessing_pipeline: Pipeline, text_columns: list
    ) -> list:
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
                svd = preprocessing_pipeline.named_transformers_[
                    transformer_name
                ].named_steps["svd"]
                svd_names = [f"{col}_svd_{i}" for i in range(svd.n_components)]
                feature_names.extend(svd_names)

        return feature_names

    def encode_text_columns(
        self, text_columns: list, sensitive_columns: list
    ) -> Tuple[pd.DataFrame, list]:
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
        text_columns = [
            column for column in text_columns if column not in sensitive_columns
        ]
        transformers = []

        for col in text_columns:
            self.data[col] = self.data[col].fillna("").astype(str)

        text_pipeline = Pipeline(
            [
                ("vectorize", TfidfVectorizer(max_features=3000)),
                ("svd", TruncatedSVD(n_components=2)),
            ]
        )

        for column in text_columns:
            transformers.append((f"tfidf_{column}", text_pipeline, column))

        preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")

        text_preprocessed = preprocessing.fit_transform(self.data)

        text_column_names = self.get_textual_feature_names(preprocessing, text_columns)

        text_df = pd.DataFrame(
            text_preprocessed, columns=text_column_names, index=self.data.index
        )

        return text_df, text_column_names

    def encode_categorical_columns_ohe(
        self, categorical_columns: list, sensitive_columns: list
    ) -> Tuple[pd.DataFrame, list]:
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
        categorical_columns = [
            column for column in categorical_columns if column not in sensitive_columns
        ]

        transformers = []
        categorical_pipeline = Pipeline(
            [  
                ("to_str", FunctionTransformer(lambda X: X.fillna("").astype(str), validate=False)),
                ("impute", SimpleImputer(missing_values="", strategy="constant", fill_value="<MISSING>")),
                ("encode", OneHotEncoder(dtype=int)),
            ]
        )

        if categorical_columns:
            transformers.append(
                ("categorical", categorical_pipeline, categorical_columns)
            )

        preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")

        if len(categorical_columns) > 0:
            categorical_preprocessed = preprocessing.fit_transform(self.data)
            if hasattr(categorical_preprocessed, "toarray"):
                categorical_preprocessed = categorical_preprocessed.toarray()
            else:
                categorical_preprocessed = categorical_preprocessed
            ohe: OneHotEncoder = preprocessing.named_transformers_[
                "categorical"
            ].named_steps["encode"]
            categorical_columns_names = ohe.get_feature_names_out(categorical_columns)
        else:
            categorical_preprocessed = preprocessing.fit_transform(self.data)
            categorical_columns_names = []

        categorical_df = pd.DataFrame(
            categorical_preprocessed,
            columns=categorical_columns_names,
            index=self.data.index,
        )

        return categorical_df, list(categorical_columns_names)

    def encode_categorical_columns_non_ohe(
        self, categorical_columns: list, sensitive_columns: list
    ) -> pd.DataFrame:
        """
        Pass through categorical columns without encoding.

        Args:
            categorical_columns (List[str]): Categorical column names.
            sensitive_columns (List[str]): Sensitive columns to exclude.

        Returns:
            pd.DataFrame: DataFrame containing original categorical columns.
        """
        categorical_columns = [
            column for column in categorical_columns if column not in sensitive_columns
        ]
        transformers = []

        categorical_pipeline = Pipeline(
            [
                ("to_str", FunctionTransformer(lambda X: X.fillna("").astype(str), validate=False)),
                ("impute", SimpleImputer(missing_values="",strategy="most_frequent")),
            ]
        )

        if categorical_columns:
            transformers.append(
                ("categorical", categorical_pipeline, categorical_columns)
            )

        preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")

        categorical_preprocessed = preprocessing.fit_transform(self.data)

        categorical_df = pd.DataFrame(
            categorical_preprocessed, columns=categorical_columns, index=self.data.index
        )

        return categorical_df

    def encode_sensitive_columns(
        self, sensitive_columns: list, numeric_columns: list, ohe: bool
    ) -> pd.DataFrame:
        """
        Encode sensitive columns (e.g., age binning) and passthrough others.

        Args:
            sensitive_columns (List[str]): Sensitive column names.
            numeric_columns (List[str]): Numeric column names.
            ohe (bool): Whether to one-hot encode sensitive categorical columns.

        Returns:
            pd.DataFrame: DataFrame with encoded sensitive columns.
        """
        # imputing nan values
        for column in sensitive_columns:
            if column in numeric_columns:
                median = np.nanmedian(self.data[column].values.astype(float))
                self.data[column] = self.data[column].fillna(median)
        
        transformers = []

        n_bins = 10
        age_min, age_max = 0, 100
        bin_edges = np.linspace(age_min, age_max, n_bins + 1)
        age_labels = [
            f"{int(bin_edges[i])}-{int(bin_edges[i+1]) - 1}" for i in range(n_bins)
        ]

        age_pipeline = Pipeline(
            [   
                (
                    "clip",
                    FunctionTransformer(
                        lambda x: np.clip(x, age_min, age_max), validate=False
                    ),
                ),
                (
                    "ordinal",
                    KBinsDiscretizer(
                        n_bins=n_bins, encode="ordinal", strategy="uniform"
                    ),
                ),
                (
                    "to_label",
                    FunctionTransformer(
                        lambda x: np.array(age_labels)[x.astype(int).ravel()].reshape(
                            -1, 1
                        ),
                        validate=False,
                    ),
                ),
            ]
        )

        categorical_pipeline = Pipeline(
            [
                ("to_str", FunctionTransformer(lambda X: X.fillna("").astype(str), validate=False)),
                ("impute", SimpleImputer(missing_values="", strategy="most_frequent"))
            ]
        )


        for column in sensitive_columns:
            column_low = column.lower()
            if any(
                age_keyword in column_low and column in numeric_columns
                for age_keyword in SENSITIVE_KEYWORDS.get("age")
            ):
                transformers.append((f"sensitive_{column}", age_pipeline, [column]))
            elif column in numeric_columns:
                transformers.append((f"sensitive_{column}", "passthrough", [column]))
            else:
                transformers.append((f"sensitive_{column}", categorical_pipeline, [column]))

        preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")

        sensitive_preprocessed = preprocessing.fit_transform(self.data)

        sensitive_df = pd.DataFrame(
            sensitive_preprocessed, columns=sensitive_columns, index=self.data.index
        )

        for column in sensitive_columns:
            column_low = column.lower()
            # if any(socioeconomic_keyword in column_low and column in numeric_columns for socioeconomic_keyword in SENSITIVE_KEYWORDS.get("socioeconomic")):
            if column in numeric_columns and column not in SENSITIVE_KEYWORDS.get(
                "age"
            ):
                series = self.data[column]
                cats, bins = pd.qcut(
                    series, q=10, retbins=True, duplicates="drop"
                )  # precision=0
                if is_float_dtype(series):  # bins.max() - bins.min()) < 10
                    edges = np.round(bins, 2)
                    labels = [
                        f"{edges[i]:.2f}–{edges[i+1]:.2f}"
                        for i in range(len(edges) - 1)
                    ]
                else:
                    edges = bins.astype(int)
                    labels = [
                        f"{edges[i]:,}–{edges[i+1] - 1:,}"
                        for i in range(len(edges) - 1)
                    ]
                sensitive_df[column] = pd.Categorical.from_codes(
                    cats.cat.codes, categories=labels
                )

        for column in sensitive_df.columns:
            sensitive_df[column] = sensitive_df[column].astype("category")

        if ohe and len(sensitive_df.columns) > 0:
            sensitive_df = pd.get_dummies(
                sensitive_df, columns=sensitive_df.columns, dtype=int
            )

        return sensitive_df

    def find_sensitive_columns(self, data: pd.DataFrame, text_columns: list) -> list:
        """
        Identify sensitive columns by keywords in column names.

        Args:
            data (pd.DataFrame): DataFrame to search.
            text_columns (List[str]): columns already treated as text.

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

    def treat_none_values(self) -> None:
        """
        Replace None values with pd.NaN

        Returns:
            None
        """
        for column in self.data.columns:
            self.data[column] = self.data[column].replace({"": np.nan, "?": np.nan})
            self.data[column] = self.data[column].replace(r"^\s*$", np.nan, regex=True)


class PreprocessorFactory:
    def __init__(self, data, method, target_column: str):
        """
        Factory for creating a Preprocessor.

        Args:
            data (pd.DataFrame): Input data.
            method (str): Preprocessing method ('data quality' or 'fairness').
            target_column (str): name of the prediction target.
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

    def process_data(
        self, ohe: bool = False
    ) -> Tuple[pd.DataFrame, list, list, list, list, str]:
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
        sensitive_columns = []

        self.treat_none_values()

        numeric_columns_df = self.encode_numeric_columns(
            numeric_columns, sensitive_columns
        )

        text_columns_df, text_columns_transformed = self.encode_text_columns(
            text_columns, sensitive_columns
        )

        if ohe:
            categorical_columns_df, categorical_columns_names = (
                self.encode_categorical_columns_ohe(
                    categorical_columns, sensitive_columns
                )
            )
        else:
            categorical_columns_df = self.encode_categorical_columns_non_ohe(
                categorical_columns, sensitive_columns
            )
            categorical_columns_names = list(categorical_columns)

        target_column_df = self.encode_target_column()

        transformed_data = pd.concat(
            [
                categorical_columns_df,
                numeric_columns_df,
                text_columns_df,
                target_column_df,
            ],
            axis=1,
        )

        return (
            transformed_data,
            numeric_columns,
            categorical_columns_names,
            text_columns_transformed,
            sensitive_columns,
            self.target_column,
        )


class FairnessPreprocessor(Preprocessor):
    def __init__(self, data, target_column):
        super().__init__(data, target_column)

    def encode_numeric_columns(
        self, numeric_columns: list, sensitive_columns: list
    ) -> pd.DataFrame:
        """
        Impute, winsorize, and scale non-sensitive numeric features.

        Args:
            numeric_columns (List[str]): columns to treat.
            sensitive_columns (List[str]): to exclude.

        Returns:
            pd.DataFrame: processed numeric matrix.
        """
        numeric_columns = [
            column for column in numeric_columns if column not in sensitive_columns
        ]
        for column in numeric_columns:
            median = np.nanmedian(self.data[column].values.astype(float))
            self.data[column] = self.data[column].fillna(median)

        numeric_pipeline = Pipeline(
            [
                (
                    "impute",
                    KNNImputer(
                        n_neighbors=5, weights="distance", metric="nan_euclidean", missing_values=np.nan
                    ),
                ),
                (
                    "winsorize",
                    FunctionTransformer(
                        lambda x: np.vstack(
                            [
                                winsorize(x[:, j], limits=(0.01, 0.01))
                                for j in range(x.shape[1])
                            ]
                        ).T,
                        validate=False,
                    ),
                ),
                ("scale", StandardScaler()),
            ]
        )

        transformers = []

        if numeric_columns:
            transformers.append(("numeric", numeric_pipeline, numeric_columns))

        preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")

        numeric_preprocessed = preprocessing.fit_transform(self.data)

        numeric_df = pd.DataFrame(
            numeric_preprocessed, columns=numeric_columns, index=self.data.index
        )

        return numeric_df

    def process_data(self, ohe: bool = False):
        """
        Full fairness pipeline: categorize → encode text/cat/numeric/sensitive/target.

        Args:
            ohe (bool): whether to one-hot encode categoricals.

        Returns:
            same signature as DataQualityPreprocessor.process_data
        """
        numeric_columns, text_columns, categorical_columns = self.categorize_columns()
        sensitive_columns = self.find_sensitive_columns(self.data, text_columns)

        self.treat_none_values()

        text_columns_df, text_columns_transformed = self.encode_text_columns(
            text_columns, sensitive_columns
        )

        if ohe:
            categorical_columns_df, categorical_columns_names = (
                self.encode_categorical_columns_ohe(
                    categorical_columns, sensitive_columns
                )
            )
        else:
            categorical_columns_df = self.encode_categorical_columns_non_ohe(
                categorical_columns, sensitive_columns
            )
            categorical_columns_names = list(categorical_columns)

        numeric_columns_df = self.encode_numeric_columns(
            numeric_columns, sensitive_columns
        )

        sensitive_columns_df = self.encode_sensitive_columns(
            sensitive_columns, numeric_columns, ohe
        )
        numeric_columns = [
            column for column in numeric_columns if column not in sensitive_columns
        ]
        categorical_columns_names.extend(
            [
                column
                for column in sensitive_columns
                if column not in categorical_columns_names
            ]
        )

        target_column_df = self.encode_target_column()

        transformed_data = pd.concat(
            [
                sensitive_columns_df,
                categorical_columns_df,
                numeric_columns_df,
                text_columns_df,
                target_column_df,
            ],
            axis=1,
        )

        return (
            transformed_data,
            numeric_columns,
            categorical_columns_names,
            text_columns_transformed,
            sensitive_columns,
            self.target_column,
        )
