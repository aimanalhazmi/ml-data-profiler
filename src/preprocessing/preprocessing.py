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
from pandas.api.types import is_float_dtype, is_numeric_dtype, is_categorical_dtype
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
        # Identify columns to drop: entirely NaN or only one unique non-null value
        drop_columns = [
            column
            for column in self.data.columns
            if self.data[column].isna().all()
            or self.data[column].dropna().nunique() <= 1
        ]

        numeric_columns = []
        text_columns, categorical_columns = [], []

        #detect numeric-like columns
        for column in self.data.columns:
            series = self.data[column]
            series = series.replace({
                "?"  : np.nan,
                ""   : np.nan,
                "NA" : np.nan,
                "N/A": np.nan
            })
            # Attempt to cast to float if not already numeric
            if pd.api.types.is_numeric_dtype(series) is False:
                try:
                    series = series.astype(float)
                except (ValueError, TypeError):
                    pass
            if pd.api.types.is_numeric_dtype(series):
                # Exclude dropped columns and the target itself
                if column not in drop_columns:
                    unique_values = series.nunique()
                    # treat numeric with <=10 unique values as categorical
                    if column != self.target_column and unique_values>10:
                        numeric_columns.append(column)
                    elif column != self.target_column and unique_values<=10:
                        categorical_columns.append(column)
        
        # Collect object/string dtype columns
        object_columns = self.data.select_dtypes(["object", "string"]).columns.tolist()
        object_columns = [
            column
            for column in object_columns
            if column != self.target_column 
            and column not in drop_columns 
            and column not in numeric_columns
        ]

        # Include explicitly typed categorical columns
        categorical_dtype_columns = [
            column for column in self.data.columns
            if is_categorical_dtype(self.data[column])
            and column not in drop_columns
            and column not in numeric_columns
        ]

        # Include boolean columns
        bool_columns = self.data.select_dtypes("bool").columns.tolist()
        bool_columns = [
            column for column in bool_columns
            if column != self.target_column 
            and column not in drop_columns 
            and column not in numeric_columns
        ]

        # Combine all non-numeric candidate columns
        object_columns.extend(categorical_dtype_columns)
        object_columns.extend(bool_columns)
        
        #split object-like columns by unique-ratio threshold
        for column in object_columns:
            unique_value_ratio = self.data[column].nunique() / self.data.shape[0]
            # >10% uniques treated as text; else categorical
            if unique_value_ratio > 0.1:
                text_columns.append(column)
            else:
                categorical_columns.append(column)

        # Any remaining columns not classified are considered categorical
        all_known = set(numeric_columns) | set(text_columns) | set(categorical_columns) | set(drop_columns)
        for col in self.data.columns:
            if col not in all_known and col != self.target_column:
                categorical_columns.append(col)

        # Deduplicate all columns
        categorical_columns = np.unique(categorical_columns).tolist()
        categorical_columns = np.unique(categorical_columns).tolist()
        categorical_columns = np.unique(categorical_columns).tolist()
 
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

        # Exclude sensitive from numeric
        numeric_columns = [
            column for column in numeric_columns if column not in sensitive_columns
        ]

        # Ensure sensitive columns are included among categoricals
        categorical_columns.extend(
            [
                column
                for column in sensitive_columns
                if column not in categorical_columns
            ]
        )

        # Ensure target column is included if not already classified
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
        # Compute unique ratio and count for decision logic
        unique_ratio = self.data[self.target_column].nunique() / self.data.shape[0]
        unique_values = self.data[self.target_column].nunique()
        series = self.data[self.target_column]

        # Attempt to cast non-numeric series to float
        if pd.api.types.is_numeric_dtype(series) is False:
            try:
                series = series.astype(float)
            except (ValueError, TypeError):
                pass
        
        # For numeric targets bin into two quantiles
        if pd.api.types.is_numeric_dtype(series) and unique_values > 10:
            # Impute missing values using k-NN
            arr = series.to_frame().values
            imputer = KNNImputer(n_neighbors=5, weights="distance", metric="nan_euclidean")
            imputed = imputer.fit_transform(arr).ravel()
            series = pd.Series(imputed, index=self.data.index)

            # Create two quantile-based bins
            cats, bins = pd.qcut(series, q=2, retbins=True, duplicates="drop")

            # Generate readable bin labels
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

        # If the target column is categorical: cast to string and impute
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

        # Convert to binary: 1 for the most frequent class, 0 otherwise
        series = df[self.target_column]
        top = series.mode()[0]
        binary = series.eq(top).astype(int)

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
        # Exclude sensitive columns from processing
        numeric_columns = [
            column for column in numeric_columns if column not in sensitive_columns
        ]

        # Fill missing values in each numeric column with its median
        for column in numeric_columns:
            median = np.nanmedian(self.data[column].values.astype(float))
            self.data[column] = self.data[column].fillna(median)
        
        transformers = []

        # Define a KNN-based imputation pipeline for numeric data
        numeric_pipeline = Pipeline([
            ("imputer", KNNImputer(
                n_neighbors=5,
                weights="distance",        
                metric="nan_euclidean",
                missing_values=np.nan   
            ))])

        # Run the pipeline
        if numeric_columns:
            transformers.append(("numeric", numeric_pipeline, numeric_columns))
        preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")
        numeric_preprocessed = preprocessing.fit_transform(self.data)

        # Convert the result back to a DataFrame with original index and column names
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

        # Iterate through each original text column
        for col in text_columns:
            transformer_name = f"tfidf_{col}"
            # Check if the TF-IDF + SVD pipeline for this column exists
            if transformer_name in preprocessing_pipeline.named_transformers_:
                svd = preprocessing_pipeline.named_transformers_[
                    transformer_name
                ].named_steps["svd"]
                # Generate feature names based on SVD component count
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
        # Exclude any sensitive columns from text processing
        text_columns = [
            column for column in text_columns if column not in sensitive_columns
        ]
        transformers = []

        # Ensure text data is string type and fill missing with empty string
        for col in text_columns:
            self.data[col] = self.data[col].fillna("").astype(str)

        # Define pipeline: TF-IDF vectorization followed by dimensionality reduction
        text_pipeline = Pipeline(
            [
                ("vectorize", TfidfVectorizer(max_features=3000)),
                ("svd", TruncatedSVD(n_components=2)),
            ]
        )

        # Run the pipeline
        for column in text_columns:
            transformers.append((f"tfidf_{column}", text_pipeline, column))
        preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")
        text_preprocessed = preprocessing.fit_transform(self.data)

        text_column_names = self.get_textual_feature_names(preprocessing, text_columns)

        # Create the Dataframe
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

        # Fill missing values with a placeholder before encoding
        self.data[categorical_columns] = self.data[categorical_columns].fillna("<MISSING>")

        transformers = []
        # Define a pipeline: cast to string (ensures consistent dtype) then one-hot encode
        categorical_pipeline = Pipeline(
            [  
                ("to_str", FunctionTransformer(lambda X: X.astype(str), validate=False)),
                ("encode", OneHotEncoder(dtype=int)),
            ]
        )

        if categorical_columns:
            transformers.append(
                ("categorical", categorical_pipeline, categorical_columns)
            )
        preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")

        # Fit & transform to get the encoded matrix
        if len(categorical_columns) > 0:
            categorical_preprocessed = preprocessing.fit_transform(self.data)
            # Convert sparse matrix to array if necessary
            if hasattr(categorical_preprocessed, "toarray"):
                categorical_preprocessed = categorical_preprocessed.toarray()
            else:
                categorical_preprocessed = categorical_preprocessed
            # Retrieve the OneHotEncoder to get feature names
            ohe: OneHotEncoder = preprocessing.named_transformers_[
                "categorical"
            ].named_steps["encode"]
            categorical_columns_names = ohe.get_feature_names_out(categorical_columns)
        else:
            # If no columns to encode, transform yields an empty array
            categorical_preprocessed = preprocessing.fit_transform(self.data)
            categorical_columns_names = []

        # Create the Dataframe
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

        # Define pipeline: fill missing with empty string, then impute most frequent
        categorical_pipeline = Pipeline(
            [
                ("to_str", FunctionTransformer(lambda X: X.fillna("").astype(str), validate=False)),
                ("impute", SimpleImputer(missing_values="",strategy="most_frequent")),
            ]
        )

        # Run the pipeline
        if categorical_columns:
            transformers.append(
                ("categorical", categorical_pipeline, categorical_columns)
            )
        preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")
        categorical_preprocessed = preprocessing.fit_transform(self.data)

        # Create the dataframe
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
        # Define uniform age bin edges and labels (0–100 split into 10 bins)
        n_bins = 10
        age_min, age_max = 0, 100
        bin_edges = np.linspace(age_min, age_max, n_bins + 1)
        age_labels = [
            f"{int(bin_edges[i])}-{int(bin_edges[i+1]) - 1}" for i in range(n_bins)
        ]

        # Pipeline for age-like features: clip, discretize, label bins
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
        
        # Pipeline for other categorical sensitive features: cast to str, impute
        categorical_pipeline = Pipeline(
            [
                ("to_str", FunctionTransformer(lambda X: X.fillna("").astype(str), validate=False)),
                ("impute", SimpleImputer(missing_values="", strategy="most_frequent"))
            ]
        )

        # Build transformers based on column type and keywords
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

        # Run the pipeline
        preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")
        sensitive_preprocessed = preprocessing.fit_transform(self.data)

        # Create the (temporary) Dataframe
        sensitive_df = pd.DataFrame(
            sensitive_preprocessed, columns=sensitive_columns, index=self.data.index
        )

        # Further binning for non-age numeric sensitive columns using quantiles
        for column in sensitive_columns:
            column_low = column.lower()
            if column in numeric_columns and column not in SENSITIVE_KEYWORDS.get(
                "age"
            ):
                series = self.data[column]
                cats, bins = pd.qcut(
                    series, q=10, retbins=True, duplicates="drop"
                )  
                # Create readable labels
                if is_float_dtype(series): 
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
                # Overwrite column with categorical bins
                sensitive_df[column] = pd.Categorical.from_codes(
                    cats.cat.codes, categories=labels
                )
        # Ensure all columns have 'category' dtype
        for column in sensitive_df.columns:
            sensitive_df[column] = sensitive_df[column].astype("category")

        # Optionally one-hot encode all sensitive categorical columns
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
        Replace empty strings, question marks, and purely whitespace entries with pd.NA.

        Returns:
            None
        """
        for column in self.data.columns:
            self.data[column] = self.data[column].replace({"": np.nan, "?": np.nan})
            self.data[column] = self.data[column].replace(r"^\s*$", np.nan, regex=True)

    def ensure_numeric(self, df) -> pd.DataFrame:
        """
        Coerce all columns of the DataFrame to numeric dtype.

        Args:
            df (pd.DataFrame): Input DataFrame with potentially non-numeric columns.

        Returns:
            pd.DataFrame: DataFrame with columns converted to numeric dtype 
        """
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            coerced = pd.to_numeric(df[col])
            df[col] = coerced

        return df
                


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
        # Step 1: initial categorization
        numeric_columns, text_columns, categorical_columns = self.categorize_columns()
        sensitive_columns = []

        # Step 2: normalize missing-value placeholders across all columns
        self.treat_none_values()

        # Step 3: encode numeric columns
        numeric_columns_df = self.encode_numeric_columns(
            numeric_columns, sensitive_columns
        )

        # Step 4: encode text columns
        text_columns_df, text_columns_transformed = self.encode_text_columns(
            text_columns, sensitive_columns
        )
        
        # Step 5: encode categorical columns, choose OHE or pass-through
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

        # Step 6: encode the target column into a binary format
        target_column_df = self.encode_target_column()

        # Step 7: concatenate all processed parts into the final DataFrame
        transformed_data = pd.concat(
            [
                categorical_columns_df,
                numeric_columns_df,
                text_columns_df,
                target_column_df,
            ],
            axis=1,
        )

        # If one-hot encoding was used, ensure all types are numeric
        if ohe:
            transformed_data = self.ensure_numeric(transformed_data)

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
        """
        Preprocessor for fairness task.

        Args:
            data (pd.DataFrame): Input data.
        """
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

        # Fill missing values with column median
        for column in numeric_columns:
            median = np.nanmedian(self.data[column].values.astype(float))
            self.data[column] = self.data[column].fillna(median)

        # Build a pipeline: KNN imputation, winsorization, standard scaling
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

        # Run the pipeline
        if numeric_columns:
            transformers.append(("numeric", numeric_pipeline, numeric_columns))
        preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")
        numeric_preprocessed = preprocessing.fit_transform(self.data)

        # Create the Dataframe
        numeric_df = pd.DataFrame(
            numeric_preprocessed, columns=numeric_columns, index=self.data.index
        )

        return numeric_df

    def process_data(self, ohe: bool = False)-> Tuple[pd.DataFrame, list, list, list, list, str]:
        
        """
        Full fairness pipeline: categorize → encode text/cat/numeric/sensitive/target.

        Args:
            ohe (bool): whether to one-hot encode categoricals.

        Returns:
            same signature as DataQualityPreprocessor.process_data
        """
        # Step 1: initial column categorization
        numeric_columns, text_columns, categorical_columns = self.categorize_columns()

        # Step 2: detect sensitive features among text columns
        sensitive_columns = self.find_sensitive_columns(self.data, text_columns)

        # Step 3: normalize missing-value placeholders across all columns
        self.treat_none_values()

        # Step 4: encode text features
        text_columns_df, text_columns_transformed = self.encode_text_columns(
            text_columns, sensitive_columns
        )

        # Step 5: encode categorical features (OHE or pass-through)
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

        # Step 6: encode numeric features
        numeric_columns_df = self.encode_numeric_columns(
            numeric_columns, sensitive_columns
        )

        # Step 7: encode sensitive features
        sensitive_columns_df = self.encode_sensitive_columns(
            sensitive_columns, numeric_columns, ohe
        )
        # Remove sensitive from numeric list and add to categorical names
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

        # Step 8: encode the target column into binary format
        target_column_df = self.encode_target_column()

        # Step 9: assemble all parts into the final DataFrame
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
        # Ensure numeric dtype if one-hot encoding was used
        if ohe:
            transformed_data = self.ensure_numeric(transformed_data)

        return (
            transformed_data,
            numeric_columns,
            categorical_columns_names,
            text_columns_transformed,
            sensitive_columns,
            self.target_column,
        )
