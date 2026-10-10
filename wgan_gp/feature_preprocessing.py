
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# These features should contain integer-valued data in the raw dataset.
INTEGER_COLUMNS = {
    "spkts", "dpkts", "sbytes", "dbytes",
    "sloss", "dloss", "sttl", "dttl",
    "swin", "stcpb", "dtcpb", "dwin",
    "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
}

# Categorical columns in UNSW-NB15.
CATEGORICAL_COLUMNS = ["proto", "service", "state"]


class WGANFeatureProcessor:
    """
    Preprocess raw network-flow features for WGAN-GP.

    Constant features are preserved separately.
    Nonconstant numerical features are scaled to [-1, 1].
    Categorical and binary features are one-hot encoded.
    """

    def __init__(self):
        self.fitted = False

    def fit(self, X):
        """Learn feature types and preprocessing from training data."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if X.empty:
            raise ValueError("Cannot fit on an empty dataset.")

        if X.isna().any().any():
            raise ValueError(
                "Input contains missing values. Handle them before fitting."
            )

        self.columns = X.columns.tolist()

        # Identify constant columns using only the supplied training data.
        self.constant_values = {}
        self.constant_columns = []

        for column in self.columns:
            unique_values = X[column].unique()

            if len(unique_values) == 1:
                self.constant_columns.append(column)
                self.constant_values[column] = unique_values[0]

        active_columns = [
            column for column in self.columns
            if column not in self.constant_columns
        ]

        # Detect binary columns with values restricted to 0 and 1.
        self.binary_columns = []

        for column in active_columns:
            if pd.api.types.is_numeric_dtype(X[column]):
                values = set(X[column].unique().tolist())

                if values.issubset({0, 1}):
                    self.binary_columns.append(column)

        # One-hot encode the original categorical and binary columns.
        self.categorical_columns = [
            column for column in CATEGORICAL_COLUMNS
            if column in active_columns
        ]
        self.categorical_columns += [
            column for column in self.binary_columns
            if column not in self.categorical_columns
        ]

        self.numeric_columns = [
            column for column in active_columns
            if column not in self.categorical_columns
        ]

        # Store observed numerical limits for safe decoding.
        self.numeric_min = X[self.numeric_columns].min()
        self.numeric_max = X[self.numeric_columns].max()

        # Scale nonconstant numerical features to [-1, 1].
        self.scaler = None

        if self.numeric_columns:
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.scaler.fit(X[self.numeric_columns])

        # Dense one-hot encoding for the Generator's output vector.
        self.encoder = None

        if self.categorical_columns:
            self.encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
            self.encoder.fit(X[self.categorical_columns])

        self.feature_dim = (
            len(self.numeric_columns)
            + (
                sum(len(categories)
                    for categories in self.encoder.categories_)
                if self.encoder is not None
                else 0
            )
        )

        self.fitted = True
        return self

    def transform(self, X):
        """Convert raw features into the WGAN-GP representation."""
        self._check_fitted()
        self._check_columns(X)

        parts = []

        if self.numeric_columns:
            numeric = self.scaler.transform(
                X[self.numeric_columns]
            )
            parts.append(numeric.astype(np.float32))

        if self.categorical_columns:
            categorical = self.encoder.transform(
                X[self.categorical_columns]
            )
            parts.append(categorical.astype(np.float32))

        if not parts:
            raise ValueError("No nonconstant features remain.")

        result = np.concatenate(parts, axis=1)

        if not np.isfinite(result).all():
            raise ValueError(
                "Processed data contains NaN or infinite values."
            )

        return result

    def inverse_transform(self, generated):
        """
        Decode generated vectors into raw-style feature records.

        Numerical values are inverse-scaled, constrained to observed
        training ranges, and rounded for known integer-valued features.
        Categorical groups are decoded using argmax.
        Constant columns are restored to their training values.
        """
        self._check_fitted()

        generated = np.asarray(generated, dtype=np.float32)

        if generated.ndim != 2:
            raise ValueError("Generated samples must be a 2D array.")

        if generated.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected {self.feature_dim} features, "
                f"received {generated.shape[1]}."
            )

        if not np.isfinite(generated).all():
            raise ValueError("Generated data contains non-finite values.")

        result = pd.DataFrame(index=range(len(generated)))
        offset = 0

        # Decode numerical columns.
        if self.numeric_columns:
            width = len(self.numeric_columns)
            numeric_generated = generated[:, offset:offset + width]
            offset += width

            numeric_decoded = self.scaler.inverse_transform(
                numeric_generated
            )

            for index, column in enumerate(self.numeric_columns):
                values = numeric_decoded[:, index]

                # Keep values within the observed training range.
                values = np.clip(
                    values,
                    self.numeric_min[column],
                    self.numeric_max[column],
                )

                # Restore integer-valued fields.
                if column in INTEGER_COLUMNS:
                    values = np.rint(values)

                result[column] = values

        # Decode categorical and binary one-hot groups.
        if self.categorical_columns:
            for column, categories in zip(
                self.categorical_columns,
                self.encoder.categories_,
            ):
                width = len(categories)
                group = generated[:, offset:offset + width]
                offset += width

                selected = np.argmax(group, axis=1)
                result[column] = categories[selected]

        # Restore constant features exactly.
        for column, value in self.constant_values.items():
            result[column] = value

        # Restore the original feature-column order.
        return result[self.columns]

    def get_metadata(self):
        """Return a concise summary of the learned representation."""
        self._check_fitted()

        return {
            "original_feature_count": len(self.columns),
            "constant_columns": self.constant_columns.copy(),
            "numeric_columns": self.numeric_columns.copy(),
            "categorical_columns": self.categorical_columns.copy(),
            "binary_columns": self.binary_columns.copy(),
            "encoded_feature_count": self.feature_dim,
        }

    def _check_fitted(self):
        if not self.fitted:
            raise RuntimeError(
                "Feature processor has not been fitted yet."
            )

    def _check_columns(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if X.columns.tolist() != self.columns:
            raise ValueError(
                "Input columns or column order differ from training data."
            )

        if X.isna().any().any():
            raise ValueError(
                "Input contains missing values. Handle them first."
            )
