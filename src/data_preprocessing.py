"""
Data Preprocessing Module
=========================

Handles data loading, missing value imputation, encoding,
train-test split, and SMOTE balancing.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import yaml
from pathlib import Path


def load_config(config_path: str = "config/params.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce DataFrame memory by downcasting numeric types."""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df


def load_data(transaction_path: str, identity_path: str, sample_size: int = None) -> pd.DataFrame:
    """
    Load and merge transaction and identity data.

    Args:
        transaction_path: Path to transaction CSV
        identity_path: Path to identity CSV
        sample_size: If set, sample this many rows (for low-memory environments)

    Returns:
        Merged DataFrame
    """
    print("Loading transaction data...")
    train_transaction = pd.read_csv(transaction_path)
    train_transaction = reduce_memory(train_transaction)

    print("Loading identity data...")
    train_identity = pd.read_csv(identity_path)
    train_identity = reduce_memory(train_identity)

    print("Merging datasets on TransactionID...")
    df = train_transaction.merge(train_identity, on='TransactionID', how='left')

    # Sample if needed (for low-memory environments)
    if sample_size and len(df) > sample_size:
        print(f"Sampling {sample_size:,} rows (low-memory mode)...")
        # Stratified sampling to preserve fraud ratio
        fraud = df[df['isFraud'] == 1]
        legit = df[df['isFraud'] == 0]
        fraud_ratio = len(fraud) / len(df)
        n_fraud = int(sample_size * fraud_ratio)
        n_legit = sample_size - n_fraud
        df = pd.concat([
            fraud.sample(n=min(n_fraud, len(fraud)), random_state=42),
            legit.sample(n=min(n_legit, len(legit)), random_state=42)
        ]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {df['isFraud'].mean()*100:.2f}%")
    print(f"Memory usage: {df.memory_usage().sum()/1e6:.1f} MB")

    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using median for numerical and mode for categorical.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with imputed values
    """
    df = df.copy()

    # Identify column types
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Remove target from numerical if present
    if 'isFraud' in numerical_cols:
        numerical_cols.remove('isFraud')
    if 'TransactionID' in numerical_cols:
        numerical_cols.remove('TransactionID')

    print(f"Imputing {len(numerical_cols)} numerical columns with median...")
    num_imputer = SimpleImputer(strategy='median')
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    print(f"Imputing {len(categorical_cols)} categorical columns with mode...")
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label encode categorical features.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with encoded categorical features
    """
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    print(f"Encoding {len(categorical_cols)} categorical columns...")
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def split_data(df: pd.DataFrame, test_size: float = 0.20,
               random_state: int = 42) -> tuple:
    """
    Split data into train and test sets with stratification.

    Args:
        df: Input DataFrame
        test_size: Proportion for test set
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(['isFraud', 'TransactionID'], axis=1)
    y = df['isFraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training fraud rate: {y_train.mean()*100:.2f}%")

    return X_train, X_test, y_train, y_test


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series,
                random_state: int = 42) -> tuple:
    """
    Apply SMOTE to balance training data.

    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed

    Returns:
        X_train_balanced, y_train_balanced
    """
    print("Applying SMOTE to balance classes...")
    print(f"Before SMOTE - Class distribution:\n{y_train.value_counts()}")

    smote = SMOTE(random_state=random_state)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE - Class distribution:\n{pd.Series(y_balanced).value_counts()}")

    return X_balanced, y_balanced


def preprocess_pipeline(config: dict = None) -> tuple:
    """
    Run full preprocessing pipeline.

    Args:
        config: Configuration dictionary

    Returns:
        X_train_balanced, X_test, y_train_balanced, y_test, feature_names
    """
    if config is None:
        config = load_config()

    # Load data (with optional sampling for low-memory environments)
    sample_size = config['data'].get('sample_size', None)
    df = load_data(
        config['data']['train_transaction'],
        config['data']['train_identity'],
        sample_size=sample_size
    )

    # Preprocess
    df = impute_missing_values(df)
    df = encode_categorical(df)

    # Split
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    # Balance
    X_train_balanced, y_train_balanced = apply_smote(
        X_train, y_train,
        random_state=config['smote']['random_state']
    )

    return X_train_balanced, X_test, y_train_balanced, y_test, X_train.columns.tolist()


if __name__ == "__main__":
    # Test preprocessing
    config = load_config()
    X_train, X_test, y_train, y_test, features = preprocess_pipeline(config)
    print(f"\nPreprocessing complete!")
    print(f"Features: {len(features)}")
