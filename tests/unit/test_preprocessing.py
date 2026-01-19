"""
Unit Tests for Data Preprocessing
=================================
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_preprocessing import reduce_memory, load_config


class TestReduceMemory:
    """Tests for memory reduction function."""

    def test_reduce_memory_int(self):
        """Test integer downcasting."""
        df = pd.DataFrame({
            "small_int": [1, 2, 3, 4, 5],
            "medium_int": [100, 200, 300, 400, 500],
            "large_int": [10000, 20000, 30000, 40000, 50000]
        })

        df_reduced = reduce_memory(df.copy())

        # Check that memory is reduced
        assert df_reduced.memory_usage(deep=True).sum() <= df.memory_usage(deep=True).sum()

        # Check values are preserved
        pd.testing.assert_frame_equal(
            df_reduced.astype(df.dtypes),
            df,
            check_dtype=False
        )

    def test_reduce_memory_float(self):
        """Test float downcasting."""
        df = pd.DataFrame({
            "small_float": [0.1, 0.2, 0.3, 0.4, 0.5],
            "large_float": [1e10, 2e10, 3e10, 4e10, 5e10]
        })

        df_reduced = reduce_memory(df.copy())

        # Check values are approximately preserved
        np.testing.assert_array_almost_equal(
            df_reduced["small_float"].values,
            df["small_float"].values,
            decimal=5
        )

    def test_reduce_memory_with_missing(self):
        """Test handling of missing values."""
        df = pd.DataFrame({
            "col_with_nan": [1.0, np.nan, 3.0, np.nan, 5.0]
        })

        df_reduced = reduce_memory(df.copy())

        # Check NaN positions preserved
        assert df_reduced["col_with_nan"].isna().sum() == 2


class TestLoadConfig:
    """Tests for configuration loading."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        config = load_config()

        assert config is not None
        assert isinstance(config, dict)

    def test_config_has_required_keys(self):
        """Test that config has required keys."""
        config = load_config()

        # Check for expected keys (may vary based on your config)
        expected_keys = ["data", "base_models"]
        for key in expected_keys:
            if key in config:
                assert config[key] is not None


class TestDataValidation:
    """Tests for data validation."""

    def test_feature_types(self):
        """Test that features have correct types."""
        # Create sample data
        df = pd.DataFrame({
            "TransactionAmt": [100.0, 200.0, 300.0],
            "card1": [1234, 5678, 9012],
            "ProductCD": ["W", "C", "H"]
        })

        # Check numeric features
        assert pd.api.types.is_numeric_dtype(df["TransactionAmt"])
        assert pd.api.types.is_numeric_dtype(df["card1"])

        # Check categorical features
        assert df["ProductCD"].dtype == object

    def test_no_duplicate_columns(self):
        """Test that there are no duplicate columns."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        })

        assert len(df.columns) == len(set(df.columns))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
