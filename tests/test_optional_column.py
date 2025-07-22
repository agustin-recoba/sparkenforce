import pytest
from pyspark.sql import SparkSession
from sparkenforce import Dataset, validate, DatasetValidationError
from typing import Optional


def test_optional_column():
    spark = SparkSession.builder.master("local[1]").appName("test_optional_column").getOrCreate()
    df_required = spark.createDataFrame([(1,)], ["a"])
    df_optional = spark.createDataFrame([(1, 2)], ["a", "b"])
    df_missing = spark.createDataFrame([(2,)], ["b"])

    @validate
    def process(data: Dataset["a":int, "b" : Optional[int]]):
        return True

    # Should work: required present, optional missing
    assert process(df_required)
    # Should work: both present
    assert process(df_optional)
    # Should fail: required missing
    with pytest.raises(DatasetValidationError):
        process(df_missing)
