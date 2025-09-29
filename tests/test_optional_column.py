import pytest
from pyspark.sql import SparkSession, DataFrame
from sparkenforce import TypedDataFrame, validate, DataFrameValidationError
from typing import Optional


TEST_THESE_CLASSES = [DataFrame, TypedDataFrame]


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_optional_column(DataFrameClass):
    spark = SparkSession.builder.master("local[1]").appName("test_optional_column").getOrCreate()
    df_required = spark.createDataFrame([(1,)], ["a"])
    df_optional = spark.createDataFrame([(1, 2)], ["a", "b"])
    df_missing = spark.createDataFrame([(2,)], ["b"])

    @validate
    def process(data: DataFrameClass["a":int, "b" : Optional[int]]):
        return True

    # Should work: required present, optional missing
    assert process(df_required)
    # Should work: both present
    assert process(df_optional)
    # Should fail: required missing
    with pytest.raises(DataFrameValidationError):
        process(df_missing)
