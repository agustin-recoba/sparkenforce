import pytest
from pyspark.sql import types as spark_types
from sparkenforce import _validate_dtypes
from pyspark.sql import SparkSession


@pytest.fixture(scope="module")
def spark():
    return SparkSession.builder.master("local[1]").appName("Test").getOrCreate()


def test_validate_dtypes_with_spark_type_instance(spark):
    from pyspark.sql import Row

    df = spark.createDataFrame([Row(a=1)], "a int")
    # expected_type is already a Spark type instance
    expected_dtypes = {"a": spark_types.IntegerType()}
    # Should not raise
    _validate_dtypes(df, expected_dtypes, "arg")


def test_validate_dtypes_with_spark_type_class_instantiation(spark):
    from pyspark.sql import Row

    df = spark.createDataFrame([Row(a=1)], "a int")
    # expected_type is a Spark type class (subclass of AtomicType)
    expected_dtypes = {"a": spark_types.IntegerType}
    # Should not raise
    _validate_dtypes(df, expected_dtypes, "arg")


def test_validate_dtypes_with_uninstantiable_spark_type(spark):
    from pyspark.sql import Row

    df = spark.createDataFrame([Row(a=1)], "a int")
    # Use a Spark type class that requires arguments (DecimalType)
    expected_dtypes = {"a": spark_types.CharType}
    with pytest.raises(TypeError) as excinfo:
        _validate_dtypes(df, expected_dtypes, "arg")
    assert "Cannot instantiate Spark type CharType" in str(excinfo.value)
