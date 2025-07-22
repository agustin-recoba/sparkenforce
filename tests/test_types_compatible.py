import pytest
from pyspark.sql import types as spark_types
from sparkenforce import _types_compatible, _convert_to_spark_type


def test_direct_type_match():
    assert _types_compatible(spark_types.IntegerType(), spark_types.IntegerType())
    assert _types_compatible(spark_types.StringType(), spark_types.StringType())
    assert not _types_compatible(spark_types.StringType(), spark_types.IntegerType())


def test_python_type_conversion():
    assert _types_compatible(spark_types.IntegerType(), _convert_to_spark_type(int))
    assert _types_compatible(spark_types.StringType(), _convert_to_spark_type(str))
    assert not _types_compatible(spark_types.StringType(), _convert_to_spark_type(int))


def test_array_type():
    arr_actual = spark_types.ArrayType(spark_types.IntegerType())
    arr_expected = spark_types.ArrayType(spark_types.IntegerType())
    assert _types_compatible(arr_actual, arr_expected)


def test_map_type():
    map_actual = spark_types.MapType(spark_types.StringType(), spark_types.IntegerType())
    map_expected = spark_types.MapType(spark_types.StringType(), spark_types.IntegerType())
    assert _types_compatible(map_actual, map_expected)


def test_struct_type():
    struct_actual = spark_types.StructType(
        [
            spark_types.StructField("a", spark_types.IntegerType()),
            spark_types.StructField("b", spark_types.StringType()),
        ]
    )
    struct_expected = spark_types.StructType(
        [
            spark_types.StructField("a", spark_types.IntegerType()),
            spark_types.StructField("b", spark_types.StringType()),
        ]
    )
    assert _types_compatible(struct_actual, struct_expected)
    struct_expected_missing = spark_types.StructType(
        [
            spark_types.StructField("a", spark_types.IntegerType()),
        ]
    )
    assert not _types_compatible(struct_actual, struct_expected_missing)
