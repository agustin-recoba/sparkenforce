from datetime import datetime
from typing import ModuleType

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as fn

from sparkenforce import Dataset, DatasetValidationError, validate

spark: "SparkSession" = None


def setup_module(module: ModuleType):
    global spark

    spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()


def test_validate_columns():
    df1 = spark.createDataFrame([(1,), (2,), (3,)], ["a"])
    df2 = spark.createDataFrame([(1, 4), (2, 5), (3, 6)], ["a", "b"])
    df3 = spark.createDataFrame([(1, 4, 7), (2, 5, 8), (3, 6, 9)], ["a", "b", "c"])

    @validate
    def process(data: Dataset["a", "b"]):
        pass

    process(df2)
    with pytest.raises(DatasetValidationError):
        process(df1)
    with pytest.raises(DatasetValidationError):
        process(df3)


def test_validate_combination():
    df1 = spark.createDataFrame([(1,), (2,), (3,)], ["a"])
    df2 = spark.createDataFrame([(1, 4), (2, 5), (3, 6)], ["a", "b"])

    @validate
    def process(data1: Dataset["a"], data2: Dataset["a", "b"]):
        pass

    process(df1, df2)


def test_validate_ellipsis():
    df1 = spark.createDataFrame([(1,), (2,), (3,)], ["a"])
    df2 = spark.createDataFrame([(1, 4), (2, 5), (3, 6)], ["a", "b"])
    df3 = spark.createDataFrame([(1, 4, 7), (2, 5, 8), (3, 6, 9)], ["a", "b", "c"])

    @validate
    def process(data: Dataset["a", "b", ...]):
        pass

    process(df2)
    process(df3)
    with pytest.raises(DatasetValidationError):
        process(df1)


def test_validate_empty():
    df = spark.createDataFrame([(1,), (2,), (3,)], ["a"])

    @validate
    def process(data: Dataset[...]):
        pass

    process(df)

    with pytest.raises(DatasetValidationError):
        process(False)


def test_validate_dtypes():
    df = spark.createDataFrame(
        [
            (1, 4.1, "a", datetime.now().replace(hour=7)),
            (2, 5.1, "b", datetime.now().replace(hour=8)),
            (3, 6.1, "c", datetime.now().replace(hour=9)),
        ],
        ["a", "b", "c", "d"],
    )

    @validate
    def process1(
        data: Dataset["a":int, "b":float, "c":str, "d":datetime],
    ):
        pass

    @validate
    def process2(data: Dataset["a":float, "b", ...]):
        pass

    @validate
    def process3(data: Dataset["a":datetime, ...]):
        pass

    process1(df)

    with pytest.raises(DatasetValidationError):
        process2(df)
    with pytest.raises(DatasetValidationError):
        process3(df)


def test_validate_other_types():
    df = spark.createDataFrame([(1,), (2,), (3,)], ["a"])

    @validate
    def process(data: Dataset["a"], other: int):
        pass

    process(df, 3)


def test_return_type():
    df = spark.createDataFrame([(1,), (2,), (3,)], ["a"])

    class Klass:
        pass

    @validate
    def process(data: Dataset["a"]) -> int:
        return 2

    @validate
    def process2(data: Dataset["a"]) -> Klass:
        return Klass()

    # @validate
    # def process3(data: Dataset["a"]) -> "Klass":
    #    return Klass()

    process(df)
    process2(df)
    # process3(df) # -> This scenario fails, issue with eval in get_type_hints (read PEP 563)


# New tests for improved error handling and functionality


def test_improved_error_messages():
    """Test that error messages are clear and informative."""
    df1 = spark.createDataFrame([(1,), (2,), (3,)], ["a"])
    df2 = spark.createDataFrame([(1, 4), (2, 5), (3, 6)], ["a", "b"])
    df3 = spark.createDataFrame([(1, 4, 7), (2, 5, 8), (3, 6, 9)], ["a", "b", "c"])

    @validate
    def process(data: Dataset["a", "b"]):
        pass

    # Test missing columns error message
    with pytest.raises(DatasetValidationError, match="missing columns"):
        process(df1)

    # Test unexpected columns error message
    with pytest.raises(DatasetValidationError, match="unexpected columns"):
        process(df3)


def test_non_dataframe_error():
    """Test error when passing non-DataFrame objects."""

    @validate
    def process(data: Dataset["a"]):
        pass

    with pytest.raises(DatasetValidationError, match="must be a PySpark DataFrame"):
        process("not a dataframe")

    with pytest.raises(DatasetValidationError, match="must be a PySpark DataFrame"):
        process([1, 2, 3])

    with pytest.raises(DatasetValidationError, match="must be a PySpark DataFrame"):
        process(42)


def test_missing_required_columns():
    """Test error when required columns are missing in ellipsis mode."""
    df = spark.createDataFrame([(1,), (2,), (3,)], ["a"])

    @validate
    def process(data: Dataset["a", "b", "c", ...]):
        pass

    with pytest.raises(DatasetValidationError, match="missing required columns"):
        process(df)


def test_type_validation_error_messages():
    """Test that type validation errors are informative."""
    df = spark.createDataFrame([(1, "not_a_number"), (2, "also_not"), (3, "nope")], ["a", "b"])

    @validate
    def process(data: Dataset["a":int, "b":float]):
        pass

    with pytest.raises(DatasetValidationError, match="has incorrect type"):
        process(df)


def test_dataset_repr():
    """Test string representation of Dataset types."""
    # Simple columns
    DSimple = Dataset["a", "b"]
    assert "Dataset[a, b]" in repr(DSimple)

    # With types
    DTyped = Dataset["a":int, "b":str]
    assert "Dataset[a: int, b: str]" in repr(DTyped)

    # Empty dataset
    DEmpty = Dataset[...]
    assert repr(DEmpty) == "Dataset"


def test_nested_dataset_validation():
    """Test validation with nested Dataset definitions."""
    df = spark.createDataFrame([(1, "test", 1.5)], ["id", "name", "value"])

    DBase = Dataset["id":int, "name":str]
    DExtended = Dataset[DBase, "value":float]

    @validate
    def process(data: DExtended):
        pass

    # Should work with exact match
    process(df)

    # Should fail with missing column
    df_missing = spark.createDataFrame([(1, "test")], ["id", "name"])
    with pytest.raises(DatasetValidationError):
        process(df_missing)


def test_unsupported_type_validation():
    """Test that unsupported types raise TypeError instead of silent fallback."""
    # Test with unsupported Python types
    with pytest.raises(TypeError, match="Unsupported type for Dataset column validation"):
        Dataset["col":list]

    with pytest.raises(TypeError, match="Unsupported type for Dataset column validation"):
        Dataset["col":dict]

    with pytest.raises(TypeError, match="Unsupported type for Dataset column validation"):
        Dataset["col":tuple]

    # Test with custom class
    class CustomClass:
        pass

    with pytest.raises(TypeError, match="Unsupported type for Dataset column validation"):
        Dataset["col":CustomClass]


def test_supported_types():
    """Test that all supported types work correctly."""
    from datetime import datetime

    from pyspark.sql.types import (
        BooleanType,
        DoubleType,
        IntegerType,
        StringType,
        TimestampType,
    )

    # These should all work without raising errors
    Dataset["col1":int]
    Dataset["col2":str]
    Dataset["col3":float]
    Dataset["col4":bool]
    Dataset["col5":datetime]

    # Direct Spark types should also work
    Dataset["col6":IntegerType]
    Dataset["col7":StringType]
    Dataset["col8":DoubleType]
    Dataset["col9":BooleanType]
    Dataset["col10":TimestampType]


def test_datetime_type_validation():
    """Test that datetime type validation works correctly."""
    from datetime import datetime

    df = spark.createDataFrame([(1, datetime.now())], ["id", "timestamp"])

    @validate
    def process(data: Dataset["id":int, "timestamp":datetime]):
        pass

    # Should work correctly
    process(df)

    # Should fail with wrong type
    df_wrong = spark.createDataFrame([(1, "not_a_timestamp")], ["id", "timestamp"])

    @validate
    def process_wrong(data: Dataset["id":int, "timestamp":datetime]):
        pass

    with pytest.raises(DatasetValidationError, match="has incorrect type"):
        process_wrong(df_wrong)


def test_return_value_validation():
    """Test that return values are validated when annotated with Dataset types."""
    input_df = spark.createDataFrame([(1, "test")], ["id", "name"])
    valid_return_df = spark.createDataFrame([("result",)], ["output"])
    invalid_return_df = spark.createDataFrame([(1,)], ["wrong_column"])

    @validate
    def valid_return_function(
        data: Dataset["id":int, "name":str],
    ) -> Dataset["output":str]:
        return valid_return_df

    @validate
    def invalid_return_function(
        data: Dataset["id":int, "name":str],
    ) -> Dataset["output":str]:
        return invalid_return_df

    @validate
    def non_dataframe_return(
        data: Dataset["id":int, "name":str],
    ) -> Dataset["output":str]:
        return "not a dataframe"

    # Valid case should work
    result = valid_return_function(input_df)
    assert result == valid_return_df

    # Invalid schema should fail
    with pytest.raises(DatasetValidationError, match="return value.*missing columns"):
        invalid_return_function(input_df)

    # Non-DataFrame return should fail
    with pytest.raises(DatasetValidationError, match="return value must be a PySpark DataFrame"):
        non_dataframe_return(input_df)


def test_return_value_type_validation():
    """Test that return value types are validated correctly."""
    input_df = spark.createDataFrame([(1, "test")], ["id", "name"])

    # Return DataFrame with correct types
    correct_return_df = spark.createDataFrame([(42, "success")], ["count", "status"])

    # Return DataFrame with incorrect types
    incorrect_return_df = spark.createDataFrame([("not_int", 123)], ["count", "status"])

    @validate
    def correct_types_return(
        data: Dataset["id":int, "name":str],
    ) -> Dataset["count":int, "status":str]:
        return correct_return_df

    @validate
    def incorrect_types_return(
        data: Dataset["id":int, "name":str],
    ) -> Dataset["count":int, "status":str]:
        return incorrect_return_df

    # Correct types should work
    result = correct_types_return(input_df)
    assert result == correct_return_df

    # Incorrect types should fail
    with pytest.raises(DatasetValidationError, match="return value.*has incorrect type"):
        incorrect_types_return(input_df)


def test_return_value_ellipsis_validation():
    """Test return value validation with ellipsis (minimum columns)."""
    input_df = spark.createDataFrame([(1, "test")], ["id", "name"])

    # Return DataFrame with required columns and extra ones
    extended_return_df = spark.createDataFrame(
        [(42, "success", "extra")],
        ["count", "status", "additional"],
    )

    # Return DataFrame missing required columns
    missing_return_df = spark.createDataFrame([("success",)], ["status"])

    @validate
    def ellipsis_return_valid(
        data: Dataset["id":int, "name":str],
    ) -> Dataset["count":int, "status":str, ...]:
        return extended_return_df

    @validate
    def ellipsis_return_invalid(
        data: Dataset["id":int, "name":str],
    ) -> Dataset["count":int, "status":str, ...]:
        return missing_return_df

    # Should work with extra columns
    result = ellipsis_return_valid(input_df)
    assert result == extended_return_df

    # Should fail with missing required columns
    with pytest.raises(DatasetValidationError, match="return value.*missing required columns"):
        ellipsis_return_invalid(input_df)


def test_return_value_no_annotation():
    """Test that functions without return annotations are not validated."""
    input_df = spark.createDataFrame([(1, "test")], ["id", "name"])

    @validate
    def no_return_annotation(data: Dataset["id":int, "name":str]):
        # Return anything - should not be validated
        return "this is not a dataframe but should not be validated"

    @validate
    def explicit_none_return(data: Dataset["id":int, "name":str]) -> None:
        # Return anything - should not be validated since return type is None
        return 42

    # Both should work since there's no Dataset return annotation
    no_return_annotation(input_df)
    explicit_none_return(input_df)


def test_return_value_nested_dataset():
    """Test return value validation with nested Dataset definitions."""
    input_df = spark.createDataFrame([(1, "test")], ["id", "name"])

    # Define nested Dataset types
    BaseReturn = Dataset["id":int, "status":str]
    ExtendedReturn = Dataset[BaseReturn, "timestamp":datetime]

    correct_return_df = spark.createDataFrame(
        [(1, "success", datetime.now())],
        ["id", "status", "timestamp"],
    )

    @validate
    def nested_return_function(data: Dataset["id":int, "name":str]) -> ExtendedReturn:
        return correct_return_df

    # Should work correctly
    result = nested_return_function(input_df)
    assert result == correct_return_df


def test_integer_type_compatibility():
    """Test that different Spark integer types are compatible with int annotation."""
    input_df = spark.createDataFrame([(1, "test")], ["id", "name"])

    @validate
    def length_function(
        data: Dataset["id":int, "name":str],
    ) -> Dataset["name":str, "length":int]:
        """Function that returns IntegerType for length (not LongType)."""
        return data.select(
            data.name,
            fn.length(data.name).alias("length"),  # F.length returns IntegerType
        )

    # This should work - IntegerType should be compatible with int annotation (LongType)
    result = length_function(input_df)
    result.show()

    # Verify the actual type is IntegerType but validation passes
    length_col_type = dict(result.dtypes)["length"]
    assert "int" in length_col_type.lower()  # Should be some form of integer type
