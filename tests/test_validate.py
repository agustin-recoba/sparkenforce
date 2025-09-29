from dataclasses import dataclass
from datetime import datetime
from typing import Type, Union

import pytest
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as fn
from pyspark.sql import types as spark_types

from sparkenforce import (
    TypedDataFrame,
    DataFrameValidationError,
    validate,
    register_type_mapping,
)

spark: "SparkSession" = None


def setup_module(module):
    global spark

    spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()


TEST_THESE_CLASSES = [DataFrame, TypedDataFrame]
DFType = Union[Type[DataFrame], Type[TypedDataFrame]]


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_validate_columns(DataFrameClass: DFType):
    df1 = spark.createDataFrame([(1,), (2,), (3,)], ["a"])
    df2 = spark.createDataFrame([(1, 4), (2, 5), (3, 6)], ["a", "b"])
    df3 = spark.createDataFrame([(1, 4, 7), (2, 5, 8), (3, 6, 9)], ["a", "b", "c"])

    @validate
    def process(data: DataFrameClass["a", "b"]):
        pass

    process(df2)

    with pytest.raises(DataFrameValidationError):
        process(df1)
    with pytest.raises(DataFrameValidationError):
        process(df3)


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_validate_combination(DataFrameClass: DFType):
    df1 = spark.createDataFrame([(1,), (2,), (3,)], ["a"])
    df2 = spark.createDataFrame([(1, 4), (2, 5), (3, 6)], ["a", "b"])

    @validate
    def process(data1: DataFrameClass["a"], data2: DataFrameClass["a", "b"]):
        pass

    process(df1, df2)


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_validate_custom_type(DataFrameClass: DFType):
    struct_type = spark_types.StructType(
        [
            spark_types.StructField("forename", spark_types.StringType(), True),
            spark_types.StructField("surname", spark_types.StringType(), True),
        ]
    )

    df1 = spark.createDataFrame(
        [
            ({"forename": "John", "surname": "Doe"},),
            ({"forename": "Jane", "surname": "Doe"},),
        ],
        schema=spark_types.StructType(
            [
                spark_types.StructField(
                    "name",
                    struct_type,
                    True,
                )
            ]
        ),
    )
    df2 = spark.createDataFrame(
        [
            ({"forename": "Jim"},),
        ],
        schema=spark_types.StructType(
            [
                spark_types.StructField(
                    "name",
                    spark_types.StructType(
                        [
                            spark_types.StructField("forename", spark_types.StringType(), True),
                        ]
                    ),
                    True,
                )
            ]
        ),
    )

    @dataclass
    class NameType:
        forename: str
        surname: str

    register_type_mapping(NameType, struct_type)

    @validate
    def process(data1: DataFrameClass["name":NameType], data2: DataFrameClass["name":struct_type]):
        pass

    process(df1, df1)
    with pytest.raises(DataFrameValidationError):
        process(df1, df2)
    with pytest.raises(DataFrameValidationError):
        process(df2, df1)


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_validate_ellipsis(DataFrameClass: DFType):
    df1 = spark.createDataFrame([(1,), (2,), (3,)], ["a"])
    df2 = spark.createDataFrame([(1, 4), (2, 5), (3, 6)], ["a", "b"])
    df3 = spark.createDataFrame([(1, 4, 7), (2, 5, 8), (3, 6, 9)], ["a", "b", "c"])

    @validate
    def process(data: DataFrameClass["a", "b", ...]):
        pass

    process(df2)
    process(df3)
    with pytest.raises(DataFrameValidationError):
        process(df1)


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_validate_empty(DataFrameClass: DFType):
    df = spark.createDataFrame([(1,), (2,), (3,)], ["a"])

    @validate
    def process(data: DataFrameClass[...]):
        pass

    process(df)

    with pytest.raises(DataFrameValidationError):
        process(False)


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_validate_dtypes(DataFrameClass: DFType):
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
        data: DataFrameClass["a":int, "b":float, "c":str, "d":datetime],
    ):
        pass

    @validate
    def process2(data: DataFrameClass["a":float, "b", ...]):
        pass

    @validate
    def process3(data: DataFrameClass["a":datetime, ...]):
        pass

    process1(df)

    with pytest.raises(DataFrameValidationError):
        process2(df)
    with pytest.raises(DataFrameValidationError):
        process3(df)


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_validate_other_types(DataFrameClass: DFType):
    df = spark.createDataFrame([(1,), (2,), (3,)], ["a"])

    @validate
    def process(data: DataFrameClass["a"], other: int):
        pass

    process(df, 3)


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_return_type(DataFrameClass: DFType):
    df = spark.createDataFrame([(1,), (2,), (3,)], ["a"])

    class Klass:
        pass

    @validate
    def process(data: DataFrameClass["a"]) -> int:
        return 2

    @validate
    def process2(data: DataFrameClass["a"]) -> Klass:
        return Klass()

    # @validate
    # def process3(data: DataFrame["a"]) -> "Klass":
    #    return Klass()

    process(df)
    process2(df)
    # process3(df) # -> This scenario fails, issue with eval in get_type_hints (read PEP 563)


# New tests for improved error handling and functionality


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_improved_error_messages(DataFrameClass: DFType):
    """Test that error messages are clear and informative."""
    df1 = spark.createDataFrame([(1,), (2,), (3,)], ["a"])
    df3 = spark.createDataFrame([(1, 4, 7), (2, 5, 8), (3, 6, 9)], ["a", "b", "c"])

    @validate
    def process(data: DataFrameClass["a", "b"]):
        pass

    # Test missing columns error message
    with pytest.raises(DataFrameValidationError, match="missing required columns"):
        process(df1)

    # Test unexpected columns error message
    with pytest.raises(DataFrameValidationError, match="unexpected columns"):
        process(df3)


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_non_dataframe_error(DataFrameClass: DFType):
    """Test error when passing non-DataFrame objects."""

    @validate
    def process(data: DataFrameClass["a"]):
        pass

    with pytest.raises(DataFrameValidationError, match="must be a PySpark DataFrame"):
        process("not a dataframe")

    with pytest.raises(DataFrameValidationError, match="must be a PySpark DataFrame"):
        process([1, 2, 3])

    with pytest.raises(DataFrameValidationError, match="must be a PySpark DataFrame"):
        process(42)


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_missing_required_columns(DataFrameClass: DFType):
    """Test error when required columns are missing in ellipsis mode."""
    df = spark.createDataFrame([(1,), (2,), (3,)], ["a"])

    @validate
    def process(data: DataFrameClass["a", "b", "c", ...]):
        pass

    with pytest.raises(DataFrameValidationError, match="missing required columns"):
        process(df)


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_type_validation_error_messages(DataFrameClass: DFType):
    """Test that type validation errors are informative."""
    df = spark.createDataFrame([(1, "not_a_number"), (2, "also_not"), (3, "nope")], ["a", "b"])

    @validate
    def process(data: DataFrameClass["a":int, "b":float]):
        pass

    with pytest.raises(DataFrameValidationError, match="has incorrect type"):
        process(df)


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_dataframe_repr(DataFrameClass: DFType):
    """Test string representation of DataFrame types."""
    # Simple columns
    DSimple = DataFrameClass["a", "b"]
    assert f"{DataFrameClass.__name__}[a, b]" in repr(DSimple)

    # With types
    DTyped = DataFrameClass["a":int, "b":str]
    assert f"{DataFrameClass.__name__}[a: int, b: str]" in repr(DTyped)

    # Empty TypedDataFrame
    DEmpty = DataFrameClass[...]
    assert repr(DEmpty) == DataFrameClass.__name__


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_nested_dataframe_validation(DataFrameClass: DFType):
    """Test validation with nested DataFrame definitions."""
    df = spark.createDataFrame([(1, "test", 1.5)], ["id", "name", "value"])

    DBase = DataFrameClass["id":int, "name":str]
    DExtended = DataFrameClass[DBase, "value":float]

    @validate
    def process(data: DExtended):
        pass

    # Should work with exact match
    process(df)

    # Should fail with missing column
    df_missing = spark.createDataFrame([(1, "test")], ["id", "name"])
    with pytest.raises(DataFrameValidationError):
        process(df_missing)


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_unsupported_type_validation(DataFrameClass: DFType):
    """Test that unsupported types raise TypeError instead of silent fallback."""

    @validate
    def process_unsupported(data: DataFrameClass["col":list]):
        pass

    df = spark.createDataFrame([(1,)], ["col"])

    # Test with unsupported Python types
    with pytest.raises(TypeError, match="Unsupported type for DataFrame column 'col'"):
        process_unsupported(df)

    # Test with custom class
    class CustomClass:
        pass

    @validate
    def process_custom(data: DataFrameClass["col":CustomClass]):
        pass

    with pytest.raises(TypeError, match="Unsupported type for DataFrame column 'col'"):
        process_custom(df)


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_supported_types(DataFrameClass: DFType):
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
    DataFrameClass["col1":int]
    DataFrameClass["col2":str]
    DataFrameClass["col3":float]
    DataFrameClass["col4":bool]
    DataFrameClass["col5":datetime]

    # Direct Spark types should also work
    DataFrameClass["col6":IntegerType]
    DataFrameClass["col7":StringType]
    DataFrameClass["col8":DoubleType]
    DataFrameClass["col9":BooleanType]
    DataFrameClass["col10":TimestampType]


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_datetime_type_validation(DataFrameClass: DFType):
    """Test that datetime type validation works correctly."""
    from datetime import datetime

    df = spark.createDataFrame([(1, datetime.now())], ["id", "timestamp"])

    @validate
    def process(data: DataFrameClass["id":int, "timestamp":datetime]):
        pass

    # Should work correctly
    process(df)

    # Should fail with wrong type
    df_wrong = spark.createDataFrame([(1, "not_a_timestamp")], ["id", "timestamp"])

    @validate
    def process_wrong(data: DataFrameClass["id":int, "timestamp":datetime]):
        pass

    with pytest.raises(DataFrameValidationError, match="has incorrect type"):
        process_wrong(df_wrong)


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_return_value_validation(DataFrameClass: DFType):
    """Test that return values are validated when annotated with DataFrame types."""
    input_df = spark.createDataFrame([(1, "test")], ["id", "name"])
    valid_return_df = spark.createDataFrame([("result",)], ["output"])
    invalid_return_df = spark.createDataFrame([(1,)], ["wrong_column"])

    @validate
    def valid_return_function(
        data: DataFrameClass["id":int, "name":str],
    ) -> DataFrameClass["output":str]:
        return valid_return_df

    @validate
    def invalid_return_function(
        data: DataFrameClass["id":int, "name":str],
    ) -> DataFrameClass["output":str]:
        return invalid_return_df

    @validate
    def non_dataframe_return(
        data: DataFrameClass["id":int, "name":str],
    ) -> DataFrameClass["output":str]:
        return "not a dataframe"

    # Valid case should work
    result = valid_return_function(input_df)
    assert result == valid_return_df

    # Invalid schema should fail
    with pytest.raises(DataFrameValidationError, match="missing required columns.*unexpected columns"):
        invalid_return_function(input_df)

    # Non-DataFrame return should fail
    with pytest.raises(DataFrameValidationError, match="return value must be a PySpark DataFrame"):
        non_dataframe_return(input_df)


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_return_value_type_validation(DataFrameClass: DFType):
    """Test that return value types are validated correctly."""
    input_df = spark.createDataFrame([(1, "test")], ["id", "name"])

    # Return DataFrame with correct types
    correct_return_df = spark.createDataFrame([(42, "success")], ["count", "status"])

    # Return DataFrame with incorrect types
    incorrect_return_df = spark.createDataFrame([("not_int", 123)], ["count", "status"])

    @validate
    def correct_types_return(
        data: DataFrameClass["id":int, "name":str],
    ) -> DataFrameClass["count":int, "status":str]:
        return correct_return_df

    @validate
    def incorrect_types_return(
        data: DataFrameClass["id":int, "name":str],
    ) -> DataFrameClass["count":int, "status":str]:
        return incorrect_return_df

    # Correct types should work
    result = correct_types_return(input_df)
    assert result == correct_return_df

    # Incorrect types should fail
    with pytest.raises(DataFrameValidationError, match="return value.*has incorrect type"):
        incorrect_types_return(input_df)


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_return_value_ellipsis_validation(DataFrameClass: DFType):
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
        data: DataFrameClass["id":int, "name":str],
    ) -> DataFrameClass["count":int, "status":str, ...]:
        return extended_return_df

    @validate
    def ellipsis_return_invalid(
        data: DataFrameClass["id":int, "name":str],
    ) -> DataFrameClass["count":int, "status":str, ...]:
        return missing_return_df

    # Should work with extra columns
    result = ellipsis_return_valid(input_df)
    assert result == extended_return_df

    # Should fail with missing required columns
    with pytest.raises(DataFrameValidationError, match="return value.*missing required columns"):
        ellipsis_return_invalid(input_df)


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_return_value_no_annotation(DataFrameClass: DFType):
    """Test that functions without return annotations are not validated."""
    input_df = spark.createDataFrame([(1, "test")], ["id", "name"])

    @validate
    def no_return_annotation(data: DataFrameClass["id":int, "name":str]):
        # Return anything - should not be validated
        return "this is not a dataframe but should not be validated"

    @validate
    def explicit_none_return(data: DataFrameClass["id":int, "name":str]) -> None:
        # Return anything - should not be validated since return type is None
        return 42

    # Both should work since there's no DataFrame return annotation
    no_return_annotation(input_df)
    explicit_none_return(input_df)


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_return_value_nested_dataframe(DataFrameClass: DFType):
    """Test return value validation with nested DataFrame definitions."""
    input_df = spark.createDataFrame([(1, "test")], ["id", "name"])

    # Define nested DataFrame types
    BaseReturn = DataFrameClass["id":int, "status":str]
    ExtendedReturn = DataFrameClass[BaseReturn, "timestamp":datetime]

    correct_return_df = spark.createDataFrame(
        [(1, "success", datetime.now())],
        ["id", "status", "timestamp"],
    )

    @validate
    def nested_return_function(data: DataFrameClass["id":int, "name":str]) -> ExtendedReturn:
        return correct_return_df

    # Should work correctly
    result = nested_return_function(input_df)
    assert result == correct_return_df


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_integer_type_compatibility(DataFrameClass: DFType):
    """Test that different Spark integer types are compatible with int annotation."""
    input_df = spark.createDataFrame([(1, "test")], ["id", "name"])

    @validate
    def length_function(
        data: DataFrameClass["id":int, "name":str],
    ) -> DataFrameClass["name":str, "length":int]:
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
