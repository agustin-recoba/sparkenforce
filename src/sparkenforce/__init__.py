"""
sparkenforce: Type validation for PySpark DataFrames.

This module provides a Dataset type annotation system that allows you to specify
and validate the schema of PySpark DataFrames using Python type hints for both
function arguments and return values.

Example:
    @validate
    def process_data(df: Dataset["id": int, "name": str, "age": int]) -> Dataset["result": str]:
        # Function will validate that df has the required columns and types
        # Return value will also be validated against the specified schema
        return spark.createDataFrame([("processed",)], ["result"])

Apache Software License 2.0

Copyright (c) 2025, Agustín Recoba

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import datetime
import decimal
import inspect
from functools import wraps
from typing import Any, Dict, Set, Tuple, Type, TypeVar, get_type_hints

from pyspark.sql import DataFrame
from pyspark.sql import types as spark_types

__all__ = ["Dataset", "DatasetValidationError", "validate"]


class DatasetValidationError(TypeError):
    """Raised when DataFrame validation fails."""


T = TypeVar("T", bound=callable)


def validate(func: T) -> T:
    """
    Decorator that validates function arguments and return values annotated with Dataset types.

    Args:
        func: Function to decorate with validation

    Returns:
        Wrapped function with validation logic

    Raises:
        DatasetValidationError: When validation fails

    Example:
        @validate
        def process(data: Dataset["id", "name"]) -> Dataset["result": str]:
            return spark.createDataFrame([("success",)], ["result"])
    """
    signature = inspect.signature(func)
    hints = get_type_hints(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)

        # Validate input arguments
        for argument_name, value in bound.arguments.items():
            if argument_name in hints and isinstance(
                hints[argument_name],
                _DatasetMeta,
            ):
                hint = hints[argument_name]
                _validate_dataframe(value, hint, argument_name)

        # Execute the function
        result = func(*args, **kwargs)

        # Validate return value if annotated with Dataset type
        return_annotation = hints.get("return")
        if return_annotation is not None and isinstance(
            return_annotation,
            _DatasetMeta,
        ):
            _validate_dataframe(result, return_annotation, "return value")

        return result

    return wrapper


def _validate_dataframe(value: object, hint: "_DatasetMeta", argument_name: str) -> None:
    """
    Validate a single DataFrame against a Dataset hint.

    Args:
        value: Value to validate
        hint: Dataset metadata with validation rules
        argument_name: Name of the argument being validated

    Raises:
        DatasetValidationError: When validation fails
    """
    if not isinstance(value, DataFrame):
        raise DatasetValidationError(
            f"{argument_name} must be a PySpark DataFrame, got {type(value)}",
        )

    columns = set(value.columns)

    # Check column presence
    if hint.only_specified and columns != hint.columns:
        missing = hint.columns - columns
        extra = columns - hint.columns
        msg_parts = []
        if missing:
            msg_parts.append(f"missing columns: {missing}")
        if extra:
            msg_parts.append(f"unexpected columns: {extra}")
        raise DatasetValidationError(
            f"{argument_name} columns mismatch. Expected exactly {hint.columns}, "
            f"got {columns}. {', '.join(msg_parts)}",
        )

    if not hint.only_specified and not hint.columns.issubset(columns):
        missing = hint.columns - columns
        raise DatasetValidationError(f"{argument_name} is missing required columns: {missing}")

    # Check data types
    if hint.dtypes:
        _validate_dtypes(value, hint.dtypes, argument_name)


def _validate_dtypes(
    df: DataFrame,
    expected_dtypes: dict[str, Any],
    argument_name: str,
) -> None:
    """
    Validate DataFrame column types.

    Args:
        df: DataFrame to validate
        expected_dtypes: Expected column types
        argument_name: Name of the argument being validated

    Raises:
        DatasetValidationError: When type validation fails
    """
    actual_dtypes = dict(df.dtypes)

    for col_name, expected_type in expected_dtypes.items():
        if col_name not in actual_dtypes:
            continue  # Column presence already validated

        actual_type_str = actual_dtypes[col_name]
        actual_spark_type = spark_types.DataType.fromDDL(actual_type_str)

        # Convert expected type to Spark type
        expected_spark_type = _convert_to_spark_type(expected_type)

        if not _types_compatible(actual_spark_type, expected_spark_type):
            raise DatasetValidationError(
                f"{argument_name} column '{col_name}' has incorrect type. "
                f"Expected {expected_type}, got {type(actual_spark_type).__name__}",
            )


def _convert_to_spark_type(python_type: Any) -> type[spark_types.DataType]:
    """
    Convert Python type to Spark DataType.

    Args:
        python_type: Python type to convert

    Returns:
        Corresponding Spark DataType class

    Raises:
        TypeError: When python_type is not a supported type
    """
    if isinstance(python_type, type) and issubclass(python_type, spark_types.DataType):
        return python_type

    if python_type in spark_types._type_mappings:
        return spark_types._type_mappings[python_type]

    # Explicit mapping for common types
    type_mapping = {
        type(None): spark_types.NullType,
        bool: spark_types.BooleanType,
        int: spark_types.LongType,
        float: spark_types.DoubleType,
        str: spark_types.StringType,
        bytearray: spark_types.BinaryType,
        decimal.Decimal: spark_types.DecimalType,
        datetime.date: spark_types.DateType,
        datetime.datetime: spark_types.TimestampType,
        datetime.time: spark_types.TimestampType,
    }

    if python_type in type_mapping:
        return type_mapping[python_type]

    # Raise error for unsupported types instead of silent fallback
    supported_types = list(type_mapping.keys())
    raise TypeError(
        f"Unsupported type for Dataset column validation: {python_type}. "
        f"Supported types are: {', '.join(str(t) for t in supported_types)} "
        f"or any subclass of pyspark.sql.types.DataType",
    )


def _types_compatible(actual: spark_types.DataType, expected: type[spark_types.DataType]) -> bool:
    """
    Check if actual Spark type is compatible with expected type.

    Args:
        actual: Actual Spark DataType instance
        expected: Expected Spark DataType class

    Returns:
        True if types are compatible
    """
    # Direct type match
    if isinstance(actual, expected):
        return True

    # Define compatible type groups for more flexible validation
    integer_types = (
        spark_types.IntegerType,
        spark_types.LongType,
        spark_types.ShortType,
        spark_types.ByteType,
    )
    float_types = (spark_types.FloatType, spark_types.DoubleType)
    string_types = (spark_types.StringType,)
    timestamp_types = (spark_types.TimestampType, spark_types.DateType)

    # Check if actual and expected types are in the same compatibility group
    compatibility_groups = [integer_types, float_types, string_types, timestamp_types]

    return any(isinstance(actual, group) and expected in group for group in compatibility_groups)


def _get_columns_dtypes(parameters: Any) -> tuple[set[str], dict[str, Any]]:
    """
    Extract column names and types from Dataset parameters.

    Args:
        parameters: Dataset parameters to parse

    Returns:
        Tuple of (column_names, column_types)

    Raises:
        TypeError: When parameters are invalid or types are unsupported
    """
    columns: set[str] = set()
    dtypes: dict[str, Any] = {}

    if isinstance(parameters, str):
        columns.add(parameters)
    elif isinstance(parameters, slice):
        if not isinstance(parameters.start, str):
            raise TypeError("Column name must be a string")
        columns.add(parameters.start)
        if parameters.stop is not None:
            # Validate the type early to catch unsupported types
            try:
                _convert_to_spark_type(parameters.stop)
            except TypeError as e:
                # Re-raise with more context about where the error occurred
                raise TypeError(f"Invalid type for column '{parameters.start}': {e!s}") from e
            dtypes[parameters.start] = parameters.stop
    elif isinstance(parameters, (list, tuple, set)):
        for element in parameters:
            sub_columns, sub_dtypes = _get_columns_dtypes(element)
            columns.update(sub_columns)
            dtypes.update(sub_dtypes)
    elif isinstance(parameters, _DatasetMeta):
        columns.update(parameters.columns)
        dtypes.update(parameters.dtypes)
    else:
        raise TypeError(
            f"Dataset parameters must be strings, slices, lists, or DatasetMeta instances. "
            f"Got {type(parameters)}",
        )

    return columns, dtypes


class _DatasetMeta(type):
    """
    Metaclass for Dataset type annotations.

    This metaclass handles the creation of Dataset types with column and type specifications.
    """

    only_specified: bool
    columns: set[str]
    dtypes: dict[str, Any]

    def __getitem__(cls, parameters: object) -> "_DatasetMeta":
        """
        Create a Dataset type with specified columns and types.

        Args:
            parameters: Column specifications (strings, slices, lists, etc.)

        Returns:
            New DatasetMeta instance with validation rules

        Example:
            Dataset["id", "name"]  # Exact columns
            Dataset["id": int, "name": str]  # With types
            Dataset["id", "name", ...]  # Minimum columns
        """
        if not isinstance(parameters, tuple):
            parameters = (parameters,)

        parameters = list(parameters)

        # Check for ellipsis (indicates minimum columns, not exact match)
        only_specified = True
        if parameters and parameters[-1] is Ellipsis:
            only_specified = False
            parameters.pop()

        columns, dtypes = _get_columns_dtypes(parameters)

        # Create new metaclass instance
        meta = _DatasetMeta(
            cls.__name__,
            cls.__bases__ if hasattr(cls, "__bases__") else (),
            {},
        )
        meta.only_specified = only_specified
        meta.columns = columns
        meta.dtypes = dtypes

        return meta

    def __repr__(cls) -> str:
        """String representation of Dataset type."""
        if hasattr(cls, "dtypes") and cls.dtypes:
            type_strs = [
                f"{col}: {dt.__name__ if hasattr(dt, '__name__') else str(dt)}"
                for col, dt in cls.dtypes.items()
            ]
            return f"Dataset[{', '.join(type_strs)}]"

        if hasattr(cls, "columns") and cls.columns:
            return f"Dataset[{', '.join(sorted(cls.columns))}]"

        return "Dataset"


class Dataset(DataFrame, metaclass=_DatasetMeta):
    """
    Type annotation for PySpark DataFrames with schema validation.

    Dataset cannot be instantiated directly. It serves as a type annotation
    to specify the expected schema of DataFrames passed to functions and
    returned from functions.

    Examples:
        Dataset["id", "name"]  # Exact columns required
        Dataset["id": int, "name": str]  # With type checking
        Dataset["id", "name", ...]  # Minimum columns (can have more)
        Dataset[...]  # Any DataFrame

    Usage:
        @validate
        def process_users(users: Dataset["id": int, "name": str]) -> Dataset["result": str]:
            # users is guaranteed to have 'id' (int) and 'name' (str) columns
            # return value will be validated to have 'result' (str) column
            return spark.createDataFrame([("success",)], ["result"])
    """

    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        """Prevent direct instantiation."""
        raise TypeError(
            "Dataset cannot be instantiated directly. "
            "Use spark.createDataFrame() to create DataFrames, "
            "and use Dataset[...] for type annotations only.",
        )


def infer_dataset_type(df: DataFrame) -> str:
    """
    Infer a Dataset[...] annotation string from a PySpark DataFrame's schema.

    Args:
        df: The PySpark DataFrame to inspect.

    Returns:
        A string representing the Dataset annotation, e.g.:
        'Dataset["id": int, "name": str]'
    """
    spark_to_py = {
        spark_types.StringType: str,
        spark_types.BooleanType: bool,
        spark_types.ByteType: int,
        spark_types.ShortType: int,
        spark_types.IntegerType: int,
        spark_types.LongType: int,
        spark_types.FloatType: float,
        spark_types.DoubleType: float,
        spark_types.DecimalType: decimal.Decimal,
        spark_types.DateType: datetime.date,
        spark_types.TimestampType: datetime.datetime,
        spark_types.BinaryType: bytearray,
        spark_types.NullType: type(None),
    }
    fields = []
    for field in df.schema.fields:
        py_type = None
        for spark_type, candidate_py in spark_to_py.items():
            if isinstance(field.dataType, spark_type):
                py_type = candidate_py
                break
        if py_type is None:
            py_type = type(field.dataType)  # fallback to Spark type class
        fields.append(
            f'"{field.name}": {py_type.__name__ if hasattr(py_type, "__name__") else str(py_type)}',
        )
    return f"Dataset[{', '.join(fields)}]"
