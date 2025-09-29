from typing import Type, Union
import pytest
from pyspark.sql import SparkSession, DataFrame

from sparkenforce import infer_dataframe_annotation, TypedDataFrame

spark: "SparkSession" = None


def setup_module(module):
    global spark

    spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()


TEST_THESE_CLASSES = [DataFrame, TypedDataFrame]
DFType = Union[Type[DataFrame], Type[TypedDataFrame]]


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_empty(DataFrameClass: DFType):
    DEmpty = DataFrameClass[...]

    assert DEmpty.columns == set()
    assert DEmpty.dtypes == {}
    assert not DEmpty.only_specified


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_columns(DataFrameClass: DFType):
    DName = DataFrameClass["id", "name"]

    assert DName.columns == {"id", "name"}
    assert DName.dtypes == {}
    assert DName.only_specified


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_ellipsis(DataFrameClass: DFType):
    DName = DataFrameClass["id", "name", ...]

    assert DName.columns == {"id", "name"}
    assert DName.dtypes == {}
    assert not DName.only_specified


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_dtypes(DataFrameClass: DFType):
    DName = DataFrameClass["id":int, "name":str, "location"]

    assert DName.columns == {"id", "name", "location"}
    assert DName.dtypes == {"id": int, "name": str}
    assert DName.only_specified


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_nested(DataFrameClass: DFType):
    DName = DataFrameClass["id":int, "name":str]
    DLocation = DataFrameClass["id":int, "longitude":float, "latitude":float]

    DNameLoc = DataFrameClass[DName, DLocation]

    assert DNameLoc.columns == {"id", "name", "longitude", "latitude"}
    assert DNameLoc.dtypes == {
        "id": int,
        "name": str,
        "longitude": float,
        "latitude": float,
    }
    assert DNameLoc.only_specified

    DNameLocEtc = DataFrameClass[DNameLoc, "description":str, ...]
    assert DNameLocEtc.columns == {"id", "name", "longitude", "latitude", "description"}
    assert DNameLocEtc.dtypes == {
        "id": int,
        "name": str,
        "longitude": float,
        "latitude": float,
        "description": str,
    }
    assert not DNameLocEtc.only_specified


@pytest.mark.parametrize("DataFrameClass", TEST_THESE_CLASSES)
def test_init(DataFrameClass: DFType):
    with pytest.raises(TypeError):
        DataFrameClass()


def test_infer_dataframe_annotation_basic():
    import datetime
    import decimal

    df = spark.createDataFrame(
        [
            (
                1,
                "Alice",
                True,
                3.14,
                datetime.date(2020, 1, 1),
                decimal.Decimal("1.23"),
            ),
            (2, "Bob", False, 2.71, datetime.date(2021, 2, 2), decimal.Decimal("4.56")),
        ],
        ["id", "name", "active", "score", "birthdate", "amount"],
    )

    result = infer_dataframe_annotation(df)
    # Accept both 'date' and 'datetime.date' for birthdate, and 'Decimal' or 'decimal.Decimal' for amount
    assert '"id": int' in result
    assert '"name": str' in result
    assert '"active": bool' in result
    assert '"score": float' in result
    assert '"birthdate": date' in result or '"birthdate": datetime.date' in result
    assert '"amount": Decimal' in result or '"amount": decimal.Decimal' in result


def test_infer_dataframe_annotation_nulltype():
    from pyspark.sql.types import NullType, StructField, StructType

    schema = StructType([StructField("maybe", NullType(), True)])
    df = spark.createDataFrame([(None,)], schema=schema)
    result = infer_dataframe_annotation(df)
    assert '"maybe": NoneType' in result or '"maybe": type' in result


def test_infer_dataframe_annotation_binary():
    df = spark.createDataFrame([(b"abc",)], ["data"])
    result = infer_dataframe_annotation(df)
    assert '"data": bytearray' in result
