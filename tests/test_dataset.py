import pytest
from sparkenforce import Dataset, infer_dataset_type

from pyspark.sql import SparkSession

spark: "SparkSession" = None


def setup_module(module):
    global spark

    spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()


def test_empty():
    DEmpty = Dataset[...]

    assert DEmpty.columns == set()
    assert DEmpty.dtypes == {}
    assert DEmpty.only_specified == False


def test_columns():
    DName = Dataset["id", "name"]

    assert DName.columns == {"id", "name"}
    assert DName.dtypes == {}
    assert DName.only_specified == True


def test_ellipsis():
    DName = Dataset["id", "name", ...]

    assert DName.columns == {"id", "name"}
    assert DName.dtypes == {}
    assert DName.only_specified == False


def test_dtypes():
    DName = Dataset["id":int, "name":str, "location"]

    assert DName.columns == {"id", "name", "location"}
    assert DName.dtypes == {"id": int, "name": str}
    assert DName.only_specified == True


def test_nested():
    DName = Dataset["id":int, "name":str]
    DLocation = Dataset["id":int, "longitude":float, "latitude":float]

    DNameLoc = Dataset[DName, DLocation]

    assert DNameLoc.columns == {"id", "name", "longitude", "latitude"}
    assert DNameLoc.dtypes == {
        "id": int,
        "name": str,
        "longitude": float,
        "latitude": float,
    }
    assert DNameLoc.only_specified == True

    DNameLocEtc = Dataset[DNameLoc, "description":str, ...]
    assert DNameLocEtc.columns == {"id", "name", "longitude", "latitude", "description"}
    assert DNameLocEtc.dtypes == {
        "id": int,
        "name": str,
        "longitude": float,
        "latitude": float,
        "description": str,
    }
    assert DNameLocEtc.only_specified == False


def test_init():
    with pytest.raises(TypeError):
        Dataset()


def test_infer_dataset_type_basic():
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

    result = infer_dataset_type(df)
    # Accept both 'date' and 'datetime.date' for birthdate, and 'Decimal' or 'decimal.Decimal' for amount
    assert '"id": int' in result
    assert '"name": str' in result
    assert '"active": bool' in result
    assert '"score": float' in result
    assert '"birthdate": date' in result or '"birthdate": datetime.date' in result
    assert '"amount": Decimal' in result or '"amount": decimal.Decimal' in result


def test_infer_dataset_type_nulltype():
    from pyspark.sql.types import StructType, StructField, NullType

    schema = StructType([StructField("maybe", NullType(), True)])
    df = spark.createDataFrame([(None,)], schema=schema)
    result = infer_dataset_type(df)
    assert '"maybe": NoneType' in result or '"maybe": type' in result


def test_infer_dataset_type_binary():
    df = spark.createDataFrame([(b"abc",)], ["data"])
    result = infer_dataset_type(df)
    assert '"data": bytearray' in result
