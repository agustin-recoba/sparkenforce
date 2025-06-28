import pytest
from sparkenforce import Dataset


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
