import pandas as pd
import numpy as np
import pytest
import sys

sys.path.append("../src/py/")

from Triangle.triangle.triangle import Triangle

@pytest.fixture
def test_triangle():
    df = pd.DataFrame({
        '12':[10, 10, 10, 10],
        '24':[20, 20, 20, np.nan],
        '36':[30, 30, np.nan, np.nan],
        '48':[40, np.nan, np.nan, np.nan]
    }, index=[2000, 2001, 2002, 2003])
    return Triangle.from_dataframe(df=df, id="t")

def test_triangle_init(test_triangle):
    t = test_triangle
    
    assert t.id == "t", "TRI-001 - Triangle id not set correctly"
    assert t.id is not None, "TRI-001 - Triangle id not set correctly"

    # test triangle properties
    assert isinstance(t.tri, pd.DataFrame), "TRI-002 - Triangle.tri is not a dataframe"
    assert isinstance(t, Triangle), f"TRI-003 - Triangle is not a Triangle data type: {type(t)}"
    
    assert isinstance(t.tri0, pd.DataFrame), "TRI-004 - Triangle.tri0 is not a dataframe"
    assert isinstance(t.triangle, pd.DataFrame), "TRI-005 - Triangle.triangle is not a dataframe"
    assert np.equal(t.tri.fillna(0).values, t.triangle.fillna(0).values).all(), f"""TRI-005-2 - Triangle.tri is not equal to Triangle.triangle:
    tri: {t.tri}

    triangle: {t.triangle}"""
    assert isinstance(t.exposure, pd.Series), f"TRI-006 - Triangle.exposure is not a series: {t.exposure}"

def test_acc_init(test_triangle):
    t = test_triangle
    assert isinstance(t.acc, pd.Series), f"TRI-007 - Triangle.acc is not a series: {type(t.acc)}"
    assert t.acc.name == "accident_period", f"TRI-008 - Triangle.acc name is not 'acc': {t.acc.name}"
    
    # did the conversion to datetime work?
    assert isinstance(t.tri.index, pd.DatetimeIndex), f"""TRI-009 - Triangle.tri index is not a datetime index:
    index: {t.tri.index}
    type: {type(t.tri.index)}
    first element: {t.tri.index[0]}"""

    # did the accident years convert to datetime, with month & day 1/1?
    assert t.tri.index[0].month == 1, "TRI-010 - Accident year month is not 1"
    assert t.tri.index[0].day == 1, "TRI-011 - Accident year day is not 1"
    assert (t.tri.index.tolist() == [pd.Timestamp('2000-01-01 00:00:00'),
                                    pd.Timestamp('2001-01-01 00:00:00'),
                                    pd.Timestamp('2002-01-01 00:00:00'),
                                    pd.Timestamp('2003-01-01 00:00:00')]), "TRI-012 - triangle index (datetimes) is not correct"
    assert t.n_acc == 4, "TRI-013 - Triangle.n_acc is not 4"

def test_dev_init(test_triangle):
    t = test_triangle
    # did the developmment years get read in correctly?
    assert t.n_dev == 4, "TRI-014-A - Triangle.n_dev is not 4"
    assert t.dev.tolist() == ['12', '24', '36', '48'], f"TRI-014-B - Triangle.dev is not correct: {t.dev.tolist()} instead of [12, 24, 36, 48]"
    assert t.dev.name == "development_period", f"TRI-014-C - Triangle.dev name is not 'development_period': {t.dev.name}"

def test_cal_init(test_triangle):
    t = test_triangle
    # did the calendar years get read in correctly?
    assert t.n_cal == 7, f"TRI-016 - Triangle.n_cal is not 4: {t.n_cal}"

def test_n_rows(test_triangle):
    t = test_triangle
    assert t.n_rows == t.n_acc, f"TRI-017 - Triangle.n_rows is not equal to Triangle.n_acc: {t.n_rows} != {t.n_acc}"