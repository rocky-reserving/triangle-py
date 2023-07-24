import itertools
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append("../")
sys.path.append("../triangle_py")

from triangle_py.triangle import Triangle

n_ays = 4
n_devs = 4

@pytest.fixture
def test_triangle():
    """
    build test triangle
    """
    df = pd.DataFrame({
        '12':[10, 10, 10, 10],
        '24':[20, 20, 20, np.nan],
        '36':[30, 30, np.nan, np.nan],
        '48':[40, np.nan, np.nan, np.nan]
    }, index=[2000, 2001, 2002, 2003])
    return Triangle.from_dataframe(df=df, id="t")

@pytest.fixture
def test_accs():
    out =[pd.Timestamp('2000-01-01 00:00:00'),
          pd.Timestamp('2001-01-01 00:00:00'),
          pd.Timestamp('2002-01-01 00:00:00'),
          pd.Timestamp('2003-01-01 00:00:00')]
    return out

@pytest.fixture
def test_devs():
    out = ['12', '24', '36', '48']
    return out

@pytest.fixture
def test_diagonal():
    return pd.Series([40, 30, 20, 10]).astype(float).values

@pytest.fixture
def test_diagonal3():
    return pd.Series([30, 20, 10]).astype(float).values

@pytest.fixture
def test_ultimate():
    """vol-wtd average all year ultimate"""
    return pd.Series([40, 40, 40, 40]).astype(float).values

@pytest.fixture
def test_incremental():
    df = pd.DataFrame(
        np.array([[10, 10, 10, 10],
                  [10, 10, 10, np.nan],
                  [10, 10, np.nan, np.nan],
                  [10, np.nan, np.nan, np.nan]]),
        index=[2000, 2001, 2002, 2003],
        columns=[12, 24, 36, 48])
    return df

@pytest.fixture
def test_ata_triangle():
    df = pd.DataFrame(
        np.array([[2, 1.5, 1.3, np.nan],
                  [2, 1.5, np.nan, np.nan],
                  [2, np.nan, np.nan, np.nan],
                  [np.nan, np.nan, np.nan, np.nan]]),
        index=[2000, 2001, 2002, 2003],
        columns=[12, 24, 36, 48])
    return df

@pytest.fixture
def test_ata_averages():
    out = {
        'vwa-all': pd.Series([2, 1.5, 1.3, 1]),
        'vwa-4': pd.Series([2, 1.5, 1.3, 1]),
        'vwa-2-tail': pd.Series([2, 1.5, 1.3, 1.05]),
        'vwa-5-tail110': pd.Series([2, 1.5, 1.3, 1.1]),
        'simple-all': pd.Series([2, 1.5, 1.3, 1]),
        'simple-3': pd.Series([2, 1.5, 1.3, 1]),
        'simple-2-tail': pd.Series([2, 1.5, 1.3, 1.05]),
        'medial-all': pd.Series([2, 1.5, 1.3, 1]),
        'medial-all-exhigh': pd.Series([2, 1.5, 1.3, 1]),
        'med-5-ex-hlm-tail105': pd.Series([2, 1.5, 1.3, 1.05]),
    }
    return out

@pytest.fixture
def test_n_cal():
    """
    build test calendar years
    """
    n_cal = n_ays + (n_devs - 1)
    return n_cal

@pytest.fixture
def test_calendar_index():
    out = np.array([[1, 2, 3, 4],
                    [2, 3, 4, 5],
                    [3, 4, 5, 6],
                    [4, 5, 6, 7]])
    return out

@pytest.fixture
def test_melted():
    df = pd.DataFrame({
        'accident_period':[2000, 2001, 2002, 2003,
                           2000, 2001, 2002, 2003,
                           2000, 2001, 2002, 2003,
                           2000, 2001, 2002, 2003],
        'development_period':[12, 12, 12, 12,
                              24, 24, 24, 24,
                              36, 36, 36, 36,
                              48, 48, 48, 48],
        'tri':[10, 10, 10, 10,
                10, 10, 10, 0,
                10, 10, 0, 0,
                10, 0, 0, 0]
    }).astype(float)

    return df

@pytest.fixture
def test_dm_ay_levels():
    """
    build test design matrix
    """
    df = pd.DataFrame({
        'accident_period':[2000, 2001, 2002, 2003,
                           2000, 2001, 2002, 2003,
                           2000, 2001, 2002, 2003,
                           2000, 2001, 2002, 2003],
        'accident_period_2001':[0, 1, 0, 0,
                                0, 1, 0, 0,
                                0, 1, 0, 0,
                                0, 1, 0, 0],
        'accident_period_2002':[0, 0, 1, 0,
                                0, 0, 1, 0,
                                0, 0, 1, 0,
                                0, 0, 1, 0],
        'accident_period_2003':[0, 0, 0, 1,
                                0, 0, 0, 1,
                                0, 0, 0, 1,
                                0, 0, 0, 1]
    })
    return df

@pytest.fixture
def test_dm_ay_trends():
    """
    build test design matrix
    """
    df = pd.DataFrame({
        'accident_period':[2000, 2001, 2002, 2003, 2000, 2001, 2002, 2003,
                           2000, 2001, 2002, 2003, 2000, 2001, 2002, 2003],
        'accident_period_2001':[0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
        'accident_period_2002':[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        'accident_period_2003':[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]})
    return df

@pytest.fixture
def test_base_dm():
    df = pd.DataFrame({
        'tri':[10, 10, 10, 10,
                10, 10, 10, np.nan,
                10, 10, np.nan, np.nan,
                10, np.nan, np.nan, np.nan],
        'is_observed':[1, 1, 1, 1,
                       1, 1, 1, 0,
                       1, 1, 0, 0,
                       1, 0, 0, 0],
        'accident_period':[2000, 2001, 2002, 2003,
                           2000, 2001, 2002, 2003,
                           2000, 2001, 2002, 2003,
                           2000, 2001, 2002, 2003],
        'development_period':[12, 12, 12, 12,
                              24, 24, 24, 24,
                              36, 36, 36, 36,
                              48, 48, 48, 48],
        'accident_period_2001':[0, 1, 0, 0,
                                0, 1, 0, 0,
                                0, 1, 0, 0,
                                0, 1, 0, 0],
        'accident_period_2002':[0, 0, 1, 0,
                                0, 0, 1, 0,
                                0, 0, 1, 0,
                                0, 0, 1, 0],
        'accident_period_2003':[0, 0, 0, 1,
                                0, 0, 0, 1,
                                0, 0, 0, 1,
                                0, 0, 0, 1],
        'development_period_024':[0, 0, 0, 0,
                                  1, 1, 1, 1,
                                  1, 1, 1, 1,
                                  1, 1, 1, 1],
        'development_period_036':[0, 0, 0, 0,
                                  0, 0, 0, 0,
                                  1, 1, 1, 1,
                                  1, 1, 1, 1],
        'development_period_048':[0, 0, 0, 0,
                                  0, 0, 0, 0,
                                  0, 0, 0, 0,
                                  1, 1, 1, 1]
                                  
    })
    return df

def are_triangles_equal(tri_df1:pd.DataFrame, tri_df2:pd.DataFrame) -> bool:
    """
    Check if the values in two triangles are equal, ignoring NaNs
    """
    return np.allclose(tri_df1.fillna(0).values,
                       tri_df2.fillna(0).values,
                       rtol=1e-3,
                       atol=1e-3)

def are_dfs_equal(df1:pd.DataFrame, df2:pd.DataFrame) -> bool:
    """
    Check if the values in two triangles are equal, ignoring NaNs
    """
    return np.allclose(df1.values,
                       df2.values,
                       rtol=1e-3,
                       atol=1e-3)


def test_triangle_init(test_triangle: Triangle):
    """
    test triangle initialization
    """
    # t = triangle_()
    t = test_triangle
    
    assert t.id == "t", "TRI-001 - Triangle id not set correctly"
    assert t.id is not None, "TRI-001 - Triangle id not set correctly"

    # test triangle properties
    assert isinstance(t.tri, pd.DataFrame), "TRI-002 - Triangle.tri is not a dataframe"
    assert isinstance(t, Triangle), f"TRI-003 - Triangle is not a Triangle data type: {type(t)}"
    
    assert isinstance(t.tri0, pd.DataFrame), "TRI-004 - Triangle.tri0 is not a dataframe"
    assert isinstance(t.triangle, pd.DataFrame), "TRI-005 - Triangle.triangle is not a dataframe"
    assert np.equal(t.tri.fillna(0).values, t.triangle.fillna(0).values).all(), f"""
    TRI-005-2 - Triangle.tri is not equal to Triangle.triangle:
    tri: {t.tri}

    triangle: {t.triangle}"""
    assert isinstance(t.exposure, pd.Series), f"""
    TRI-006 - Triangle.exposure is not a series: {t.exposure}"""

def test_acc_init(test_triangle, test_accs):
    # t = triangle_()
    t = test_triangle
    assert isinstance(t.acc, pd.Series), f"""
    TRI-007 - Triangle.acc is not a series: {type(t.acc)}"""
    
    assert t.acc.name == "accident_period", f"""
    TRI-008 - Triangle.acc name is not 'acc': {t.acc.name}"""
    
    # did the conversion to datetime work?
    assert isinstance(t.tri.index, pd.DatetimeIndex), f"""
    TRI-009 - Triangle.tri index is not a datetime index:
    index: {t.tri.index}
    type: {type(t.tri.index)}
    first element: {t.tri.index[0]}"""

    # did the accident years convert to datetime, with month & day 1/1?
    assert t.tri.index[0].month == 1, "TRI-010 - Accident year month is not 1"
    assert t.tri.index[0].day == 1, "TRI-011 - Accident year day is not 1"
    assert (t.tri.index.tolist() == test_accs), """
TRI-012 - triangle index (datetimes) is not correct"""
    assert t.n_acc == n_ays, f"""TRI-013 - Triangle.n_acc is not {n_ays}: {t.n_acc}"""

def test_dev_init(test_triangle, test_devs):
    # t = triangle_()
    t = test_triangle
    dev = test_devs
    # did the developmment years get read in correctly?
    assert t.n_dev == n_devs, f"TRI-014-A - Triangle.n_dev is not {n_devs}: {t.n_dev}"
    assert t.dev.tolist() == dev, f"""TRI-014-B - 
    Triangle.dev is not correct:
    {t.dev.tolist()} instead of {dev}"""
    assert t.dev.name == "development_period", f"""TRI-014-C -
    Triangle.dev name is not 'development_period': {t.dev.name}"""

def test_cal_init(test_triangle, test_n_cal):
    # t = triangle_()
    t = test_triangle
    ncal = test_n_cal
    # did the calendar years get read in correctly?
    assert t.n_cal == ncal, f"TRI-016 - Triangle.n_cal is not {ncal}: {t.n_cal}"

def test_n_rows(test_triangle):
    # t = triangle_()
    t = test_triangle
    assert t.n_rows == t.n_acc, f"""TRI-017 -
    Triangle.n_rows is not equal to Triangle.n_acc: {t.n_rows} != {t.n_acc}"""

def test_from_clipboard(test_triangle):
    # t = triangle_()
    t = test_triangle
    
    # write triangle data frame to clipboard
    t.tri.reset_index(drop=True).reset_index().to_clipboard(index=False)

    # read new triangle from clipboard
    # df2 = pd.read_clipboard()
    t2 = Triangle.from_clipboard(id="t2")

    # check that the two triangles are equal
    assert are_triangles_equal(t.tri, t2.tri), f"""TRI-018 -
Triangle.from_clipboard did not read in the same triangle as Triangle.from_dataframe:
    t2.tri: {t2.tri.values}
    t.tri: {t.tri.values}"""

def test_from_csv():
    # read in triangle from csv
    filename = "../triangle_py/data/mack1994.csv"
    t = Triangle.from_csv(filename, id="mack")
    t.tri = t.tri.astype(float).reset_index(drop=True)

    # read in triangle from csv using pandas
    t2 = pd.read_csv(filename, index_col=0, header=0).astype(float).reset_index(drop=True)

    # make sure the two triangles are equal
    assert are_triangles_equal(t.tri, t2), f"""TRI-019 -
    Triangle.from_csv did not read in the same triangle as pd.read_csv:
    t.tri: {t.tri}
    t2: {t2}"""

def test_from_excel():
    # read in triangle from excel
    filename = "../triangle_py/data/dahms reserve triangles.xlsx"
    t = Triangle.from_excel(filename,
                            sheet_name="rpt",
                            origin_columns=1,
                            id="rpt",
                            use_cal=False,
                            sheet_range="a1:k11")
    t.tri = t.tri.astype(float)
    print(f"t.tri: {t.tri}")

    # read in triangle from excel using pandas
    t2 = (pd.read_excel(filename, sheet_name="rpt")
          .iloc[:10, :11]
          .set_index('accident_period')
          .astype(float))
    print(f"t2: {t2}")

    # make sure the two triangles are equal
    assert are_triangles_equal(t.tri, t2), f"""TRI-020 -
    Triangle.from_excel did not read in the same triangle as pd.read_excel:    
    t.tri: {t.tri}
    t2: {t2}"""

def test_from_mack_1994():
    # read in triangle from method
    t = Triangle.from_mack_1994()

    # read in triangle from csv (already tested .from_csv above)
    t2 = Triangle.from_csv("../triangle_py/data/mack1994.csv", id="paid_loss")

    # make sure the two triangles are equal
    assert are_triangles_equal(t.tri, t2.tri), f"""TRI-021 -
    Triangle.from_mack_1994 did not read in the same triangle as Triangle.from_csv:
    t.tri: {t.tri}
    t2: {t2}"""

def test_from_taylor_ashe():
    # read in triangle from method
    t = Triangle.from_taylor_ashe()

    # read in triangle from csv (already tested .from_csv above)
    t2 = Triangle.from_csv("../triangle_py/data/taylorashe.csv", id="paid_loss")

    # make sure the two triangles are equal
    assert are_triangles_equal(t.tri, t2.tri), f"""TRI-022 -
    Triangle.from_taylor_ashe did not read in the same triangle as Triangle.from_csv:
    t.tri: {t.tri}
    t2: {t2}"""

def test_from_dahms_rpt():
    # read in triangle from method
    t, _ = Triangle.from_dahms()

    # read in triangle from excel (already tested .from_excel above)
    t2 = Triangle.from_excel("../triangle_py/data/dahms reserve triangles.xlsx",
                             sheet_name="rpt",
                             origin_columns=1,
                             id="rpt",
                             use_cal=False,
                             sheet_range="a1:k11")

    # make sure the two triangles are equal
    assert are_triangles_equal(t.tri, t2.tri), f"""TRI-023 -
    Triangle.from_dahms did not read in the same triangle as Triangle.from_excel:
    t.tri: {t.tri}
    t2: {t2}"""

def test_from_dahms_paid():
    # read in triangle from method
    _, t = Triangle.from_dahms()

    # read in triangle from excel (already tested .from_excel above)
    t2 = Triangle.from_excel("../triangle_py/data/dahms reserve triangles.xlsx",
                             sheet_name="paid",
                             origin_columns=1,
                             id="paid",
                             use_cal=False,
                             sheet_range="a1:k11")

    # make sure the two triangles are equal
    assert are_triangles_equal(t.tri, t2.tri), f"""TRI-024 -
    Triangle.from_dahms did not read in the same triangle as Triangle.from_excel:
    t.tri: {t.tri}
    t2: {t2}"""

def test_getTriangle(test_triangle):
    # t = triangle_()
    t = test_triangle

    # get the triangle
    tri = t.getTriangle()

    # make sure the triangle is the same as the triangle in the triangle object
    assert are_triangles_equal(tri, t.tri), f"""TRI-025 -
    Triangle.getTriangle did not return the same triangle as Triangle.tri:
    tri: {tri}
    t.tri: {t.tri}"""

def test_getCalendarIndex(test_triangle, test_calendar_index):
    # t = triangle_()
    t = test_triangle
 
    expected_calendar_index = test_calendar_index

    # get the calendar index from the triangle (values to match expected_calendar_index)
    calendar_index_from_triangle = t.getCalendarIndex().values

    # make sure the calendar index is the same as the expected calendar index
    assert np.array_equal(calendar_index_from_triangle,
                          expected_calendar_index), f"""TRI-026 -
    Triangle.getCalendarIndex did not return the same calendar index as expected:
    calendar_index_from_triangle: {calendar_index_from_triangle}
    expected_calendar_index: {expected_calendar_index}"""

def test_getCurCalendarIndex(test_triangle):
    # t = triangle_()
    t = test_triangle
    expected_cur_calendar_index = n_ays

    # get the current calendar index from the triangle
    cur_calendar_index_from_triangle = t.getCurCalendarIndex()

    # make sure the current calendar index is the same as the expected current
    # calendar index
    assert cur_calendar_index_from_triangle == expected_cur_calendar_index, f"""
    TRI-027 -
    Triangle.getCurCalendarIndex did not return the same current
    calendar index as expected:
    cur_calendar_index_from_triangle: {cur_calendar_index_from_triangle}
    expected_cur_calendar_index: {expected_cur_calendar_index}"""
    

def test_cum_to_inc(test_triangle, test_incremental):
    # t = triangle_()
    t = test_triangle
    expected_inc = test_incremental

    # get the incremental triangle from the cumulative triangle
    inc = t.cum_to_inc(_return=True)

    # get the incremental triangle from the triangle object
    inc_from_triangle = t.incr_triangle

    # make sure the incremental triangle is the same as the expected incremental triangle
    assert are_triangles_equal(inc, expected_inc), f"""TRI-028 -
    Triangle.cum_to_inc did not return the same incremental triangle as expected:
    inc: {inc}
    expected_inc: {expected_inc}"""

    # make sure the incremental triangle is the same as the incremental triangle
    # in the triangle object
    assert are_triangles_equal(expected_inc, inc_from_triangle), f"""TRI-029 -
    Triangle.cum_to_inc did not return the same incremental triangle
    as Triangle.incr_triangle:
    inc: {inc}
    inc_from_triangle: {inc_from_triangle}"""

def test_ata_tri(test_triangle, test_ata_triangle):
    # t = triangle_()
    t = test_triangle
    expected_ata = test_ata_triangle

    # get the ata triangle from the triangle object - round to 1 decimal place
    ata = t._ata_tri().round(1)

    # make sure the ata triangle is the same as the expected ata triangle
    assert are_triangles_equal(ata, expected_ata), f"""TRI-030 -
    Triangle.ata_tri did not return the same ata triangle as expected:
    ata: {ata}
    expected_ata: {expected_ata}"""

def test_vwa(test_triangle, test_ata_averages):
    # t = triangle_()
    t = test_triangle
    
    expected_vwa_all = test_ata_averages['vwa-all']
    expected_vwa_4 = test_ata_averages['vwa-4']
    expected_vwa_2_tail = test_ata_averages['vwa-2-tail'].round(1)

    # get the various vwa's from the triangle object (rounded)
    vwa_all = t._vwa().round(1).reset_index(drop=True)
    vwa_all2 = t._vwa('all').round(1).reset_index(drop=True)
    vwa_4 = t._vwa(4).round(1).reset_index(drop=True)
    vwa_2_tail = t._vwa(2, tail=1.05).round(1).reset_index(drop=True)

    # make sure the vwa's are the same as the expected vwa's
    assert vwa_all.equals(expected_vwa_all), f"""TRI-031 -
    Triangle._vwa did not return the same vwa as expected:
    vwa_all: {vwa_all}
    expected_vwa_all: {expected_vwa_all}"""

    assert vwa_all2.equals(expected_vwa_all), f"""TRI-031-A -
    Triangle._vwa did not return the same vwa as expected:
    vwa_all2: {vwa_all2}
    expected_vwa_all: {expected_vwa_all}"""

    assert vwa_4.equals(expected_vwa_4), f"""TRI-032 -
    Triangle._vwa did not return the same vwa as expected:
    vwa_4: {vwa_4}
    expected_vwa_4: {expected_vwa_4}"""

    assert vwa_2_tail.equals(expected_vwa_2_tail), f"""TRI-033 -
    Triangle._vwa did not return the same vwa as expected:
    vwa_2_tail: {vwa_2_tail}
    expected_vwa_2_tail: {expected_vwa_2_tail}"""

def test_simple_ave_all(test_triangle, test_ata_averages):
    t = test_triangle
    expected_simple_ave_all = test_ata_averages['simple-all']

    # get the simple averages from the triangle object
    simple_ave_all = t._ave_ata('all').round(1).reset_index(drop=True)

    # make sure the simple averages are the same as the expected simple averages
    assert simple_ave_all.equals(expected_simple_ave_all), f"""TRI-034 -
    Triangle._ave_ata did not return the same simple averages as expected:
    simple_ave_all: {simple_ave_all}
    expected_simple_ave_all: {expected_simple_ave_all}"""

def test_simple_ave_3(test_triangle, test_ata_averages):
    t = test_triangle
    expected_simple_ave_3 = test_ata_averages['simple-3']

    # get the simple averages from the triangle object
    simple_ave_3 = t._ave_ata(3).round(1).reset_index(drop=True)

    # make sure the simple averages are the same as the expected simple averages
    assert simple_ave_3.equals(expected_simple_ave_3), f"""TRI-035 -
    Triangle._ave_ata did not return the same simple averages as expected:
    simple_ave_3: {simple_ave_3}
    expected_simple_ave_3: {expected_simple_ave_3}"""

def test_simple_ave_2_tail(test_triangle, test_ata_averages):
    t = test_triangle
    expected_simple_ave_2_tail = test_ata_averages['simple-2-tail'].round(1)

    # get the simple averages from the triangle object
    simple_ave_2_tail = t._ave_ata(2, tail=1.05).round(1).reset_index(drop=True)

    # make sure the simple averages are the same as the expected simple averages
    assert simple_ave_2_tail.equals(expected_simple_ave_2_tail), f"""TRI-036 -
    Triangle._ave_ata did not return the same simple averages as expected:
    simple_ave_2_tail: {simple_ave_2_tail}
    expected_simple_ave_2_tail: {expected_simple_ave_2_tail}"""

def test_medial_all(test_triangle, test_ata_averages):
    t = test_triangle
    expected_medial_all = test_ata_averages['medial-all']

    # get the medial averages from the triangle object
    medial_all = t._medial_ata('all').round(1).reset_index(drop=True)

    # make sure the medial averages are the same as the expected medial averages
    assert medial_all.equals(expected_medial_all), f"""TRI-037 -
    Triangle._medial_ata did not return the same medial averages as expected:
    medial_all: {medial_all}
    expected_medial_all: {expected_medial_all}"""

def test_medial_all_exhigh(test_triangle, test_ata_averages):
    t = test_triangle
    expected_medial_all_exhigh = test_ata_averages['medial-all-exhigh']

    # get the medial averages from the triangle object
    medial_all_exhigh = t._medial_ata('all', excludes='h').round(1).reset_index(drop=True)

    # make sure the medial averages are the same as the expected medial averages
    assert medial_all_exhigh.equals(expected_medial_all_exhigh), f"""TRI-038-A -
    Triangle._medial_ata did not return the same medial averages as expected:
    medial_all_exhigh: {medial_all_exhigh}
    expected_medial_all_exhigh: {expected_medial_all_exhigh}"""

def test_medial_all_exlow(test_triangle):
    t = test_triangle
    expected_medial_all_exlow = pd.Series([2, 1.5, 1.3, 1.0])

    # get the medial averages from the triangle object
    medial_all_exlow = t._medial_ata('all', excludes='l').round(1).reset_index(drop=True)

    # make sure the medial averages are the same as the expected medial averages
    assert medial_all_exlow.equals(expected_medial_all_exlow), f"""TRI-038-B -
    Triangle._medial_ata did not return the same medial averages as expected:
    medial_all_ex_low: {medial_all_exlow}
    expected_medial_all_ex_low: {expected_medial_all_exlow}"""

def test_medial_all_ex_mid(test_triangle):
    t = test_triangle
    expected_medial_all_ex_mid = pd.Series([2, 1.5, 1.3, 1.0])

    # get the medial averages from the triangle object
    medial_all_ex_mid = t._medial_ata('all', excludes='m').round(1).reset_index(drop=True)

    # make sure the medial averages are the same as the expected medial averages
    assert medial_all_ex_mid.equals(expected_medial_all_ex_mid), f"""TRI-038-C -
    Triangle._medial_ata did not return the same medial averages as expected:
    medial_all_ex_mid: {medial_all_ex_mid}
    expected_medial_all_ex_mid: {expected_medial_all_ex_mid}"""

def test_medial_5_ex_high_low(test_triangle):
    t = test_triangle
    expected_medial_5_ex_high_low = pd.Series([2, 1.5, 1.3, 1.0])

    # get the medial averages from the triangle object
    medial_5_ex_high_low = t._medial_ata(5, excludes='hl').round(1).reset_index(drop=True)

    # make sure the medial averages are the same as the expected medial averages
    assert medial_5_ex_high_low.equals(expected_medial_5_ex_high_low), f"""TRI-038-D -
    Triangle._medial_ata did not return the same medial averages as expected:
    medial_5_ex_high_low: {medial_5_ex_high_low}
    expected_medial_5_ex_high_low: {expected_medial_5_ex_high_low}"""

def test_medial_5_ex_high_low_mid_tail105(test_triangle, test_ata_averages):
    t = test_triangle
    expected_medial_5_ex_high_low_mid_tail105 = (test_ata_averages['med-5-ex-hlm-tail105']
                                                .round(1)
                                                .reset_index(drop=True))

    # get the medial averages from the triangle object
    medial_5_ex_high_low_mid_tail105 = (t._medial_ata(5, excludes='hlm', tail=1.05)
                                        .round(1)
                                        .reset_index(drop=True))

    # make sure the medial averages are the same as the expected medial averages
    assert medial_5_ex_high_low_mid_tail105.equals(expected_medial_5_ex_high_low_mid_tail105), f"""TRI-038-E -
    Triangle._medial_ata did not return the same medial averages as expected:
    medial_5_ex_high_low_mid_tail105: {medial_5_ex_high_low_mid_tail105}
    expected_medial_5_ex_high_low_mid_tail105: {expected_medial_5_ex_high_low_mid_tail105}"""

def test_ata_with_tri(test_triangle):
    t = test_triangle

    # get the expected ata with tri (already tested ._ata_tri() above)
    expected_ata_with_tri = t._ata_tri().round(1)

    # get the ata with tri from the triangle object
    ata_with_tri = t.ata('triangle').round(1)

    # make sure the ata with tri are the same as the expected ata with tri
    assert are_triangles_equal(expected_ata_with_tri, ata_with_tri), f"""TRI-039-A -
    Triangle._ata_with_tri did not return the same ata with tri as expected:
    ata_with_tri: {ata_with_tri}
    expected_ata_with_tri: {expected_ata_with_tri}"""

def test_ata_with_vwa_all(test_triangle):
    t = test_triangle

    # get the expected ata with vwa (already tested ._vwa() above)
    expected_ata_with_vwa_all = t._vwa('all').round(1)

    # get the ata with vwa from the triangle object
    ata_with_vwa_all = t.ata('vwa', 'all').round(1)

    # make sure the ata with vwa are the same as the expected ata with vwa
    assert are_triangles_equal(expected_ata_with_vwa_all, ata_with_vwa_all), f"""TRI-039-B -
    Triangle._ata_with_vwa did not return the same ata with vwa as expected:
    ata_with_vwa_all: {ata_with_vwa_all}
    expected_ata_with_vwa_all: {expected_ata_with_vwa_all}"""

def test_ata_with_vwa_5_tail110(test_triangle):
    t = test_triangle

    # get the expected ata with vwa (already tested ._vwa() above)
    expected_ata_with_vwa_5_tail110 = t._vwa(5, tail=1.1).round(1)

    # get the ata with vwa from the triangle object
    ata_with_vwa_5_tail110 = t.ata('vwa', 5, tail=1.1).round(1)

    # make sure the ata with vwa are the same as the expected ata with vwa
    assert are_triangles_equal(expected_ata_with_vwa_5_tail110, ata_with_vwa_5_tail110), f"""TRI-039-C -
    Triangle._ata_with_vwa did not return the same ata with vwa as expected:
    ata_with_vwa_5_tail110: {ata_with_vwa_5_tail110}
    expected_ata_with_vwa_5_tail110: {expected_ata_with_vwa_5_tail110}"""

def test_ata_with_simple3_tail120(test_triangle):
    t = test_triangle

    # get the expected ata with simple3 (already tested ._ave_ata() above)
    expected_ata_with_simple3_tail120 = t._ave_ata(3, tail=1.2).round(1)

    # get the ata with simple3 from the triangle object
    ata_with_simple3_tail120 = t.ata('simple', 3, tail=1.2).round(1)

    # make sure the ata with simple3 are the same as the expected ata with simple3
    assert are_triangles_equal(expected_ata_with_simple3_tail120, ata_with_simple3_tail120), f"""TRI-039-D -
    Triangle._ata_with_simple3 did not return the same ata with simple3 as expected:
    ata_with_simple3_tail120: {ata_with_simple3_tail120}
    expected_ata_with_simple3_tail120: {expected_ata_with_simple3_tail120}"""

def test_ata_with_medial5_ex_high_low_tail105(test_triangle):
    t = test_triangle

    # get the expected ata with medial (already tested ._medial_ata() above)
    expected_ata_with_medial5_ex_high_low_tail105 = t._medial_ata(5, excludes='hl', tail=1.05).round(1)

    # get the ata with medial from the triangle object
    ata_with_medial5_ex_high_low_tail105 = t.ata('medial', 5, excludes='hl', tail=1.05).round(1)

    # make sure the ata with medial are the same as the expected ata with medial
    assert are_triangles_equal(expected_ata_with_medial5_ex_high_low_tail105, ata_with_medial5_ex_high_low_tail105), f"""TRI-039-E -
    Triangle._ata_with_medial did not return the same ata with medial as expected:
    ata_with_medial5_ex_high_low_tail105: {ata_with_medial5_ex_high_low_tail105}
    expected_ata_with_medial5_ex_high_low_tail105: {expected_ata_with_medial5_ex_high_low_tail105}"""

def test_atu_vwa_all(test_triangle, test_ata_averages):
    t = test_triangle

    # expected atu with vwa
    ata = test_ata_averages['vwa-all']
    expected_atu = ata[::-1].cumprod()[::-1].round(1).reset_index(drop=True)
    print(f"expected_atu: {expected_atu}")

    # get the atu with vwa from the triangle object
    atu = t.atu('vwa', 'all').round(1).reset_index(drop=True)

    # make sure the atu with vwa are the same as the expected atu with vwa
    assert np.allclose(expected_atu.values, atu.values, rtol=1e-1, atol=1e-1), f"""TRI-040-A -
    Triangle.atu('vwa', 'all') did not return the same atu with vwa as expected:
    atu: {atu}
    expected_atu: {expected_atu}""" 

def test_atu_vwa_5_tail(test_triangle, test_ata_averages):
    t = test_triangle

    # expected atu with vwa
    ata = test_ata_averages['vwa-5-tail110']
    expected_atu = ata[::-1].cumprod()[::-1].round(1).reset_index(drop=True)
    print(f"expected_atu: {expected_atu}")

    # get the atu with vwa from the triangle object
    atu = t.atu('vwa', 5, tail=1.1).round(1).reset_index(drop=True)

    # make sure the atu with vwa are the same as the expected atu with vwa
    assert np.allclose(expected_atu.values, atu.values, rtol=1e-1, atol=1e-1), f"""TRI-040-B -
    Triangle.atu('vwa', 5, tail=1.1) did not return the same atu with vwa as expected:
    atu: {atu}
    expected_atu: {expected_atu}""" 

def test_atu_simple3(test_triangle, test_ata_averages):
    t = test_triangle

    # expected atu with vwa
    ata = test_ata_averages['simple-3']
    expected_atu = ata[::-1].cumprod()[::-1].round(1).reset_index(drop=True)
    print(f"expected_atu: {expected_atu}")

    # get the atu with vwa from the triangle object
    atu = t.atu('simple', 3).round(1).reset_index(drop=True)

    # make sure the atu with vwa are the same as the expected atu with vwa
    assert np.allclose(expected_atu.values, atu.values, rtol=1e-1, atol=1e-1), f"""TRI-040-C -
    Triangle.atu('simple', 3) did not return the same atu with vwa as expected:
    atu: {atu}
    expected_atu: {expected_atu}"""
    
def test_atu_medial_5_exhlm_tail105(test_triangle, test_ata_averages):
    t = test_triangle

    # expected atu with vwa
    ata = test_ata_averages['med-5-ex-hlm-tail105']
    expected_atu = ata[::-1].cumprod()[::-1].round(1).reset_index(drop=True)
    print(f"expected_atu: {expected_atu}")

    # get the atu with vwa from the triangle object
    atu = t.atu('medial', 5, tail=1.05).round(1).reset_index(drop=True)

    # make sure the atu with vwa are the same as the expected atu with vwa
    assert np.allclose(expected_atu.values, atu.values, rtol=1e-1, atol=1e-1), f"""TRI-040-D -
    Triangle.atu('medial', 5, tail=1.05) did not return the same atu with vwa as expected:
    atu: {atu}
    expected_atu: {expected_atu}"""
    
    

def test_diag(test_triangle, test_diagonal):
    t = test_triangle
    expected_diag = test_diagonal

    # get the diagonal from the triangle object
    diag = t.diag().values

    # make sure the diagonal is the same as the expected diagonal
    assert np.allclose(diag, expected_diag), f"""TRI-041-A -
    Triangle.diag did not return the same diagonal as expected:
    diag: {diag}
    expected_diag: {expected_diag}"""

def test_diag3(test_triangle, test_diagonal3):
    t = test_triangle
    expected_diag = test_diagonal3
    print(f"expected_diag: {expected_diag}")

    # get the diagonal from the triangle object
    diag = t.diag(calendar_year=3).values
    print(f"diag: {diag}")

    # make sure the diagonal is the same as the expected diagonal
    assert np.allclose(diag, expected_diag), f"""TRI-041-B -
    Triangle.diag(3) did not return the same diagonal as expected:
    diag: {diag}
    expected_diag: {expected_diag}"""

def test_ult(test_triangle, test_ultimate):
    t = test_triangle
    expected_ult = test_ultimate

    # get the ultimate from the triangle object (have already tested .diag()) 
    # and .atu() above)
    ult = t.ult('vwa', 'all').round(0).astype(float).values
    
    # make sure the diagonal is the same as the expected diagonal
    assert np.allclose(ult, expected_ult), f"""TRI-042 -
    Triangle.ult('vwa', 'all') did not return the same diagonal as expected:
    diag: {ult}
    expected_diag: {expected_ult}"""

def test_melt_triangle(test_triangle, test_melted):
    t = test_triangle
    expected_melted = test_melted

    melted = t.melt_triangle().astype(float).fillna(0)

    assert are_dfs_equal(melted, expected_melted), f"""TRI-044 -
    Triangle.melt_triangle did not return the same melted triangle as expected:
    melted: {melted}
    expected_melted: {expected_melted}"""

def test_create_design_matrix_levels(test_triangle, test_dm_ay_levels):
    t = test_triangle
    expected_df = test_dm_ay_levels
    print(f"expected_df: {expected_df}")
    print(f"expected_df.shape: {expected_df.shape}")

    # get the design matrix from the triangle object
    df = t.create_design_matrix_levels(t.X_id['accident_period'], z=4, s="accident_period")
    print(f"df: {df}")
    print(f"df.shape: {df.shape}")

    # make sure the design matrix is the same as the expected design matrix
    assert are_triangles_equal(df,expected_df), f"""TRI-041 -
    Triangle.create_design_matrix_levels did not return the same design matrix as expected:
    df: {df}
    expected_df: {expected_df}"""

def test_create_design_matrix_trends1_types(test_triangle, test_dm_ay_trends):
    t = test_triangle
    expected_df = test_dm_ay_trends
    print(f"expected_df: {expected_df}")
    print(f"expected_df.shape: {expected_df.shape}")

    # get the design matrix from the triangle object
    df = t.create_design_matrix_trends(t.X_id['accident_period'], z=4, s="accident_period")
    print(f"df: {df}")
    print(f"df.shape: {df.shape}")

    # make sure the dtypes of the columns are the same
    for i, c in enumerate(df.columns.tolist()):
        print(f"df[c].dtype: {df[c].dtype}")
        print(f"expected_df[c].dtype: {expected_df[c].dtype}")
        assert df[c].dtype == expected_df[c].dtype, f"""TRI-043-1-{i+1} -
        Triangle.create_design_matrix_trends did not return the same dtypes as expected:
        df[c].dtype: {df[c].dtype}
        expected_df[c].dtype: {expected_df[c].dtype}"""

def test_create_design_matrix_trends1_values(test_triangle, test_dm_ay_trends):
    t = test_triangle
    expected_df = test_dm_ay_trends

    # get the design matrix from the triangle object
    df = t.create_design_matrix_trends(t.X_id['accident_period'], z=4, s="accident_period")
    
    # Compare each column
    for i, col in enumerate(df.columns.tolist()):
        assert df[col].equals(expected_df[col]), f"""TRI-043-2-{i+1} -
        Difference found in column {col}:
        df: {df[col]}
        expected_df: {expected_df[col]}"""
    

def test_create_design_matrix_trends1(test_triangle, test_dm_ay_trends):
    t = test_triangle
    expected_df = test_dm_ay_trends
    print(f"expected_df: {expected_df}")
    print(f"expected_df.shape: {expected_df.shape}")

    # get the design matrix from the triangle object
    df = t.create_design_matrix_trends(t.X_id['accident_period'], z=4, s="accident_period")
    print(f"df: {df}")
    print(f"df.shape: {df.shape}")

    # make sure the dtypes of the columns are the same

    # make sure the design matrix is the same as the expected design matrix
    assert df.equals(expected_df), f"""TRI-043-A1 -
    Triangle.create_design_matrix_trends did not return the same design matrix as expected:
    df: {df}
    expected_df: {expected_df}"""

    assert are_dfs_equal(df,expected_df), f"""TRI-043-A2 -
    Triangle.create_design_matrix_trends did not return the same design matrix as expected:
    df: {df}
    expected_df: {expected_df}"""
    

def test_create_design_matrix_trends2(test_triangle, test_dm_ay_trends):
    t = test_triangle
    expected_df = test_dm_ay_trends[['accident_period_2001']]
    print(f"expected_df: {expected_df}")
    print(f"expected_df.shape: {expected_df.shape}")

    # get the design matrix from the triangle object
    df = t.create_design_matrix_trends(t.X_id['accident_period'], z=4, s="accident_period")[['accident_period_2001']]
    print(f"df: {df}")
    print(f"df.shape: {df.shape}")

    # make sure the design matrix is the same as the expected design matrix
    assert are_dfs_equal(df,expected_df), f"""TRI-043-B -
    Triangle.create_design_matrix_trends did not return the same design matrix as expected:
    df: {df}
    expected_df: {expected_df}"""
    
def test_create_design_matrix_trends3(test_triangle, test_dm_ay_trends):
    t = test_triangle
    expected_df = test_dm_ay_trends[['accident_period_2002']]
    print(f"expected_df: {expected_df}")
    print(f"expected_df.shape: {expected_df.shape}")

    # get the design matrix from the triangle object
    df = t.create_design_matrix_trends(t.X_id['accident_period'], z=4, s="accident_period")[['accident_period_2002']]
    print(f"df: {df}")
    print(f"df.shape: {df.shape}")

    # make sure the design matrix is the same as the expected design matrix
    assert are_dfs_equal(df,expected_df), f"""TRI-043-C -
    Triangle.create_design_matrix_trends did not return the same design matrix as expected:
    df: {df}
    expected_df: {expected_df}"""
    
def test_create_design_matrix_trends4(test_triangle, test_dm_ay_trends):
    t = test_triangle
    expected_df = test_dm_ay_trends[['accident_period_2003']]
    print(f"expected_df: {expected_df}")
    print(f"expected_df.shape: {expected_df.shape}")

    # get the design matrix from the triangle object
    df = t.create_design_matrix_trends(t.X_id['accident_period'], z=4, s="accident_period")[['accident_period_2003']]
    print(f"df: {df}")
    print(f"df.shape: {df.shape}")

    # make sure the design matrix is the same as the expected design matrix
    assert are_dfs_equal(df,expected_df), f"""TRI-043-D -
    Triangle.create_design_matrix_trends did not return the same design matrix as expected:
    df: {df}
    expected_df: {expected_df}"""
    
def test_base_design_matrix(test_triangle, test_base_dm):
    t = test_triangle
    expected_df = test_base_dm.fillna(0).round(0).astype(int)
    

    # get the design matrix from the triangle object
    df = t.base_design_matrix().fillna(0).round(0).astype(int)

    # drop calendar columns
    cols_to_drop = df.columns[df.columns.str.contains('cal')].tolist()
    df = df.drop(columns=cols_to_drop)

    # loop over the cells one by one
    for i, j in itertools.product(range(df.shape[0]), range(df.shape[1])):
        i2, j2 = i, j
        x = df.iloc[i2,j2]
        x_expected = expected_df.iloc[i2,j2]
        assert x == x_expected, f"""TRI-044-({i2}, {j2}) -
        The values at cell ({i}, {j}) are not the same:
        x: {x}
        x_expected: {x_expected}
        column_name: {df.columns[i]}
        row_name: {df.index[i]}
        x.dtype: {x.dtype}  
        x_expected.dtype: {x_expected.dtype}"""


    # loop over the columns, testing one-by-one
    for i, col in enumerate(df.columns.tolist()):
        if df[col].dtype != expected_df[col].dtype:
            print(f"df[{col}].dtype: {df[col].dtype}")
            print(f"expected_df[{col}].dtype: {expected_df[col].dtype}")
        assert df[col].dtype == expected_df[col].dtype, f"""TRI-044-{i+1}-A -
        Triangle.base_design_matrix did not return the same dtypes as expected:
        df[{col}].dtype: {df[col].dtype}
        expected_df[{col}].dtype: {expected_df[col].dtype}"""

        if not df[col].equals(expected_df[col]):
            print(f"df[{col}]: {df[col]}")
            print(f"expected_df[{col}]: {expected_df[col]}")
        assert df[col].equals(expected_df[col]), f"""TRI-044-{i+1}-B -
        Difference found in column {col}:
        df: {df[col]}
        expected_df: {expected_df[col]}"""

    if not are_dfs_equal(df,expected_df):
        print(f"expected_df: {pd.DataFrame(expected_df, columns=df.columns.tolist())}")
        print(f"df: {df}")
        print(f"expected_df.shape: {expected_df.shape}")
        
        print(f"df.shape: {df.shape}")

    # make sure the design matrix is the same as the expected design matrix
    assert are_dfs_equal(df, expected_df), f"""TRI-044 -
    Triangle.base_design_matrix did not return the same design matrix as expected:
    df: {df}
    expected_df: {expected_df}"""

def test_get_train_forecast_split(test_triangle, test_base_dm):
    t = test_triangle
    base_dm = test_base_dm.is_observed.astype(int)

    # get the split from the triangle object
    train_forecast_split = t.get_train_forecast_split().fillna(0).round(0).astype(int)

    # make sure the split is the same as the expected split
    assert are_dfs_equal(train_forecast_split, base_dm), f"""TRI-045 -
    Triangle.get_train_forecast_split did not return the same split as expected:
    train_forecast_split: {train_forecast_split}
    base_dm: {base_dm}"""

def test_get_X(test_triangle, test_base_dm):
    t = test_triangle
    base_dm = test_base_dm.fillna(0).drop(columns=['tri', 'accident_period', 'development_period'])
    base_dm['intercept'] = 1

    # get the X from the triangle object
    X = t.get_X().fillna(0)
    cols_to_drop = X.columns[X.columns.str.contains('cal')].tolist()
    X = X.drop(columns=cols_to_drop + ['intercept'])
    X['intercept'] = 1

    print(f"X: {X}")
    print(f"base_dm: {base_dm}")
    # make sure the X is the same as the expected X
    assert are_dfs_equal(X, base_dm), f"""TRI-046 -
    Triangle.get_X did not return the same X as expected:
    X: {X}
    base_dm: {base_dm}"""

def test_get_X_train(test_triangle, test_base_dm):
    t = test_triangle
    base_dm = test_base_dm.fillna(0).drop(columns=['tri', 'accident_period', 'development_period'])
    base_dm['intercept'] = 1
    base_dm = base_dm.loc[base_dm.is_observed.eq(1)]

    # get the X from the triangle object
    X = t.get_X().fillna(0)
    cols_to_drop = X.columns[X.columns.str.contains('cal')].tolist()
    X = X.drop(columns=cols_to_drop + ['intercept'])
    X['intercept'] = 1
    X = X.loc[X.is_observed.eq(1)]

    print(f"X: {X}")
    print(f"base_dm: {base_dm}")
    # make sure the X is the same as the expected X
    assert are_dfs_equal(X, base_dm), f"""TRI-047 -
    Triangle.get_X('train') did not return the same X as expected:
    X: {X}
    base_dm: {base_dm}"""


def test_get_X_test(test_triangle, test_base_dm):
    t = test_triangle
    base_dm = test_base_dm.fillna(0).drop(columns=['tri', 'accident_period', 'development_period'])
    base_dm['intercept'] = 1
    base_dm = base_dm.loc[base_dm.is_observed.eq(0)]

    # get the X from the triangle object
    X = t.get_X().fillna(0)
    cols_to_drop = X.columns[X.columns.str.contains('cal')].tolist()
    X = X.drop(columns=cols_to_drop + ['intercept'])
    X['intercept'] = 1
    X = X.loc[X.is_observed.eq(0)]

    print(f"X: {X}")
    print(f"base_dm: {base_dm}")
    # make sure the X is the same as the expected X
    assert are_dfs_equal(X, base_dm), f"""TRI-047 -
    Triangle.get_X('test') did not return the same X as expected:
    X: {X}
    base_dm: {base_dm}"""

def test_get_X_cal(test_triangle):
    t = test_triangle

    X = t.get_X_cal()
    try:
        X = X.drop(columns=['calendar_period'])
    except KeyError:
        pass

    X2 = t.create_design_matrix_trends(t.X_id['calendar_period'], z=4, s="calendar_period")
    try:
        X2 = X2.drop(columns=['calendar_period'])
    except KeyError:
        pass
    

    print(f"X: {X}")
    print(f"X2: {X2}")
    assert are_dfs_equal(X, X2), f"""TRI-048 -
    Triangle.get_X_cal did not return the same X as expected:
    X: {X}
    X2: {X2}"""

def test_get_X_cal_noUseCAL(test_triangle):
    t0 = test_triangle.df.copy()

    # remake the same triangle w/o using calendar periods
    t = Triangle.from_dataframe(t0, use_cal=False)
    X = t.get_X_cal()
    try:
        X = X.drop(columns=['calendar_period'])
    except KeyError:
        pass
    
    X2 = t.create_design_matrix_trends(t.X_id['calendar_period'], z=4, s="calendar_period")
    try:
        X2 = X2.drop(columns=['calendar_period'])
    except KeyError:
        pass

    print(f"X: {X}")
    print(f"X2: {X2}")
    assert are_dfs_equal(X, X2), f"""TRI-048 -
    Triangle.get_X_cal did not return the same X as expected:
    X: {X}
    X2: {X2}"""

def test_get_X_cal_train(test_triangle):
    t = test_triangle

    X = t.get_X_cal('train')
    try:
        X = X.drop(columns=['calendar_period'])
    except KeyError:
        pass

    X2 = t.create_design_matrix_trends(t.X_id['calendar_period'],
                                       z=4,
                                       s="calendar_period")
    try:
        X2 = X2.drop(columns=['calendar_period'])
    except KeyError:
        pass

    X2 = X2.loc[t.X_base.is_observed.eq(1)]
    

    print(f"X: {X}")
    print(f"X2: {X2}")
    assert are_dfs_equal(X, X2), f"""TRI-049 -
    Triangle.get_X_cal(train) did not return the same X as expected:
    X: {X}
    X2: {X2}"""

def test_get_X_cal_forecast(test_triangle):
    t = test_triangle

    X = t.get_X_cal('forecast')
    try:
        X = X.drop(columns=['calendar_period'])
    except KeyError:
        pass

    X2 = t.create_design_matrix_trends(t.X_id['calendar_period'],
                                       z=4,
                                       s="calendar_period")
    try:
        X2 = X2.drop(columns=['calendar_period'])
    except KeyError:
        pass

    X2 = X2.loc[t.X_base.is_observed.eq(0)]
    

    print(f"X: {X}")
    print(f"X2: {X2}")
    assert are_dfs_equal(X, X2), f"""TRI-050 -
    Triangle.get_X_cal(test) did not return the same X as expected:
    X: {X}
    X2: {X2}"""


def test_get_X_exposure(test_triangle):
    t = test_triangle

    X = t.get_X_exposure()
    X2 = pd.Series(np.ones(t.df.shape[0]),
                   index=t.df.index,
                   name='exposure')

    print(f"X: {X}")
    print(f"X2: {X2}")
    assert are_dfs_equal(X, X2), f"""TRI-051 -
    Triangle.get_X_exposure() did not return the same X as expected:
    X: {X}
    X2: {X2}"""

