import pandas as pd
import numpy as np
import pytest
import sys

sys.path.append("../triangle")

from triangle import Triangle

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
def test_triangle2():
    """
    build test triangle 2
    """
    df = pd.DataFrame({
        '12':[10, 11, 12, 13],
        '24':[20, 30, 40, np.nan],
        '36':[30, 37, np.nan, np.nan],
        '48':[40, np.nan, np.nan, np.nan]
    }, index=[1990, 1991, 1992, 1993])
    return Triangle.from_dataframe(df=df, id="t")

test_parameters = [test_triangle]

expected_atas = {
    'vw-all': pd.Series([2, 1.5, 1.3, 1]),

    'med-5-ex-hlm-tail105': pd.Series([2, 1.5, 1.3, 1.05]),
}

def are_triangles_equal(tri_df1:pd.DataFrame, tri_df2:pd.DataFrame) -> bool:
    """
    Check if the values in two triangles are equal, ignoring NaNs
    """
    return np.allclose(tri_df1.fillna(0).values,
                       tri_df2.fillna(0).values,
                       rtol=1e-3,
                       atol=1e-3)

def cumprod(s:pd.Series) -> pd.Series:
    

  idx = s.index.tolist()
  idx.reverse()

  ata = s
  ata.index = idx
  ata = np.log(ata)
  ata.sort_index(inplace=True)
  ata = ata.cumsum()


  ata.index = idx
  ata = ata.sort_index()
  ata = np.exp(ata)
  return(ata)

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

def test_acc_init(test_triangle):
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
    assert (t.tri.index.tolist() == [pd.Timestamp('2000-01-01 00:00:00'),
                                    pd.Timestamp('2001-01-01 00:00:00'),
                                    pd.Timestamp('2002-01-01 00:00:00'),
                                    pd.Timestamp('2003-01-01 00:00:00')]), """
TRI-012 - triangle index (datetimes) is not correct"""
    assert t.n_acc == 4, "TRI-013 - Triangle.n_acc is not 4"

def test_dev_init(test_triangle):
    # t = triangle_()
    t = test_triangle
    # did the developmment years get read in correctly?
    assert t.n_dev == 4, "TRI-014-A - Triangle.n_dev is not 4"
    assert t.dev.tolist() == ['12', '24', '36', '48'], f"""TRI-014-B - 
    Triangle.dev is not correct:
    {t.dev.tolist()} instead of [12, 24, 36, 48]"""
    assert t.dev.name == "development_period", f"""TRI-014-C -
    Triangle.dev name is not 'development_period': {t.dev.name}"""

def test_cal_init(test_triangle):
    # t = triangle_()
    t = test_triangle
    # did the calendar years get read in correctly?
    assert t.n_cal == 7, f"TRI-016 - Triangle.n_cal is not 4: {t.n_cal}"

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
    filename = "../triangle/data/mack1994.csv"
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
    filename = "../triangle/data/dahms reserve triangles.xlsx"
    t = Triangle.from_excel(filename,
                            sheet_name="rpt",
                            origin_columns=1,
                            id="rpt",
                            sheet_range="a1:k11")
    t.tri = t.tri.astype(float)
    print(f"t.tri: {t.tri}")

    # read in triangle from excel using pandas
    t2 = (pd.read_excel(filename, sheet_name="rpt")
          .iloc[:10, :11]
          .set_index('ay')
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
    t2 = Triangle.from_csv("../triangle/data/mack1994.csv", id="paid_loss")

    # make sure the two triangles are equal
    assert are_triangles_equal(t.tri, t2.tri), f"""TRI-021 -
    Triangle.from_mack_1994 did not read in the same triangle as Triangle.from_csv:
    t.tri: {t.tri}
    t2: {t2}"""

def test_from_taylor_ashe():
    # read in triangle from method
    t = Triangle.from_taylor_ashe()

    # read in triangle from csv (already tested .from_csv above)
    t2 = Triangle.from_csv("../triangle/data/taylorashe.csv", id="paid_loss")

    # make sure the two triangles are equal
    assert are_triangles_equal(t.tri, t2.tri), f"""TRI-022 -
    Triangle.from_taylor_ashe did not read in the same triangle as Triangle.from_csv:
    t.tri: {t.tri}
    t2: {t2}"""

def test_from_dahms_rpt():
    # read in triangle from method
    t, _ = Triangle.from_dahms()

    # read in triangle from excel (already tested .from_excel above)
    t2 = Triangle.from_excel("../triangle/data/dahms reserve triangles.xlsx",
                             sheet_name="rpt",
                             origin_columns=1,
                             id="rpt",
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
    t2 = Triangle.from_excel("../triangle/data/dahms reserve triangles.xlsx",
                             sheet_name="paid",
                             origin_columns=1,
                             id="paid",
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

def test_getCalendarIndex(test_triangle):
    # t = triangle_()
    t = test_triangle
 
    expected_calendar_index = np.array([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
    ])

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
    expected_cur_calendar_index = 4

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
    

def test_cum_to_inc(test_triangle):
    # t = triangle_()
    t = test_triangle
    expected_inc = pd.DataFrame(np.array([
        [10, 10, 10, 10],
        [10, 10, 10, np.nan],
        [10, 10, np.nan, np.nan],
        [10, np.nan, np.nan, np.nan],
    ]), index=[2000, 2001, 2002, 2003], columns=[12, 24, 36, 48])

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

def test_ata_tri(test_triangle):
    # t = triangle_()
    t = test_triangle
    expected_ata = pd.DataFrame(np.array([
        [2, 1.5, 1.3, np.nan],
        [2, 1.5, np.nan, np.nan],
        [2, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan]
    ]), index=[2000, 2001, 2002, 2003], columns=[12, 24, 36, 48])

    # get the ata triangle from the triangle object - round to 1 decimal place
    ata = t._ata_tri().round(1)

    # make sure the ata triangle is the same as the expected ata triangle
    assert are_triangles_equal(ata, expected_ata), f"""TRI-030 -
    Triangle.ata_tri did not return the same ata triangle as expected:
    ata: {ata}
    expected_ata: {expected_ata}"""

def test_vwa(test_triangle):
    # t = triangle_()
    t = test_triangle
    expected_vwa_all = pd.Series([2, 1.5, 1.3, 1.0])
    expected_vwa_4 = pd.Series([2, 1.5, 1.3, 1.0])
    expected_vwa_2_tail = pd.Series([2, 1.5, 1.3, 1.05]).round(1)

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

def test_simple_ave_all(test_triangle):
    t = test_triangle
    expected_simple_ave_all = pd.Series([2, 1.5, 1.3, 1.0])

    # get the simple averages from the triangle object
    simple_ave_all = t._ave_ata('all').round(1).reset_index(drop=True)

    # make sure the simple averages are the same as the expected simple averages
    assert simple_ave_all.equals(expected_simple_ave_all), f"""TRI-034 -
    Triangle._ave_ata did not return the same simple averages as expected:
    simple_ave_all: {simple_ave_all}
    expected_simple_ave_all: {expected_simple_ave_all}"""

def test_simple_ave_3(test_triangle):
    t = test_triangle
    expected_simple_ave_3 = pd.Series([2, 1.5, 1.3, 1.0])

    # get the simple averages from the triangle object
    simple_ave_3 = t._ave_ata(3).round(1).reset_index(drop=True)

    # make sure the simple averages are the same as the expected simple averages
    assert simple_ave_3.equals(expected_simple_ave_3), f"""TRI-035 -
    Triangle._ave_ata did not return the same simple averages as expected:
    simple_ave_3: {simple_ave_3}
    expected_simple_ave_3: {expected_simple_ave_3}"""

def test_simple_ave_2_tail(test_triangle):
    t = test_triangle
    expected_simple_ave_2_tail = pd.Series([2, 1.5, 1.3, 1.05]).round(1)

    # get the simple averages from the triangle object
    simple_ave_2_tail = t._ave_ata(2, tail=1.05).round(1).reset_index(drop=True)

    # make sure the simple averages are the same as the expected simple averages
    assert simple_ave_2_tail.equals(expected_simple_ave_2_tail), f"""TRI-036 -
    Triangle._ave_ata did not return the same simple averages as expected:
    simple_ave_2_tail: {simple_ave_2_tail}
    expected_simple_ave_2_tail: {expected_simple_ave_2_tail}"""

def test_medial_all(test_triangle):
    t = test_triangle
    expected_medial_all = pd.Series([2, 1.5, 1.3, 1.0])

    # get the medial averages from the triangle object
    medial_all = t._medial_ata('all').round(1).reset_index(drop=True)

    # make sure the medial averages are the same as the expected medial averages
    assert medial_all.equals(expected_medial_all), f"""TRI-037 -
    Triangle._medial_ata did not return the same medial averages as expected:
    medial_all: {medial_all}
    expected_medial_all: {expected_medial_all}"""

def test_medial_all_exhigh(test_triangle):
    t = test_triangle
    expected_medial_all_exhigh = pd.Series([2, 1.5, 1.3, 1.0])

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

def test_medial_5_ex_high_low_mid_tail105(test_triangle):
    t = test_triangle
    expected_medial_5_ex_high_low_mid_tail105 = (expected_atas['med-5-ex-hlm-tail105']
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

def test_atu_vwa_all(test_triangle):
    t = test_triangle

    # expected atu with vwa
    ata = expected_atas['vw-all']
    expected_atu = ata[::-1].cumprod()[::-1].round(1).reset_index(drop=True)
    print(f"expected_atu: {expected_atu}")

    # get the atu with vwa from the triangle object
    atu = t.atu('vwa', 'all').round(1).reset_index(drop=True)

    # make sure the atu with vwa are the same as the expected atu with vwa
    assert np.allclose(expected_atu.values, atu.values, rtol=1e-1, atol=1e-1), f"""TRI-040-A -
    Triangle._atu_vwa did not return the same atu with vwa as expected:
    atu: {atu}
    expected_atu: {expected_atu}""" 