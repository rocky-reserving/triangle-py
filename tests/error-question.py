# this is my triangle for testing
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

## this is my testing design matrix:
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

# this is a helper function to test if two dataframes are equal
def are_dfs_equal(df1:pd.DataFrame, df2:pd.DataFrame) -> bool:
    return np.allclose(df1.values,
                       df2.values,
                       rtol=1e-3,
                       atol=1e-3)

# this is the testing function that is failing:
def test_base_design_matrix(test_triangle, test_base_dm):
    t = test_triangle
    expected_df = test_base_dm
    

    # get the design matrix from the triangle object
    df = t.base_design_matrix()

    # drop calendar columns
    cols_to_drop = df.columns[df.columns.str.contains('cal')].tolist()
    df = df.drop(columns=cols_to_drop)

    


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
        print(f"expected_df: {expected_df}")
        print(f"df: {df}")
        print(f"expected_df.shape: {expected_df.shape}")
        
        print(f"df.shape: {df.shape}")

    # make sure the design matrix is the same as the expected design matrix
    assert are_dfs_equal(df,expected_df), f"""TRI-044 -
    Triangle.base_design_matrix did not return the same design matrix as expected:
    df: {df.values}
    expected_df: {expected_df.values}"""

    # this is the triangle class definition:
    @dataclass
class Triangle:
    """
    Create a `Triangle` object. The `Triangle` object is used to store and
    manipulate triangle data.

    Attributes:
    -----------
    id : str
        The type of triangle the object represents - paid loss, reported
        loss, etc.
    tri : pd.DataFrame, default=None
        The triangle data. Must be a pandas DataFrame with:
            1. The origin period set as the index.
            2. The development periods set as the column names.
            3. The values set as the values in the DataFrame.
        If any of these conditions are not met, the triangle data will
        be set to None.
    triangle : pd.DataFrame, default=None
        Alias for `tri`.
    acc : pd.Series, default=None
        The accident period labels. Default is None, in which case the
        accident period labels will be set to the index of the triangle
        data.
    dev : pd.Series, default=None
        The development period labels. Default is None, in which case the
        development period labels will be set to the column names of the
        triangle data.
    cal : pd.DataFrame, default=None
        The calendar period labels. Default is None, in which case the
        calendar period labels will be calculated from the acc and dev
        attributes.
    n_acc : int, default=None
        The number of accident periods in the triangle data. Default is
        None, in which case the number of accident periods will be calculated
        from `tri.shape[0]`.
    n_dev : int, default=None
        The number of development periods in the triangle data. Default is
        None, in which case the number of development periods will be
        calculated from `tri.shape[1]`.
    n_cal : int, default=None
        The number of calendar periods in the triangle data. Default is None,
        in which case the number of calendar periods will be calculated from
        the number of unique calendar periods in the `cal` attribute.
    acc_trends : bool, default=False
        Whether or not to model the accident period effects as trends. If
        True, linear trends will be used to model the accident periods. 
        Default is False, in which case the accident period effects will be
        modeled as levels unrelated to the previous accident period.
    dev_trends : bool, default=True
        Whether or not to model the development period effects as trends. See
        `acc_trends` for more information.
    cal_trends : bool, default=True
        Whether or not to model the calendar period effects as trends. See
        `acc_trends` for more information.
    n_vals : int, default=3
        The number of diagonals used for time-series validation. Default is 3,
        which corresponds to the 3 most recent diagonals. If `n_vals` is set
        to 0 or None, no time-series validation will be performed.
    incr_triangle : pd.DataFrame, default=None
        The incremental triangle data. Default is None, in which case the
        incremental triangle data will be calculated from the triangle data.
    X_base : pd.DataFrame, default=None
        The design matrix for the "base" model, eg a model with accident and
        development periods as features, and no calendar period effect.
        Default is None, in which case the design matrix will be calculated
        from the triangle data.
    y_base : np.ndarray, default=None
        The response vector for the "base" model.
    X_base_train : pd.DataFrame, default=None
        The design matrix for the "base" model.
    y_base_train : np.ndarray, default=None
        The response vector for the "base" model.
    X_base_forecast : pd.DataFrame, default=None
        The design matrix for the "base" model.
    y_base_forecast : np.ndarray, default=None
        The response vector for the "base" model.
    """

    id: str = None
    tri: pd.DataFrame = None
    tri0: pd.DataFrame = None
    triangle: pd.DataFrame = None
    tri_exposure: pd.Series = None
    acc: pd.Series = None
    dev: pd.Series = None
    cal: pd.Series = None
    n_acc: int = None
    n_dev: int = None
    n_cal: int = None
    acc_trends: bool = False
    dev_trends: bool = True
    cal_trends: bool = True
    use_cal: bool = True
    n_vals: int = 3
    incr_triangle: pd.DataFrame = None
    X_base: pd.DataFrame = None
    y_base: pd.Series = None
    exposure: pd.Series = None
    X_base_train: pd.DataFrame = None
    y_base_train: pd.Series = None
    exposure_train: pd.Series = None
    X_base_forecast: pd.DataFrame = None
    y_base_forecast: pd.Series = None
    exposure_forecast: pd.Series = None
    has_cum_model_file: bool = False
    is_cum_model: Any = None

    def __post_init__(self) -> None:
        """
        Reset triangle id if it is not allowed.

        Parameters:
        -----------
        None
        Returns:
        --------
        None
        """
        # if a triangle was passed in:
        if self.tri is not None:
            # convert the origin to a datetime object
            self.convert_origin_to_datetime()

            # set the acc, dev, cal attributes
            self.acc = self.tri.index.to_series().reset_index(drop=True)
            self.dev = self.tri.columns.to_series().reset_index(drop=True)
            self.acc.name = "accident_period"
            self.dev.name = "development_period"

            self.tri.index = self.acc
            self.tri.columns = self.dev
            self.tri.index.name = "accident_period"
            self.tri.columns.name = "development_period"
            self.triangle = self.tri

            self.ay = self.acc.dt.year.astype(int)
            self.ay.name = "accident_year"

            self.aq = self.acc.dt.quarter.astype(int)
            self.aq.name = "accident_quarter"

            self.am = self.acc.dt.month.astype(int)
            self.am.name = "accident_month"

            # set the n_rows and n_cols attributes
            self.n_rows = self.tri.shape[0]
            self.n_cols = self.tri.shape[1]
            self.n_dev = self.n_cols
            self.n_acc = self.n_rows

            # set frequency of triangle rows
            self._set_frequency()

            # set the cal attribute
            self.cal = self.getCalendarIndex()
            self.cur_cal = self.getCurCalendarIndex()

            # set the n_cal attribute
            self.n_cal = self.cal.max().max() - self.cal.min().min() + 1

            # convert triangle data to float
            for c in self.tri.columns:
                try:
                    self.tri[c] = self.tri[c].astype(float)
                except TypeError:
                    self.tri[c] = (
                        self.tri[c]
                        .str.replace(",", "")
                        .str.replace(")", "")
                        .str.replace("(", "-")
                        .astype(float)
                    )

            # set the incr_triangle attribute
            if self.incr_triangle is None:
                self.incr_triangle = self.cum_to_inc(_return=True)

        # create alias for self.tri as self.df that matches the triangle as
        # it is updated, and does not need to be updated separately
        self.df = self.tri
        self.base_design_matrix()
        self.positive_y = self.y_base.loc[self.y_base > 0].index.values
        self.is_observed = self.X_base.is_observed
        self.tri0 = self.tri.round(0)

        # if no exposure vector is passed, set all exposures to 1
        if self.exposure is None:
            self.exposure = pd.Series(1, index=self.acc.drop_duplicates())

# this is the base_design_matrix method of the triangle class:
def base_design_matrix(
        self,
        id_cols: list = None,
        var_name: str = "development_period",
        value_name: str = "tri",
        incr_tri: bool = True,
        return_: bool = False) -> pd.DataFrame:
        """
        Creates a design matrix from the triangle data. The design matrix is a pandas
        dataframe with one row for each triangle cell, and one column for each origin
        and development period. The origin and development periods are encoded as
        dummy variables, and if `trends` is True, the origin and development periods
        are also encoded as linear trends, instead of just dummy variables.

        This is the base design matrix for a rocky3 model. The base design matrix
        is used to create the full design matrix, which includes any interaction
        terms, and any other covariates.

        All diagnostics will implicitly check that any changes to the base model provide
        improvements to the base model fit.

        Parameters:
        -----------
        id_cols: list
            The columns to use as the id variables. Default is None, in which
            case the index is used.
        cols: list | str
            The columns to use in the design matrix. Accepts either a list
            of column names, or a string to use as a regex to match column
            names. Default is None, in which case all columns are used.
        var_name: str
            The name of the variable column. Default is 'dev'.
        value_name: str
            The name of the value column. Default is None, in which case
            the value column is set equal to the triangle ID.
        trends: bool
            If True, include linear trends in the design matrix. Default is True.
        _return: bool
            If True, return the design matrix as a pandas dataframe.
            Default is True.
        incr_tri: bool
            If True, use the incremental triangle data. Default is True. If
            False, use the cumulative triangle data.
        return_: bool
            If True, return the design matrix as a pandas dataframe.
            Default is False.

        Returns:
        --------
        dm_total: pd.DataFrame
            The design matrix.
        """
        if id_cols is None:
            id_cols = 'accident_period'

        if value_name is None:
            value_name = 'development_period'

        # melt the triangle data
        melted = self.melt_triangle(id_cols=id_cols,
                                    var_name=var_name,
                                    value_name=value_name,
                                    _return=True,
                                    incr_tri=incr_tri)
        
        # add calendar period:
        melted['calendar_period'] = (
            melted
            .apply(lambda x: int(x[0]) - 
                             melted['accident_period'].astype(int).min() + 
                             int(x[1]), axis=1))
        melted['is_observed'] = melted[value_name].notnull().astype(int)

        # create the design matrices for each column
        # accident period
        if self.acc_trends:
            acc = self.create_design_matrix_trends(melted['accident_period'],
                                                   s='accident_period',
                                                   z=4)
        else:
            acc = self.create_design_matrix_levels(melted['accident_period'],
                                                   s='accident_period',
                                                   z=4)
        
        # development period
        if self.dev_trends:
            dev = self.create_design_matrix_trends(melted['development_period'],
                                                   s='development_period',
                                                   z=3)
        else:
            dev = self.create_design_matrix_levels(melted['development_period'],
                                                   s='development_period',
                                                   z=3)

        # calendar period
        if self.cal_trends:
            cal = self.create_design_matrix_trends(melted['calendar_period'],
                                                   s='calendar_period',
                                                   z=3)
        else:
            cal = self.create_design_matrix_levels(melted['calendar_period'],
                                                   s='calendar_period',
                                                   z=3)

        # combine the design matrices
        dm_total = pd.concat(
            [melted[[value_name, 'is_observed']], acc, dev, cal],
            axis=1)

        if return_:
            return dm_total

        # sort the columns
        front_cols = [value_name,
                      'is_observed',
                      'accident_period',
                      'development_period',
                      'calendar_period']
        dm_total = dm_total[front_cols + list(dm_total.columns.drop(front_cols))]

        # drop calendar period variables if self.use_cal is False
        if self.use_cal:
            pass
        else:
            cal_columns = dm_total.columns[dm_total.columns.str.contains('cal')]
            calendar_period = dm_total['calendar_period']
            dm_total = dm_total.drop(columns=cal_columns.tolist())
            front_cols = [c for c in front_cols if 'cal' not in c and c != 'calendar_period']
        
        # assign class attributes with the design matrix and target variable
        self.X_base = dm_total.drop(columns=front_cols).astype(int)
        self.X_base['is_observed'] = dm_total['is_observed'].astype(int)
        self.X_base['intercept'] = 1
        self.X_base = self.X_base[['is_observed', 'intercept'] + self.X_base.columns.drop('is_observed').tolist()]
        self.y_base = dm_total[value_name]
        self.y_base.name = "y"

        # ay/dev id for each row
        if self.use_cal:
            self.X_id = dm_total[front_cols]
        else:
            self.X_id = pd.concat([dm_total[front_cols],
                                   calendar_period], axis=1)
        self.X_id.index = self.X_base.index

        # create the train/forecast data split based on the is_observed
        # column
        self.get_train_forecast_split(return_=False)

        return dm_total


# this is the error I recieve:
# ============================= test session starts ==============================
# platform linux -- Python 3.11.3, pytest-7.4.0, pluggy-1.2.0
# rootdir: /home/aweaver/git/Triangle/tests
# collected 48 items                                                             

# test_triangle.py ...............................................F        [100%]

# =================================== FAILURES ===================================
# ___________________________ test_base_design_matrix ____________________________

# test_triangle = development_period    12    24    36    48
# accident_period                           
# 2000-01-01          10.0  20.0  ...1          10.0  20.0  30.0   NaN
# 2002-01-01          10.0  20.0   NaN   NaN
# 2003-01-01          10.0   NaN   NaN   NaN
# test_base_dm =      tri  is_observed  ...  development_period_036  development_period_048
# 0   10.0            1  ...                 ...                   1
# 15   NaN            0  ...                       1                       1

# [16 rows x 10 columns]

#     def test_base_design_matrix(test_triangle, test_base_dm):
#         t = test_triangle
#         expected_df = test_base_dm
    
    
#         # get the design matrix from the triangle object
#         df = t.base_design_matrix()
    
#         # drop calendar columns
#         cols_to_drop = df.columns[df.columns.str.contains('cal')].tolist()
#         df = df.drop(columns=cols_to_drop)
    
    
    
    
#         # loop over the columns, testing one-by-one
#         for i, col in enumerate(df.columns.tolist()):
#             if df[col].dtype != expected_df[col].dtype:
#                 print(f"df[{col}].dtype: {df[col].dtype}")
#                 print(f"expected_df[{col}].dtype: {expected_df[col].dtype}")
#             assert df[col].dtype == expected_df[col].dtype, f"""TRI-044-{i+1}-A -
#             Triangle.base_design_matrix did not return the same dtypes as expected:
#             df[{col}].dtype: {df[col].dtype}
#             expected_df[{col}].dtype: {expected_df[col].dtype}"""
    
#             if not df[col].equals(expected_df[col]):
#                 print(f"df[{col}]: {df[col]}")
#                 print(f"expected_df[{col}]: {expected_df[col]}")
#             assert df[col].equals(expected_df[col]), f"""TRI-044-{i+1}-B -
#             Difference found in column {col}:
#             df: {df[col]}
#             expected_df: {expected_df[col]}"""
    
#         if not are_dfs_equal(df,expected_df):
#             print(f"expected_df: {expected_df}")
#             print(f"df: {df}")
#             print(f"expected_df.shape: {expected_df.shape}")
    
#             print(f"df.shape: {df.shape}")
    
#         # make sure the design matrix is the same as the expected design matrix
# >       assert are_dfs_equal(df,expected_df), f"""TRI-044 -
#         Triangle.base_design_matrix did not return the same design matrix as expected:
#         df: {df.values}
#         expected_df: {expected_df.values}"""
# E       AssertionError: TRI-044 -
# E             Triangle.base_design_matrix did not return the same design matrix as expected:
# E             df: [[1.000e+01 1.000e+00 2.000e+03 1.200e+01 0.000e+00 0.000e+00 0.000e+00
# E           0.000e+00 0.000e+00 0.000e+00]
# E          [1.000e+01 1.000e+00 2.001e+03 1.200e+01 1.000e+00 0.000e+00 0.000e+00
# E           0.000e+00 0.000e+00 0.000e+00]
# E          [1.000e+01 1.000e+00 2.002e+03 1.200e+01 0.000e+00 1.000e+00 0.000e+00
# E           0.000e+00 0.000e+00 0.000e+00]
# E          [1.000e+01 1.000e+00 2.003e+03 1.200e+01 0.000e+00 0.000e+00 1.000e+00
# E           0.000e+00 0.000e+00 0.000e+00]
# E          [1.000e+01 1.000e+00 2.000e+03 2.400e+01 0.000e+00 0.000e+00 0.000e+00
# E           1.000e+00 0.000e+00 0.000e+00]
# E          [1.000e+01 1.000e+00 2.001e+03 2.400e+01 1.000e+00 0.000e+00 0.000e+00
# E           1.000e+00 0.000e+00 0.000e+00]
# E          [1.000e+01 1.000e+00 2.002e+03 2.400e+01 0.000e+00 1.000e+00 0.000e+00
# E           1.000e+00 0.000e+00 0.000e+00]
# E          [      nan 0.000e+00 2.003e+03 2.400e+01 0.000e+00 0.000e+00 1.000e+00
# E           1.000e+00 0.000e+00 0.000e+00]
# E          [1.000e+01 1.000e+00 2.000e+03 3.600e+01 0.000e+00 0.000e+00 0.000e+00
# E           1.000e+00 1.000e+00 0.000e+00]
# E          [1.000e+01 1.000e+00 2.001e+03 3.600e+01 1.000e+00 0.000e+00 0.000e+00
# E           1.000e+00 1.000e+00 0.000e+00]
# E          [      nan 0.000e+00 2.002e+03 3.600e+01 0.000e+00 1.000e+00 0.000e+00
# E           1.000e+00 1.000e+00 0.000e+00]
# E          [      nan 0.000e+00 2.003e+03 3.600e+01 0.000e+00 0.000e+00 1.000e+00
# E           1.000e+00 1.000e+00 0.000e+00]
# E          [1.000e+01 1.000e+00 2.000e+03 4.800e+01 0.000e+00 0.000e+00 0.000e+00
# E           1.000e+00 1.000e+00 1.000e+00]
# E          [      nan 0.000e+00 2.001e+03 4.800e+01 1.000e+00 0.000e+00 0.000e+00
# E           1.000e+00 1.000e+00 1.000e+00]
# E          [      nan 0.000e+00 2.002e+03 4.800e+01 0.000e+00 1.000e+00 0.000e+00
# E           1.000e+00 1.000e+00 1.000e+00]
# E          [      nan 0.000e+00 2.003e+03 4.800e+01 0.000e+00 0.000e+00 1.000e+00
# E           1.000e+00 1.000e+00 1.000e+00]]
# E             expected_df: [[1.000e+01 1.000e+00 2.000e+03 1.200e+01 0.000e+00 0.000e+00 0.000e+00
# E           0.000e+00 0.000e+00 0.000e+00]
# E          [1.000e+01 1.000e+00 2.001e+03 1.200e+01 1.000e+00 0.000e+00 0.000e+00
# E           0.000e+00 0.000e+00 0.000e+00]
# E          [1.000e+01 1.000e+00 2.002e+03 1.200e+01 0.000e+00 1.000e+00 0.000e+00
# E           0.000e+00 0.000e+00 0.000e+00]
# E          [1.000e+01 1.000e+00 2.003e+03 1.200e+01 0.000e+00 0.000e+00 1.000e+00
# E           0.000e+00 0.000e+00 0.000e+00]
# E          [1.000e+01 1.000e+00 2.000e+03 2.400e+01 0.000e+00 0.000e+00 0.000e+00
# E           1.000e+00 0.000e+00 0.000e+00]
# E          [1.000e+01 1.000e+00 2.001e+03 2.400e+01 1.000e+00 0.000e+00 0.000e+00
# E           1.000e+00 0.000e+00 0.000e+00]
# E          [1.000e+01 1.000e+00 2.002e+03 2.400e+01 0.000e+00 1.000e+00 0.000e+00
# E           1.000e+00 0.000e+00 0.000e+00]
# E          [      nan 0.000e+00 2.003e+03 2.400e+01 0.000e+00 0.000e+00 1.000e+00
# E           1.000e+00 0.000e+00 0.000e+00]
# E          [1.000e+01 1.000e+00 2.000e+03 3.600e+01 0.000e+00 0.000e+00 0.000e+00
# E           1.000e+00 1.000e+00 0.000e+00]
# E          [1.000e+01 1.000e+00 2.001e+03 3.600e+01 1.000e+00 0.000e+00 0.000e+00
# E           1.000e+00 1.000e+00 0.000e+00]
# E          [      nan 0.000e+00 2.002e+03 3.600e+01 0.000e+00 1.000e+00 0.000e+00
# E           1.000e+00 1.000e+00 0.000e+00]
# E          [      nan 0.000e+00 2.003e+03 3.600e+01 0.000e+00 0.000e+00 1.000e+00
# E           1.000e+00 1.000e+00 0.000e+00]
# E          [1.000e+01 1.000e+00 2.000e+03 4.800e+01 0.000e+00 0.000e+00 0.000e+00
# E           1.000e+00 1.000e+00 1.000e+00]
# E          [      nan 0.000e+00 2.001e+03 4.800e+01 1.000e+00 0.000e+00 0.000e+00
# E           1.000e+00 1.000e+00 1.000e+00]
# E          [      nan 0.000e+00 2.002e+03 4.800e+01 0.000e+00 1.000e+00 0.000e+00
# E           1.000e+00 1.000e+00 1.000e+00]
# E          [      nan 0.000e+00 2.003e+03 4.800e+01 0.000e+00 0.000e+00 1.000e+00
# E           1.000e+00 1.000e+00 1.000e+00]]
# E       assert False
# E        +  where False = are_dfs_equal(     tri  is_observed  ...  development_period_036  development_period_048\n0   10.0            1  ...                 ...                   1\n15   NaN            0  ...                       1                       1\n\n[16 rows x 10 columns],      tri  is_observed  ...  development_period_036  development_period_048\n0   10.0            1  ...                 ...                   1\n15   NaN            0  ...                       1                       1\n\n[16 rows x 10 columns])

# test_triangle.py:1031: AssertionError
# ----------------------------- Captured stdout call -----------------------------
# expected_df:      tri  is_observed  ...  development_period_036  development_period_048
# 0   10.0            1  ...                       0                       0
# 1   10.0            1  ...                       0                       0
# 2   10.0            1  ...                       0                       0
# 3   10.0            1  ...                       0                       0
# 4   10.0            1  ...                       0                       0
# 5   10.0            1  ...                       0                       0
# 6   10.0            1  ...                       0                       0
# 7    NaN            0  ...                       0                       0
# 8   10.0            1  ...                       1                       0
# 9   10.0            1  ...                       1                       0
# 10   NaN            0  ...                       1                       0
# 11   NaN            0  ...                       1                       0
# 12  10.0            1  ...                       1                       1
# 13   NaN            0  ...                       1                       1
# 14   NaN            0  ...                       1                       1
# 15   NaN            0  ...                       1                       1

# [16 rows x 10 columns]
# df:      tri  is_observed  ...  development_period_036  development_period_048
# 0   10.0            1  ...                       0                       0
# 1   10.0            1  ...                       0                       0
# 2   10.0            1  ...                       0                       0
# 3   10.0            1  ...                       0                       0
# 4   10.0            1  ...                       0                       0
# 5   10.0            1  ...                       0                       0
# 6   10.0            1  ...                       0                       0
# 7    NaN            0  ...                       0                       0
# 8   10.0            1  ...                       1                       0
# 9   10.0            1  ...                       1                       0
# 10   NaN            0  ...                       1                       0
# 11   NaN            0  ...                       1                       0
# 12  10.0            1  ...                       1                       1
# 13   NaN            0  ...                       1                       1
# 14   NaN            0  ...                       1                       1
# 15   NaN            0  ...                       1                       1

# [16 rows x 10 columns]
# expected_df.shape: (16, 10)
# df.shape: (16, 10)
# =========================== short test summary info ============================
# FAILED test_triangle.py::test_base_design_matrix - AssertionError: TRI-044 -
# ========================= 1 failed, 47 passed in 7.59s =========================

