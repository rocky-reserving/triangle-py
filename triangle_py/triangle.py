"""
This module implements the Triangle class, which is used to store and
manipulate triangle data.

This class also includes methods for perfoming basic loss triangle analysis
using the chain ladder method.
"""

import json
import os

# import torch
# from torch.utils.data import DataLoader
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from openpyxl.utils import range_to_tuple

triangle_type_aliases = ["paid", "reported", "case", "incurred"]


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

    def __repr__(self) -> str:
        return self.tri.__repr__()

    def __str__(self) -> str:
        return self.tri.__str__()

    def _handle_missing_id(self):
        """
        Handle a missing id by assigning a random id.
        """
        # If the id is None, assign a random id
        if self.id is None:
            # create random triangle id
            self.id = f"triangle_{np.random.randint(10000, 99999)}"

    def set_id(self, id: str) -> None:
        """
        Set the id of the triangle.
        Parameters:
        -----------
        `id`: `str`
            The id of the triangle.
        Returns:
        --------
        `None`
        """
        if id is None:
            self._handle_missing_id()
        else:
            # ensure that the id is a string
            if not isinstance(id, str):
                raise TypeError("The id must be a string.")

            # reformat the id
            self.id = self.id.lower().replace(" ", "_")

            # reset the id if it is not allowed
            if self.id not in triangle_type_aliases:
                self.id = None
                self._handle_missing_id()

            # ensure the id is allowed
            if id.lower().replace(" ", "_") in triangle_type_aliases:
                self.id = id.lower().replace(" ", "_")
            else:
                print(f"""The id {id} is not allowed.
                It must be one of the following:""")
                for alias in triangle_type_aliases:
                    print(f"  - {alias}")
                print()
                raise ValueError("The id is not allowed.")

    def convert_origin_to_datetime(self) -> None:
        """
        Convert the origin to a datetime object.
        Test the origin column(s):
        1. If the origin column(s) are a single column:
            1. Is the column either integers or a column that can be
               converted to integers? (eg. a string column with
               integers in it)
                    1. Can the values be converted to 4-digit
                       integers? If so, convert the YEARS to datetime
                       objects, assuming the month is January and the
                       day is the first day of the month.
                    2. Can the values be converted to 2-digit
                       integers? If so, convert the 2-digit YEARS to
                       datetime objects, assuming the month is January
                       and the day is the first day of the month.
                    3. Can the values be converted to 5-digit
                       integers? If so, convert the 5-digit Year-Quarter
                       values to datetime objects, assuming the quarter
                       is the 5th digit, the year is the first 4 digits,
                       and the month is the first month of the quarter,
                       first day of the month.
                    4. Can the values be converted to 6-digit
                       integers? If so, convert the 6-digit Year-Month
                       values to datetime objects, assuming the month
                       is the last 2 digits, the year is the first 4
                       digits, and the day is the first day of the
                       month.
            2. Is the column a string that cannot be converted to
               integers? (eg. a string column with non-integer values
               such as 2023Q1 or 2023-01, etc)
                1. does the string contain a dash (-) but does not
                   contain a Q or a q? If so, assume that
                   the dash is separating the year from the month, but
                   do not assume either the year or the month is first.
                   For example, 2023-01 and 01-2023 are both valid. Find
                   the dash, split the string on the dash, and convert
                   both pieces on either side of the dash to integers.
                   If both pieces can be converted to integers, then
                   the month will be the piece that is between 1 and 12,
                   and the year will be the other one. If both the year
                   and the month are between 1 and 12, then raise an
                   error telling the user that the origin column is
                   ambiguous and print out a message telling the user
                   why the origin column is ambiguous.
                   Examples:
                   --------
                     1. 2023-01 -> year = 2023, month = 1, day = 1
                     2. 01-2023 -> year = 2023, month = 1, day = 1
                     3. 2023-13 -> Raise an error because neither the
                                   year nor the month are between 1
                                   and 12.
                     4. 3-23    -> year = 2023, month = 3, day = 1
                     5. 23-10   -> year = 2023, month = 10, day = 1
                     6. 2001-1  -> year = 2001, month = 1, day = 1
                     7. 1-2001  -> year = 2001, month = 1, day = 1
                     8. 10-01   -> Raise an error because the origin
                                   column is ambiguous.
                2. does the string contain a slash (/)? If so, follow the
                   same steps above, and try to convert both pieces to
                   integers.
                3. does the string contain a Q or a q but no dash? If so,
                   assume the q is separating the year from the quarter,
                   but do not assume either the year or the quarter is
                   first. For example, 2023Q1 and Q1-2023 are both valid.
                   Find the Q or q, split the string on the Q or q, and
                   convert both pieces on either side of the Q or q to
                   integers. Find the integer that is between 1 and 4,
                   and assume that is the quarter. Find the other integer
                   that is not between 1 and 4, and assume that is the
                   year. If both the year and the quarter are between 1
                   and 4, then raise an error telling the user that the
                   origin column is ambiguous and print out a message
                   telling the user why the origin column is ambiguous.
                     Examples:
                     --------
                     1. 2023Q1 -> year = 2023, quarter = 1, month = 1, day = 1
                     2. 3q2023 -> year = 2023, quarter = 3, month = 7, day = 1
                     3. 22q4   -> year = 2022, quarter = 4, month = 10, day = 1
                     4. 2023q5 -> Raise an error because neither the year
                                  nor the quarter are between 1 and 4.
                     5. 2023q0 -> Raise an error because neither the year
                                  nor the quarter are between 1 and 4.
                     6. 2q2002 -> year = 2002, quarter = 2, month = 4, day =1
                     7. 2q02   -> Raise an error because the origin column
                                  is ambiguous.
                4. does the string contain a Q or a q and a dash? If so,
                   assume the dash is separating the year from the quarter,
                   and that the Q or q is closer to the quarter than to the
                   year.

                Examples:
                --------
                1. 2023-Q1 -> year = 2023, quarter = 1, month = 1, day = 1
                2. 2021-2q -> year = 2021, quarter = 1, month = 1, day = 1
                3. 3q-03   -> year = 2003, quarter = 3, month = 7, day = 1
                4. q4-04   -> year = 2004, quarter = 4, month = 10, day = 1
        Parameters:
        -----------
        None
        Returns:
        --------
        None
        """
        import re

        import pandas as pd

        # Function to convert a given year, month, and day to a datetime object
        def convert_to_datetime(year, month, day=1):
            # Using pandas to_datetime function to create a datetime object
            # with the provided year, month, and day
            return pd.to_datetime(f"{year}-{month}-{day}", format="%Y-%m-%d")

        # Function to convert origin values to datetime objects
        def convert_origin_to_datetime(origin):
            # default values -- these will be overwritten if the origin column
            # can be converted to integers
            year = None
            month = 1

            try:
                # if the origin value is already datetime, extract year, month, day
                if isinstance(origin, pd.Timestamp):
                    return convert_to_datetime(origin.year,
                                               origin.month,
                                               origin.day)
                else:

                    # Attempt to convert the origin value to an integer
                    value = int(origin)

                    # Check if the value is a 4-digit integer
                    if 1000 <= value <= 9999:
                        # Convert the 4-digit integer to a datetime object
                        # assuming January as the month and the first day of the
                        # month
                        return convert_to_datetime(value, 1)
                    # Check if the value is a 2-digit integer
                    elif 0 <= value <= 99:
                        # Convert the 2-digit integer to a datetime object
                        # assuming January as the month and the first day of the
                        # month
                        return convert_to_datetime(value + 2000, 1)
                    # Check if the value is a 5-digit integer
                    elif 10000 <= value <= 99999:
                        # Split the value into a 4-digit year and 1-digit quarter
                        year, quarter = divmod(value, 10)

                        # Convert the year-quarter value to a datetime object
                        # assuming the quarter is the last digit, the year is
                        # the first 4 digits, and the month is the first month
                        # of the quarter
                        return convert_to_datetime(year, quarter * 3 - 2)
                    # Check if the value is a 6-digit integer
                    elif 100000 <= value <= 999999:
                        # Split the value into a 4-digit year and 2-digit month
                        year, month = divmod(value, 100)

                        # Convert the year-month value to a datetime object
                        # assuming the month is the last 2 digits, the year is
                        # the first 4 digits, and the day is the first day of
                        # the month
                        return convert_to_datetime(year, month)
            except ValueError:
                # If the origin value cannot be converted to an integer,
                # continue
                # to the string processing section below
                print(f"Could not convert {origin} to an integer.")
                pass

            # Function to extract year, month, and quarter from a string
            # using a specified delimiter
            def get_year_month_quarter(s, delimiter):
                # Split the input string using the provided delimiter
                values = s.split(delimiter)

                # Attempt to convert both parts of the string to integers
                try:
                    a, b = int(values[0]), int(values[1])
                except ValueError:
                    # If either part of the string cannot be converted to
                    # an integer, return None values
                    return None, None, None

                # Check if the first part is a month and the second part is a
                # year
                if 1 <= a <= 12 and 1000 <= b <= 9999:
                    return b, a, None
                # Check if the first part is a year and the second part is a
                # month
                elif 1000 <= a <= 9999 and 1 <= b <= 12:
                    return a, b, None
                # Check if the first part is a quarter and the second part is
                # a year
                elif 1 <= a <= 4 and 1000 <= b <= 9999:
                    return b, None, a
                # Check if the first part is a year and the second part is a
                # quarter
                elif 1000 <= a <= 9999 and 1 <= b <= 4:
                    return a, None, b

                # If neither of the above conditions are met, raise a
                # ValueError indicating that the origin column is ambiguous
                raise ValueError("Ambiguous origin column")

            # Determine which type of string the origin is and convert
            # it accordingly
            if "-" in origin and "Q" not in origin.upper():
                # If the origin contains a dash and no 'Q', assume the dash
                # separates the year and month
                year, month, _ = get_year_month_quarter(origin, "-")
            elif "/" in origin:
                # If the origin contains a slash, assume it separates the year
                # and month
                year, month, _ = get_year_month_quarter(origin, "/")
            elif "Q" in origin.upper() and "-" not in origin:
                # If the origin contains 'Q' and no dash, assume the 'Q'
                # separates the year and quarter
                origin = re.sub("[-qQ]", "", origin)
                year, _, quarter = get_year_month_quarter(origin, "Q")
                if quarter:
                    month = quarter * 3 - 2
            elif "Q" in origin.upper() and "-" in origin:
                # If the origin contains 'Q' and a dash, assume the dash
                # separates the year and quarter
                origin = origin.upper().replace("Q", "")
                year, _, quarter = get_year_month_quarter(origin, "-")
                if quarter:
                    month = quarter * 3 - 2

            # If the year value is None, raise a ValueError indicating that
            # the origin column is invalid
            if year is None:
                raise ValueError(f"Invalid origin: {origin}")

            # If both year and month values are found, convert them to a
            # datetime object
            if year and month:
                return convert_to_datetime(year, month)

            # If none of the above conditions are met, raise a ValueError
            # indicating that the origin column is invalid
            raise ValueError(f"Invalid origin column: {origin}")

        # Map the convert_origin_to_datetime function to each value in the
        # index and replace the original index with the new datetime index
        self.tri.index = self.tri.index.map(convert_origin_to_datetime)

    def _set_frequency(self) -> None:
        """
        Sets the .frequency attribute by checking if the origin column is in
        monthly, quarterly, or annual frequency:
            1. If all of the origin values have a month of 1, the origin
            column is assumed to be in annual frequency.
            2. Elif all of the origin values have a month of 1, 4, 7, or 10,
            the origin column is assumed to be in quarterly frequency.
            3. Else, the origin column is assumed to be in monthly frequency.
        """

        # Check if the index is a datetime index
        if not isinstance(self.tri.index, pd.DatetimeIndex):
            # If not, call the convert_year_to_datetime method
            self.convert_year_to_datetime()

        # Check for annual frequency
        if all(value.month == 1 for value in self.tri.index):
            self.frequency = "A"
        # Check for quarterly frequency
        elif all(value.month in [1, 4, 7, 10] for value in self.tri.index):
            self.frequency = "Q"
        # Assume monthly frequency
        else:
            self.frequency = "M"

    def get_formatted_dataframe(self, df=None) -> pd.DataFrame:
        """
        Returns a new DataFrame with the index formatted as strings based on the
        frequency attribute:
            - Annual frequencies are formatted as YYYY.
            - Quarterly frequencies are formatted as YYYYQ#.
            - Monthly frequencies are formatted as YYYY-MM.
        Parameters:
        -----------
        df : pd.DataFrame
            If provided, the `df` DataFrame will be formatted and returned.
            If not provided, the `Triangle.tri` DataFrame will be formatted
            and returned.
            Default is None, which will format the `Triangle.tri` DataFrame.
        """
        if df is None:
            # Make a copy of the DataFrame to avoid modifying the original
            formatted_df = self.tri.copy()
        else:
            # Make a copy of the DataFrame to avoid modifying the original
            formatted_df = df.copy()

        # Format the index based on the frequency attribute
        if self.frequency == "A":
            formatted_df.index = self.tri.index.strftime("%Y")
        elif self.frequency == "Q":
            formatted_df.index = (self.tri
                                  .index
                                  .to_period("Q")
                                  .strftime("%YQ%q"))
        else:
            formatted_df.index = self.tri.index.strftime("%Y-%m")

        return formatted_df

    def to_json(self):
        """
        Converts the triangle object to json, to prepare for an API call
        """
        
        # start from a dictionary
        out_dict = {
            "id": self.id,
            "XBase": self.X_base.to_dict()
            if self.X_base is not None
            else None,
            "yBase": self.y_base.tolist()
            if self.y_base is not None
            else None,
            "XBaseTrain": self.X_base_train.to_dict()
            if self.X_base_train is not None
            else None,
            "yBaseTrain": self.y_base_train.tolist()
            if self.y_base_train is not None
            else None,
            "XBaseForecast": self.X_base_forecast.to_dict()
            if self.X_base_forecast is not None
            else None,
            "yBaseForecast": self.y_base_forecast.tolist()
            if self.y_base_forecast is not None
            else None,
            "hasCumModelFile": self.has_cum_model_file
            if self.has_cum_model_file is not None
            else None,
            "isCumModel": self.is_cum_model
            if self.is_cum_model is not None
            else None,
            "nCols": self.n_cols
            if self.n_cols is not None
            else None,
            "nRows": self.n_rows
            if self.n_rows is not None
            else None,
        }

        # convert datetime index to string
        if self.tri is not None:
            temp_tri = self.tri.copy()
            temp_tri.index = temp_tri.index.strftime("%Y-%m")
            out_dict["tri"] = (temp_tri.to_dict()
                               if self.tri is not None
                               else None)
        else:
            out_dict["tri"] = None

        if self.triangle is not None:
            temp_triangle = self.triangle.copy()
            temp_triangle.index = temp_triangle.index.strftime("%Y-%m")
            out_dict["triangle"] = temp_triangle.to_dict()
        else:
            out_dict["triangle"] = None

        if self.incr_triangle is not None:
            temp_incr_triangle = self.incr_triangle.copy()
            temp_incr_triangle.index = (temp_incr_triangle.index
                                        .strftime("%Y-%m"))
            out_dict["incrTriangle"] = temp_incr_triangle.to_dict()
        else:
            out_dict["incrTriangle"] = None

        # convert to json
        out_json = json.dumps(out_dict)

        return out_json

    @classmethod
    def from_dataframe(cls,
                       df: pd.DataFrame,
                       id: Optional[str] = None,
                       use_cal:bool = True) -> "Triangle":
        """
        Create a Triangle object from a pandas DataFrame.

        Parameters:
        -----------
        id : str
            The id of the triangle.
        df : pd.DataFrame
            The triangle data. Must be a pandas DataFrame with:
                1. The origin period set as the index.
                2. The development periods set as the column names.
                3. The values set as the values in the DataFrame.
            If any of these conditions are not met, the triangle data will
            be set to None.
        use_cal : bool
            Whether or not to use calendar period effects in the linear
            model representation. Default is True.

        Returns:
        --------
        Triangle
            A Triangle object with data loaded from the DataFrame.
        """
        # Create and return a Triangle object
        return cls(id=id, tri=df, triangle=df, use_cal=use_cal)

    @classmethod
    def from_clipboard(cls,
                       origin_columns: int = 1,
                       id: Optional[str] = None,
                       use_cal:bool = True) -> "Triangle":
        """
        Create a Triangle object from data copied to the clipboard.

        Parameters:
        -----------
        id : str
            The id of the triangle. Default is None, which will assign a
            randomly-generated ID.
        origin_columns : int
            The number of columns used for the origin period. Default is 1.
        use_cal : bool
            Whether or not to use calendar period effects in the linear
            model representation. Default is True.

        Returns:
        --------
        Triangle
            A Triangle object with data loaded from the clipboard.
        """
        # Read data from the clipboard, assuming the first row is the
        # development period and the first `origin_columns` columns should
        # make up either an index or a multi-index for the origin period
        # in the resulting DataFrame
        df = pd.read_clipboard(header=None)

        # set the first row to be the headers
        df.columns = df.iloc[0]

        # since it is included as the column names, drop the first row
        df.drop(index=0, inplace=True)

        # set the first `origin_columns` columns to be the index
        or_col = df.columns.tolist()[:origin_columns]
        df.set_index(or_col, inplace=True)

        # convert remaining columns to numeric
        for c in df.columns.tolist():
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(",", "")
                .str.replace(" ", "")
                .astype(float)
            )

        # make sure the index is numeric/integer
        df.index = df.index.astype(str).astype(int)

        # do the same for the columns
        df.columns = df.columns.astype(str).astype(float).astype(int)

        # Create and return a Triangle object
        return cls(id=id, tri=df, triangle=df, use_cal=use_cal)

    @classmethod
    def from_csv(cls,
                 filename: str,
                 origin_columns: int = 1,
                 id: Optional[str] = None,
                 use_cal:bool = True) -> "Triangle":
        """
        Create a Triangle object from data in a CSV file.

        Parameters:
        -----------
        filename : str
            The name of the CSV file containing the triangle data.
        id : str, optional
            The id of the triangle.
        origin_columns : int
            The number of columns used for the origin period. Default is 1.
        use_cal : bool
            Whether or not to use calendar period effects in the linear
            model representation. Default is True.

        Returns:
        --------
        Triangle
            A Triangle object with data loaded from the CSV file.
        """
        # Read data from the CSV file
        df = pd.read_csv(filename,
                         header=0,
                         index_col=[i for i in range(origin_columns)])

        # Create and return a Triangle object
        return cls(id=id, tri=df, triangle=df, use_cal=use_cal)

    @classmethod
    def from_excel(cls,
                   filename: str,
                   origin_columns: int,
                   id: Optional[str] = None,
                   use_cal: bool = True,
                   sheet_name: Optional[str] = None,
                   sheet_range: Optional[str] = None) -> "Triangle":
        """
        Create a Triangle object from data in an Excel file.
        Parameters:
        -----------
        filename : str
            The name of the Excel file containing the triangle data.
        id : str
            The id of the triangle.
        origin_columns : int
            The number of columns used for the origin period.
        sheet_name : str, optional
            The name of the sheet in the Excel file containing the triangle
            data. If not provided, the first sheet will be used.
        sheet_range : str, optional
            A string containing the range of cells to read from the Excel
            file. The range should be in the format "A1:B2".
        use_cal : bool
            Whether or not to use calendar period effects in the linear
            model representation. Default is True.

        Returns:
        --------
        Triangle
            A Triangle object with data loaded from the Excel file.
        """
        # Read data from the Excel file
        if sheet_range:
            # If a range is provided, read only the specified range
            _, idx = range_to_tuple(f"'{sheet_name}'!{sheet_range}")
            c1, r1, c2, r2 = idx

            # read in the subset of the excel file
            df = (pd.read_excel(filename,
                                header=None,
                                sheet_name=sheet_name)
                  .iloc[(r1 - 1):(r2), (c1 - 1):(c2)])

            # set the column names as the first row
            df.columns = df.iloc[0]
            df.columns.name = None

            # # drop the first row
            df.drop(df.index[0], inplace=True)
        else:
            # If no range is provided, read the entire sheet
            df = pd.read_excel(filename, header=0, sheet_name=sheet_name)

        # Set the origin period as the index
        df.set_index(df.columns.tolist()[:origin_columns], inplace=True)

        # round to a single digit
        df = df.round(1)

        # re-sort the columns
        df.sort_index(axis=1, inplace=True)

        # cast the columns to floats then integers
        df.columns = df.columns.astype(float).astype(int)

        # Create and return a Triangle object
        return cls(id=id, tri=df.round(1), triangle=df.round(1), use_cal=use_cal)

    @classmethod
    def from_mack_1994(cls,
                       use_cal:bool = False) -> "Triangle":
        """
        Create a Triangle object from the sample triangle in the Mack 1994
        paper, "Measuring the Variability of Chain Ladder Reserve Estimates"

        (see https://www.casact.org/sites/default/files/2021-03/7_Mack_1994.pdf)

        Parameters:
        -----------
        use_cal : bool
            Whether or not to use calendar period effects in the linear
            model representation. Default is False.        

        Returns:
        --------
        Triangle
            A Triangle object with data loaded from the Taylor Ashe sample
            data.
        """
        # Get the current directory
        current_dir = os.path.dirname(os.path.realpath(__file__))

        # Construct the file path to the sample data
        data_file = os.path.join(current_dir, "data", "mack1994.csv")

        # Read the data from the CSV file
        df = pd.read_csv(data_file, header=0, index_col=0)

        # Create and return a Triangle object
        return cls(id="gl_rpt_loss", tri=df, triangle=df, use_cal=use_cal)

    @classmethod
    def from_taylor_ashe(cls,
                         use_cal: bool = False) -> "Triangle":
        """
        Create a Triangle object from the Taylor Ashe sample data.
        
        Parameters:
        -----------
        use_cal : bool
            Whether or not to use calendar period effects in the linear
            model representation. Default is False.
        
        Returns:
        --------
        Triangle
            A Triangle object with data loaded from the Taylor Ashe sample data.
        """
        # Get the current directory
        current_dir = os.path.dirname(os.path.realpath(__file__))

        # Construct the file path to the sample data
        data_file = os.path.join(current_dir, "data", "taylorashe.csv")

        # Read the data from the CSV file
        df = pd.read_csv(data_file, header=0, index_col=0)

        # Create and return a Triangle object
        return cls(id="paid_loss", tri=df, triangle=df, use_cal=use_cal)

    @classmethod
    def from_dahms(cls,
                   use_cal:bool = False) -> tuple:
        """
        Create a Triangle object from the Dahms sample data. This sample data
        contains both a reported and a paid triangle, so this method returns
        a tuple containing both triangles.

        Return is of the form (rpt, paid).

        Parameters:
        -----------
        use_cal : bool
            Whether or not to use calendar period effects in the linear
            model representation. Default is False.

        Returns:
        --------
        tuple[Triangle, Triangle]
            A tuple containing a Triangle object with data loaded from the
            reported triangle, and a Triangle object with data loaded from
            the paid triangle.
        """
        # Get the current directory
        current_dir = os.path.dirname(os.path.realpath(__file__))

        # Construct the file path to the sample data
        data_file = os.path.join(current_dir,
                                 "data",
                                 "dahms reserve triangles.xlsx")

        # Read the data from the CSV file
        paid = cls.from_excel(
            data_file,
            sheet_name="paid",
            id="paid_loss",
            origin_columns=1,
            sheet_range="a1:k11",
            use_cal=use_cal,
        )
        rpt = cls.from_excel(
            data_file,
            sheet_name="rpt",
            id="rpt_loss",
            origin_columns=1,
            sheet_range="a1:k11",
            use_cal=use_cal,
        )

        # Create and return a Triangle object
        return rpt, paid

    # GETTERS
    def getTriangle(self) -> pd.DataFrame:
        """
        Return the formatted triangle as a dataframe.

        Returns:
        --------
        pd.DataFrame
            The triangle data, formatted as a dataframe, and with the origin
            period formatting applied
        """
        # check that the index is a datetime index, and if not, convert it
        if not isinstance(self.tri.index, pd.DatetimeIndex):
            self.convert_origin_to_datetime()

        # return the triangle, formatted as a dataframe with the origin period
        # formatting applied
        return self.get_formatted_dataframe()

    def getCalendarIndex(self) -> pd.DataFrame:
        """
        Calculates a calendar index based on the number of months since the
        start of the origin period.
        """
        # same shape as original triangle
        cal = self.tri.copy()

        # start by setting each cell equal to the number of months since
        # year 0
        for c in cal.columns.tolist():
            cal[c] = ((cal.index.year.astype(int) - cal.index.year.astype(int).min()) +
                      (cal.index.month.astype(int) - cal.index.month.astype(int).min()))
                      
            # then add the column name as an integer
            cal[c] += int(c) / cal.columns.to_series().astype(int).min()

        return cal.astype(int)

    def getCurCalendarIndex(self) -> int:
        """
        Returns the current calendar period.
        """
        # start with calendarIndex
        cal_idx = self.getCalendarIndex()

        # get the first column
        col1 = cal_idx.iloc[:, 0]

        # return the max value in the current year column
        cur_calendar_index = col1.max()

        return cur_calendar_index

    def getCalendarYearIndex(self) -> pd.DataFrame:
        """
        Calculates a calendar year index based on the year of the transaction
        date.
        """
        # start with calendar index
        cal = (self.getCalendarIndex()
               
                # add the index to the first year included in the origin periods
                # (i.e. the first year in the index) then subtract 1 to get
                # the calendar year / year in which payments made in the
                # origin period are included
               .apply(lambda x: x.index.to_series().dt.year.min() + x - 1))

        return cal

    def cum_to_inc(
        self, cum_tri: pd.DataFrame = None, _return: bool = False
    ) -> pd.DataFrame:
        """
        Convert cumulative triangle data to incremental triangle data.

        Parameters:
        -----------
        cum_tri: pd.DataFrame
            The cumulative triangle data. Default is None, in which case
            the triangle data from the Triangle object is used.
        _return: bool
            If True, return the incremental triangle data. Default is False.

        Returns:
        --------
        inc_tri: pd.DataFrame
            The incremental triangle data.
        """
        # get the cumulative triangle data
        if cum_tri is None:
            cum_tri = self.tri

        # get the cumulative triangle data
        inc_tri = cum_tri - cum_tri.shift(1, axis=1, fill_value=0)

        # set the incremental triangle data
        self.incr_triangle = inc_tri

        # return the incremental triangle data
        if _return:
            return inc_tri

    # Basic triangle methods
    def _ata_tri(self) -> None:
        """
        Calculate the age-to-age factor triangle from the triangle data.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        # instantiate the ata triangle (same shape as the triangle data)
        ata = pd.DataFrame(
            np.zeros(self.tri.shape), index=self.tri.index, columns=self.tri.columns
        )

        # if there are any values of 0 in the triangle data, set them to nan
        self.tri[self.tri == 0] = np.nan

        # loop through the columns in the triangle (excl. the last column)
        for i in range(self.tri.shape[1] - 1):
            # calculate the age-to-age factor
            ata.iloc[:, i] = self.tri.iloc[:, i + 1] / self.tri.iloc[:, i]

        # set the last column of the ata triangle to nan
        ata.iloc[:, self.tri.shape[1] - 1] = np.nan

        return ata

    def _vwa(self, n: int | str = None, tail: float = 1.0) -> pd.DataFrame:
        """
        Calculate the volume weighted average (VWA) of the triangle data.

        Parameters:
        -----------
        n: int | str
            The number of periods to use in the VWA calculation. If "all", use
            all available periods. If None, use all available periods.
            Default is None.
        tail: float
            The tail factor to use in the VWA calculation. Default is 1.0, or
            no tail factor.

        Returns:
        --------
        vwa: pandas dataframe
            The VWA triangle data.
        """
        # instantiate the vwa results - a series whose length is equal to the number of
        # columns in the triangle data, with the index set to the column names
        vwa = pd.Series(
            np.zeros(self.tri.shape[1]), index=self.tri.columns, dtype=float
        )

        if isinstance(n, str):
            n = None if n.lower() != "all" else "all"

        # if n is None, use all available periods
        is_all = n is None or n == "all"

        # need a value for n in the loop below
        n2 = n if n is not None and n != "all" else self.tri.shape[0]

        # loop through the columns in the triangle data (excl. the last column)
        for i in range(self.tri.shape[1] - 1):
            next_col = self.tri.iloc[:, i + 1]
            cur_col = self.tri.iloc[:, i]
            if is_all or next_col.dropna().shape[0] <= n2:
                num = next_col.sum()
                den = cur_col.mask(next_col.isna(), np.nan).sum()
            else:
                num = next_col.dropna().tail(n).sum()
                den = cur_col.mask(next_col.isna(), np.nan).dropna().tail(n).sum()

            vwa.iloc[i] = num / den

        # set the last column of the vwa results to the tail factor
        vwa.iloc[self.tri.shape[1] - 1] = tail

        return vwa

    def _ave_ata(self, n: int | str = None, tail: float = 1.0) -> pd.Series:
        """
        Calculate the average age-to-age factor (Ave-ATA) of the triangle data.

        Parameters:
        -----------
        n: int | str
            The number of periods to use in the Ave-ATA calculation. If "all", use
            all available periods. If None, use all available periods.
        tail: float
            The tail factor to use in the Ave-ATA calculation. Default is 1.0, or
            no tail factor.

        Returns:
        --------
        ave_ata: pd.Series
            The Ave-ATA triangle data. Shape is the same as the number of columns
            in the triangle data, with the index set to the column names.
        """
        # instantiate the ave-ata results - a series whose length is equal to the number
        # of columns in the triangle data, with the index set to the column names
        ave_ata = pd.Series(
            np.zeros(self.tri.shape[1]), index=self.tri.columns, dtype=float
        )

        if isinstance(n, str):
            n = None if n.lower() != "all" else "all"

        # if n is None, use all available periods
        is_all = n is None or n == "all"

        # need a value for n in the loop below
        n2 = n if n is not None and n != "all" else self.tri.shape[0]

        # loop through the columns in the triangle data (excl. the last column)
        for i, column in enumerate(self.tri.columns[:-1]):
            # calculate the Ave-ATA -- if n is None, use all available periods
            # otherwise, use the last n periods, until the number of periods
            # is less than n (in which case, use all available periods)
            if is_all or self.tri.iloc[:, i + 1].dropna().shape[0] <= n2:
                ave_ata[column] = self._ata_tri().iloc[:, i].mean(skipna=True)
            else:
                # drop the na values so they aren't included in the average,
                # then average the previous n periods
                ave_ata[column] = (
                    self._ata_tri().iloc[:, i].dropna().tail(n).mean(skipna=True)
                )

        # set the last column of the ave-ata results to the tail factor
        ave_ata[self.tri.columns[-1]] = tail

        return ave_ata

    def _medial_ata(
        self, n: int | str = None, tail: float = 1.0, excludes: str = "hl"
    ) -> pd.Series:
        """
        Calculate the medial age-to-age factor (Medial-ATA) of the triangle data. This
        excludes one or more of the values in the average calculation. Once the values 
        are removed, the average is calculated as a normal average.

        Parameters:
        -----------
        n: int | str
            The number of periods to use in the Medial-ATA calculation. If "all", use
            all available periods. If None, use all available periods.
            Default is None.
        tail: float
            The tail factor to use in the Medial-ATA calculation. Default is 1.0, or
            no tail factor.
        excludes: str
            The exclusions to use in the average calculation. Default is 'hl',
            or high and low. If ave_type is 'triangle', this parameter is ignored.
            This parameter is a string of characters, where each character is an
            exclusion. The options are:
                h - high
                l - low
                m - median
            These characters can be in any order, and any number of them can be
            specified. For example, 'hl' excludes the high and low values, as does
            'lh', but 'hhl' excludes only the high value.

        Returns:
        --------
        medial_ata: pd.Series
            The Medial-ATA triangle data. Shape is the same as the number of columns
            in the triangle data, with the index set to the column names.
        """
        # instantiate the medial-ata results - a series whose length is equal to the
        # number of columns in the triangle data, with the index set to the column names
        medial_ata = pd.Series(
            np.zeros(self.tri.shape[1]), index=self.tri.columns, dtype=float
        )

        if isinstance(n, str):
            n = None if n.lower() != "all" else "all"

        # if n is None, use all available periods
        is_all = n is None or n == "all"

        # need a value for n in the loop below
        n2 = n if n is not None and n != "all" else self.tri.shape[0]

        # default if can't calculate this is to use the simple average
        default = self._vwa(n=n, tail=tail)

        # if the string contains 'h', exclude the high value, 'l' excludes the low value,
        # and 'm' excludes the median value
        exclude_high = "h" in excludes.lower()
        exclude_low = "l" in excludes.lower()
        exclude_median = "m" in excludes.lower()

        # loop through the columns in the triangle data (excl. the last column)
        for i, column in enumerate(self.tri.columns[:-1]):
            # temp column:
            temp_column = (self._ata_tri()).iloc[:, i].dropna()

            # count that there are enough values to calculate the average
            need_at_least = exclude_high + exclude_low + exclude_median

            # if there are not enough values to calculate the average, use the default
            if temp_column.shape[0] <= need_at_least:
                medial_ata[column] = default[column]
                continue
            else:
                # if we are not using all available periods, filter so only have
                # the last n periods available
                if is_all or self.tri.iloc[:, i + 1].dropna().shape[0] <= n2:
                    temp_column = temp_column.dropna()
                else:
                    temp_column = temp_column.dropna().tail(n)

                # if we are excluding the high value, remove it (same with low and median)
                if exclude_high:
                    temp_column = temp_column.drop(temp_column.idxmax())
                if exclude_low:
                    temp_column = temp_column.drop(temp_column.idxmin())
                if exclude_median:
                    # get id of median value
                    median_id = temp_column.shape[0] // 2
                    
                    # drop the median value whose id is the median_id (don't worry about
                    # the case where there are an even number of values, since the
                    # median id will be the lower of the two middle values, which is
                    # what we want)
                    temp_column = temp_column.drop(temp_column.index[median_id])

                # calculate the Medial-ATA
                medial_ata[column] = temp_column.mean(skipna=True)

        # set the last column of the medial-ata results to the tail factor
        medial_ata[self.tri.columns[-1]] = tail

        return medial_ata

    def ata(
        self,
        ave_type: str = "triangle",
        n: int = None,
        tail: float = 1.0,
        excludes: str = "hl",
    ) -> pd.DataFrame:
        """
        Returns the age-to-age factors of the triangle data, depending on the
        average type. Default is the triangle of age-to-age factors, but passing
        'vwa', 'simple', or 'medial' will return the volume weighted average,
        simple average, or medial average age-to-age factors, respectively. If one
        of the averages is selected, the number of periods to use in the average,
        tail factor, and exclusions can be specified (or they will use the defaults).

        Parameters:
        -----------
        ave_type: str
            The type of average to use. Options are 'triangle', 'vwa', 'simple',
            and 'medial'. Default is 'triangle'.
        n: int | str
            The number of periods to use in the average calculation. If None, or "all",
            use all available periods. If ave_type is 'triangle', this parameter is
            ignored.
        tail: float
            The tail factor to use in the average calculation. Default is 1.0, or
            no tail factor. If ave_type is 'triangle', this parameter is ignored.
        excludes: str
            The exclusions to use in the average calculation. Default is 'hl',
            or high and low. If ave_type is 'triangle', this parameter is ignored.
            This parameter is a string of characters, where each character is an
            exclusion. The options are:
                h - high
                l - low
                m - median
            These characters can be in any order, and any number of them can be
            specified. For example, 'hl' excludes the high and low values, as does
            'lh', but 'hhl' excludes only the high value.


        Returns:
        --------
        ata: pd.DataFrame
            The age-to-age factors of the triangle data, depending on the average
            type. Shape is the same as the triangle data.
        """
        # if the average type is 'triangle', return the triangle of age-to-age factors
        if ave_type.lower() == "triangle":
            return self._ata_tri()
        # if the average type is 'vwa', return the volume weighted average age-to-age factors
        elif ave_type.lower() == "vwa":
            return self._vwa(n=n, tail=tail)
        # if the average type is 'simple', return the simple average age-to-age factors
        elif ave_type.lower() == "simple":
            return self._ave_ata(n=n, tail=tail)
        # if the average type is 'medial', return the medial average age-to-age factors
        elif ave_type.lower() == "medial":
            return self._medial_ata(n=n, tail=tail, excludes=excludes)
        # if the average type is not recognized, raise an error
        else:
            raise ValueError(
                'Invalid age-to-age type. Must be "triangle", "vwa", "simple", or "medial"'
            )

    def atu(
        self,
        ave_type: str = "vwa",
        n: int = None,
        tail: float = 1.0,
        excludes: str = "hl",
        custom: np.ndarray = None,
    ) -> pd.DataFrame:
        """
        Calculates the age-to-ultimate factors from the triangle data.

        Parameters:
        -----------
        ave_type: str
            The type of average to use. Options are 'vwa', 'simple',
            and 'medial'. Default is 'vwa'.
        n: int
            The number of periods to use in the average calculation. If None, use
            all available periods.
        tail: float
            The tail factor to use in the average calculation. Default is 1.0, or
            no tail factor.
        excludes: str
            The exclusions to use in the average calculation. Default is 'hl',
            or high and low. This parameter is a string of characters, where each
            character is an exclusion. The options are:
                h - high
                l - low
                m - median
            These characters can be in any order, and any number of them can be
            specified. For example, 'hl' excludes the high and low values, as does
            'lh', but 'hhl' excludes only the high value.
        custom: np.ndarray
            A custom array of age-to-age factors to use in the calculation. If
            None, use the age-to-age factors calculated from the 'ave_type'.
            If not None, the 'ave_type', 'n', 'tail', and 'excludes' parameters
            are ignored.
            Default is None.

        Returns:
        --------
        atu: pd.DataFrame
            The age-to-ultimate factors of the triangle data.
        """
        # calculate the age-to-age factors
        if custom is None:
            age_to_age = self.ata(ave_type=ave_type, n=n, tail=tail, excludes=excludes)
        else:
            age_to_age = pd.Series(custom, index=self.tri.columns)

        # calculate the age-to-ultimate factors (cumulative product of the ata factors,
        # starting with the last column/the tail factor)
        age_to_ult = age_to_age[::-1].cumprod()[::-1]

        return age_to_ult

    def diag(self, calendar_year: int = None) -> pd.DataFrame:
        """
        Calculates the specified diagonal of the triangle data.

        Parameters:
        -----------
        calendar_year: int
            The calendar year of the diagonal to return. If None, return the
            current diagonal. Default is None.
            This is not implemented.

        Returns:
        --------
        diag: pd.DataFrame
            The diagonal of the triangle data.
        """
        # look at the triangle as an array
        triangle_array = self.tri.to_numpy()

        # if the calendar year is not specified, return the current diagonal
        if calendar_year is None:
            calendar_year = triangle_array.shape[0]
            # diagonal is a series of length equal to the number of rows in the triangle
            diag = pd.Series(np.diagonal(np.fliplr(triangle_array)), index=self.tri.index)
        # otherwise, return the specified diagonal
        else:
            diag = pd.Series(self.df.values[np.where(np.equal(self.cal, calendar_year))])

        return diag

    def ult(
        self,
        ave_type: str = "vwa",
        n: int = None,
        tail: float = 1.0,
        excludes: str = "hl",
        custom: np.ndarray = None,
        round_to: int = 0,
    ):
        """
        Calculates the ultimate loss from the standard chain ladder method.

        Parameters:
        -----------
        ave_type: str
            The type of average to use. Options are 'vwa', 'simple',
            and 'medial'. Default is 'vwa'.
        n: int
            The number of periods to use in the average calculation. If None, use
            all available periods.
        tail: float
            The tail factor to use in the average calculation. Default is 1.0, or
            no tail factor.
        excludes: str
            The exclusions to use in the average calculation. Default is 'hl',
            or high and low. This parameter is a string of characters, where each
            character is an exclusion. The options are:
                h - high
                l - low
                m - median
            These characters can be in any order, and any number of them can be
            specified. For example, 'hl' excludes the high and low values, as does
            'lh', but 'hhl' excludes only the high value.
        custom: np.ndarray
            A custom array of age-to-age factors to use in the calculation. If
            None, use the age-to-age factors calculated from the 'ave_type'.
            If not None, the 'ave_type', 'n', 'tail', and 'excludes' parameters
            are ignored.
            Default is None.
        round_to: int
            The number of decimal places to round the ultimate loss to. Default is 0.
        """
        diag = self.diag()

        # calculate the age-to-ultimate factors and reverse the order
        atu = self.atu(
            ave_type=ave_type, n=n, tail=tail, excludes=excludes, custom=custom
        )[::-1]
        atu.index = diag.index

        # calculate the ultimate loss
        ult = diag * atu
        ult.name = "Chain Ladder Ultimate Loss"
        ult.index.name = "Accident Period"

        return ult.round(round_to)

    def ata_summary(self) -> pd.DataFrame:
        """
        Produces a fixed summary of the age-to-age factors for the triangle
        data.

        Contains the following:
            - Triangle of age-to-age factors
            - Volume weighted average age-to-age factors for all years,
              5 years, 3 years, and 2 years
            - Simple average age-to-age factors for all years, 5 years,
              3 years, and 2 years
            - Medial average age-to-age factors for 5 years, excluding
              high, low, and high/low values
        """

        triangle = self

        ata_tri = triangle.ata().round(3)

        vol_wtd = pd.DataFrame(
            {
                "Vol Wtd": pd.Series(
                    ["" for _ in range(ata_tri.shape[1] + 1)],
                    index=ata_tri.reset_index().columns,
                ),
                "All Years": triangle.ata("vwa").round(3),
                "5 Years": triangle.ata("vwa", 5).round(3),
                "3 Years": triangle.ata("vwa", 3).round(3),
                "2 Years": triangle.ata("vwa", 2).round(3),
            }
        ).transpose()

        simple = pd.DataFrame(
            {
                "Simple": pd.Series(
                    ["" for _ in range(ata_tri.shape[1] + 1)],
                    index=ata_tri.reset_index().columns,
                ),
                "All Years": triangle.ata("simple").round(3),
                "5 Years": triangle.ata("simple", 5).round(3),
                "3 Years": triangle.ata("simple", 3).round(3),
                "2 Years": triangle.ata("simple", 2).round(3),
            }
        ).transpose()

        medial = pd.DataFrame(
            {
                "Medial 5-Year": pd.Series(
                    ["" for _ in range(ata_tri.shape[1] + 1)],
                    index=ata_tri.reset_index().columns,
                ),
                "Ex. Hi/Low": triangle.ata("medial", 5, excludes="hl").round(3),
                "Ex. Hi": triangle.ata("medial", 5, excludes="h").round(3),
                "Ex. Low": triangle.ata("medial", 5, excludes="l").round(3),
            }
        ).transpose()

        out = (
            pd.concat(
                [ata_tri.drop(index=ata_tri.index[-1]), vol_wtd, simple, medial],
                axis=0,
            )
            .drop(columns=self.tri.columns[-1])
            .fillna("")
        )

        # check to see if the last column is all '' (empty strings)
        if out.iloc[:, -1].str.contains("").all():
            out = out.drop(columns=out.columns[-1])

        # try to reformat the index
        out.index = [
            i.strftime("%Y") if isinstance(i, datetime) else i for i in out.index
        ]

        # rename the columns to put a title above the df
        out.columns.name = "Age-to-Age Factors as of (months)"

        return out

    def melt_triangle(
        self,
        id_cols: list = None,
        var_name: str = "development_period",
        value_name: str = "tri",
        _return: bool = True,
        incr_tri: bool = True,
    ) -> pd.DataFrame:
        """
        Melt the triangle data into a single column of values.
        Parameters:
        -----------
        id_cols: list
            The columns to use as the id variables. Default is None, in which
            case the index is used.
        var_name: str
            The name of the variable column. Default is 'development_period'.
        value_name: str
            The name of the value column. Default is None, in which case
            the value column is set equal to the triangle ID.
        _return: bool
            If True, return the melted triangle data as a pandas dataframe.
            Default is True.
        incr_tri: bool
            If True, use the incremental triangle data. Default is True. If
            False, use the cumulative triangle data.

        Returns:
        --------
        melted: pd.DataFrame
            The melted triangle data.
        """
        # if id_cols is None, use the index
        if id_cols is None:
            id_cols = self.tri.index.name

        # if value_name is None, use the triangle ID
        if value_name is None:
            value_name = self.id

        # get the triangle data
        if incr_tri:
            if self.incr_triangle is None:
                self.cum_to_inc()
            tri = self.incr_triangle
        else:
            tri = self.triangle

        tri.index = self.get_formatted_dataframe().index

        # melt the triangle data
        melted = tri.reset_index().melt(id_vars=id_cols,
                                        var_name=var_name,
                                        value_name=value_name)

        # if _return is True, return the melted triangle data
        if _return:
            return melted

    def create_design_matrix_levels(self,
                                    column: pd.Series = None,
                                    z: int = 4,
                                    s: str = None) -> pd.DataFrame:
        """
        Creates a design matrix from a given column. The column is treated as
        categorical, zero-padded to the length specified by `z`, and one-hot-
        encoded using pandas.get_dummies. The column names of the resulting
        design matrix are in the format `s_{zero-padded z}`.

        Parameters:
        -----------
        column: pd.Series, Optional
            The column to be transformed into a design matrix. This should
            be a pandas.Series object. Default is None, in which case the
            column is set equal to the accident period column from the 
            melted triangle data.
        z: int, Optional
            The length to which category labels should be zero-padded.
            Default is 4.
        s: str, Optional
            The string to be used as a prefix in the column names of the
            design matrix. Default is None, in which case the string is
            the `name` attribute of the input column.

        Returns:
        --------
        result: pd.DataFrame
            The design matrix as a pandas DataFrame. It consists of the 
            original column followed by the one- hot-encoded categories from
            the input column, with column names in the format
            `s_{zero-padded z}`.

        Examples:
        ---------
        >>> df = pd.DataFrame({'accident_period': [1, 2, 3, 4, 5]})
        >>> create_design_matrix_levels(df['accident_period'],
                                        z=2,
                                        s='acc')
        >>>   accident_period  acc_01  acc_02  acc_03  acc_04  acc_05
        >>> 0               1       1       0       0       0       0
        >>> 1               2       0       1       0       0       0
        >>> 2               3       0       0       1       0       0
        >>> 3               4       0       0       0       1       0
        >>> 4               5       0       0       0       0       1

        Raises:
        -------
        TypeError:
            1. If the input column is not a pandas.Series object.
            2. If the zero-padding length is not an integer.
            3. If the prefix string is not a string, or cannot be coerced
            to a string.

        ValueError:
            1. If the zero-padding length is not an integer greater
                than 0.
        """
        # if column is None, use the accident period column from the melted
        # triangle data
        if column is None:
            column = self.melt_triangle()['accident_period']

        if isinstance(column, str):
            col_name = column
            column = self.melt_triangle()[column]
            s = col_name if s is None else s

        # if column is not a pandas.Series object, raise an error
        if not isinstance(column, pd.Series):
            raise TypeError('Input column must be a pandas.Series object.')

        # if z is not an integer, raise an error
        if not isinstance(z, int):
            raise TypeError('Zero-padding length must be an integer.')

        # if z is not greater than 0, raise an error
        if z <= 0:
            raise ValueError('Zero-padding length must be greater than 0.')

        # Ensure column is treated as a string
        column = column.astype(str)

        # Zero-pad the category labels
        column_copy = column.copy().apply(lambda x: str(x).zfill(z))

        # One-hot-encode the column with pandas.get_dummies
        encoded = pd.get_dummies(column_copy, drop_first=True).astype(int)

        # Rename the columns
        encoded.columns = [f"{s}_{label}" for label in encoded.columns]

        # Include the original column as the first column
        result = pd.concat([column.astype(int), encoded], axis=1)

        return result

    def create_design_matrix_trends(self,
                                    column: pd.Series = None,
                                    z: int = 4,
                                    s: str = None) -> pd.DataFrame:
        """
        Creates a design matrix from a given column. The column is treated as
        categorical, zero-padded to the length specified by `z`, and encoded
        such that all categories less than or equal to the given category get
        a 1, while the rest get a 0. The column names of the resulting
        design matrix are in the format `s_{zero-padded z}`.

        Parameters:
        -----------
        column: pd.Series, Optional
            The column to be transformed into a design matrix. This should
            be a pandas.Series object. Default is None, in which case the
            column is set equal to the accident period column from the 
            melted triangle data.
        z: int, Optional
            The length to which category labels should be zero-padded.
            Default is 4.
        s: str, Optional
            The string to be used as a prefix in the column names of the
            design matrix. Default is None, in which case the string is
            the `name` attribute of the input column.

        Returns:
        --------
        result: pd.DataFrame
            The design matrix as a pandas DataFrame. It consists of encoded
            categories from the input column, with column names in the format
            `s_{zero-padded z}` and the original column as the first column.

        Raises:
        -------
        TypeError:
            1. If the input column is not a pandas.Series object.
            2. If the zero-padding length is not an integer.
            3. If the prefix string is not a string, or cannot be coerced
            to a string.
        ValueError:
            1. If the zero-padding length is not an integer greater
                than 0.
        """
        # start with the levels design matrix from before
        start = self.create_design_matrix_levels(column=column,
                                                 z=z,
                                                 s=s)

        trends = start.copy()
        # for each column in the design matrix,
        for i, c in enumerate(start.columns.tolist()):
            if i == 0:  # do not adjust the very first column
                trends[c] = start[c].values
            else:  # if the current column or any column to the right of the current column is equal to 1
                trends[c] = np.where(start.iloc[:, i:].sum(axis=1) > 0, 1, 0)

        return trends
        

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
                             int(float(x[1])/melted['development_period'].astype(float).min()),
                   axis=1))
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
        self.X_base = self.X_base[['is_observed', 'intercept'] + self.X_base.columns.drop(['is_observed', 'intercept']).tolist()]
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

    def get_train_forecast_split(self,
                                 custom_split:list|
                                              np.ndarray|
                                              pd.Series|
                                              None = None,
                                 return_:bool = True
        ) -> pd.Series:
        """
        Splits self.X_base and self.y_base into train and forecast datasets.
        This function is used as a helper function for the base_design_matrix
        method, and is not intended to be called directly.

        Parameters:
        -----------
        custom_split: list | np.ndarray | pd.Series | None
            A custom split to use for the train/forecast split. If None,
            use the default train/forecast split. Default is None.
        return_: bool
            If True, return the train/forecast split as a pandas series.

        Returns:
        --------
        train_forecast: pd.Series
            A series of 1s and 0s, where 1 indicates a row in the train
            dataset, and 0 indicates a row in the forecast dataset.

        Also sets self.X_base_train, self.X_base_forecast, self.y_base_train,
        self.X_id_train, and self.X_id_forecast.
        """
        # if custom_split is None, use set custom split to the is_observed
        # column of the base design matrix
        if custom_split is None:
            custom_split = self.X_base.is_observed

        # now use the custom train/forecast split to create the
        # train/forecast design matrices
        self.X_base_train = self.X_base.loc[custom_split.eq(1)]
        self.X_base_forecast = self.X_base.loc[custom_split.eq(0)]
        self.y_base_train = self.y_base[custom_split.eq(1)]
        self.X_id_train = self.X_id.loc[custom_split.eq(1)]
        self.X_id_forecast = self.X_id.loc[custom_split.eq(0)]

        # return the custom train/forecast split
        if return_:
            return custom_split

    def get_X(
        self,
        kind: str = None,
    ) -> pd.DataFrame:
        """
        Get the design matrix for the given kind.

        Parameters:
        -----------
        kind: str
            The kind to get the design matrix for. If None, return the full design
            matrix. If "train", return the training design matrix. If "forecast",
            return the forecast design matrix. Default is None.

        Returns:
        --------
        X: pd.DataFrame
            The design matrix for the given kind.
        """
        if kind is None:
            X = self.X_base.copy()
        elif kind.lower()=="train":
            X = self.X_base_train.copy()
            X.drop('is_observed', axis=1, inplace=True)
        elif kind.lower()=='forecast':
            X = self.X_base_forecast.copy()
            X.drop('is_observed', axis=1, inplace=True)
        else:
            raise ValueError("kind must be 'train', 'forecast', or None.")
        
        return X

    def get_X_cal(self, kind=None) -> pd.DataFrame:
        """
        Returns the calendar design matrix
        """
        
        if self.use_cal:
            df = self.get_X(kind=kind)
            cols = df.columns        
            col_qry = cols.str.contains("cal") 
            col_qry = col_qry | cols.str.contains("calendar_period")
            df = df.loc[:, col_qry.tolist()]
            
        else:
            if kind is None:
                qry = pd.Series(np.ones_like(self.X_id['calendar_period'].values),
                                index=self.X_id.index).eq(1)
            elif kind.lower() == "train":
                qry = self.X_id['is_observed'].eq(1)
            elif kind.lower() == "forecast":
                qry = self.X_id['is_observed'].eq(0)
            else:
                raise ValueError("kind must be 'train', 'forecast', or None.")
            
            cal = self.X_id['calendar_period']
            X = self.create_design_matrix_trends(cal,s="calendar_period",z=4)
            
            df = X.loc[qry]

        return df
    
    def get_X_exposure(self) -> pd.DataFrame:
        """
        Returns the exposure design matrix
        """
        return self.exposure

    def get_X_base(self, kind=None):
        """
        Returns the base design matrix
        """
        df = self.get_X(kind=kind)
        return df

    def get_y_base(self, kind=None):
        """
        Returns the labels for the base design matrix
        """
        if kind is None:
            df = self.y_base
        elif kind == "train":
            df = self.y_base_train
        elif kind == "forecast":
            df = self.y_base_forecast
        else:
            df = self.y_base

        return df

    def get_X_id(self, kind=None):
        """
        Returns the labels for the base design matrix
        """
        if kind is None:
            df = self.X_id
        elif kind == "train":
            df = self.X_id_train
        elif kind == "forecast":
            df = self.X_id_forecast
        else:
            df = self.X_id

        return df

    # def prep_for_cnn(self, steps=False) -> pd.DataFrame:
    #     """
    #     Performs data preprocessing and reshaping to transform triangle into form
    #     suitable for inference on LossTriangleClassifier

    #     If steps is True, returns a dictionary of steps taken to preprocess
    #     """
    #     from sklearn.preprocessing import StandardScaler
    #     from torch.utils.data import DataLoader

    #     # initialize if needed
    #     if steps:
    #         out_dict = {}

    #     # read indicators
    #     ind = pd.read_csv("data/LossTriangleClassifierShape.csv").iloc[:, 1:]
    #     if steps:
    #         out_dict["01. Indicators"] = ind

    #     # copy triangle
    #     df = self.tri.copy()
    #     if steps:
    #         out_dict["02. Current Triangle"] = df

    #     # only last 10 rows, first 10 columns
    #     df = df.iloc[-10:, :10]
    #     if steps:
    #         out_dict["03. Remove all but last 10 rows, first 10 cols"] = df

    #     # replace empty strings with nan values
    #     df = df.replace("", np.nan)
    #     if steps:
    #         out_dict["04. Replace empty strings with nan"] = df

    #     # set index & columns of `ind` equal to those in `df` so you can
    #     # easily multiply them
    #     ind.index = df.index
    #     ind.columns = df.columns

    #     # multiply them
    #     mult = df * ind
    #     if steps:
    #         out_dict["05. Multiply original triangle with indicators"] = mult

    #     # replace blank values with nan
    #     replaced = mult.replace("", np.nan)
    #     if steps:
    #         out_dict["06. Replace blank values with nan"] = replaced

    #     # preprocess
    #     scaler = StandardScaler()
    #     scaled = scaler.fit_transform(replaced.values)
    #     scaled = pd.DataFrame(scaled, index=replaced.index, columns=replaced.columns)
    #     if steps:
    #         out_dict["07. Standardize over the whole triangle"] = scaled

    #     # fill na values with 0s
    #     scaled = scaled.fillna(0)

    #     if steps:
    #         return out_dict
    #     else:
    #         return scaled

    #     id: str = None
    # tri: pd.DataFrame = None
    # triangle: pd.DataFrame = None
    # incr_triangle: pd.DataFrame = None
    # X_base: pd.DataFrame = None
    # y_base: np.ndarray = None
    # X_base_train: pd.DataFrame = None
    # y_base_train: np.ndarray = None
    # X_base_forecast: pd.DataFrame = None
    # y_base_forecast: np.ndarray = None
    # has_cum_model_file: bool = False
    # is_cum_model: Any = None

    # def _load_is_cum_model(self, model_file = None):

    #     # pre-fit/saved triangle model
    #     model_file = r"C:\Users\aweaver\OneDrive - The Cincinnati Insurance Company\rocky\inc_cum_tri.torch"

    #     # initialize model
    #     model = LossTriangleClassifier(torch.Size([1, 10, 10]),
    #                                    num_classes=2,
    #                                    num_conv_layers=5,
    #                                    base_conv_nodes=256,
    #                                    kernel_size=(2, 2),
    #                                    stride=(1, 1),
    #                                    padding=(1, 1),
    #                                    linear_nodes=[1024, 512, 256, 128],
    #                                    linear_dropout=[0.4, 0.3, 0.2, 0.1],
    #                                    relu_neg_slope=0.1)

    #     # load model on CPU
    #     model.to(torch.device('cpu'))

    #     # load saved parameters to instanciated model
    #     model.load_state_dict(torch.load(
    #         model_file, map_location=torch.device('cpu')))

    #     self.is_cum_model = model

    # def _is_cum_model(self):
    #     # build DataLoader from the preprocessed data
    #     data = DataLoader(self.prep_for_cnn().values, batch_size=1)

    #     # set model to evaluate (eg not train)
    #     self.is_cum_model.eval()

    #     # no gradient update
    #     with torch.no_grad():
    #         inputs = torch.from_numpy(data.dataset.reshape(1, 10, 10))
    #         pred = self.is_cum_model(inputs.float().unsqueeze(0))

    #     self._is_cum_pred = pred
    #     self.is_cum = torch.argmax(pred, dim=1).cpu().item()
