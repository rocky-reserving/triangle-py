from dataclasses import dataclass
import sys

sys.path.append("../triangle")

from triangle import Triangle
import pandas as pd
import numpy as np

@dataclass
class TestCase:
    name: str
    triangle: Triangle
    df: pd.DataFrame
    incremental: pd.DataFrame
    ata_triangle: pd.DataFrame
    vwa_all: pd.Series
    vwa_4: pd.Series
    vwa_2_105tail: pd.Series
    