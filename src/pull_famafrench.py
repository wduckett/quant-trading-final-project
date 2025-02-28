"""
This module pulls and saves data from Fama French dataset from WRDS.


"""
from pathlib import Path

import pandas as pd
import wrds

from settings import config

# ==============================================================================================
# Global Configuration
# ==============================================================================================

RAW_DATA_DIR = Path(config("RAW_DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")


# ==============================================================================================
# Fama French Data Functions
# ==============================================================================================

def pull_fama_french_factors(wrds_username=WRDS_USERNAME, freq="D"):
    conn = wrds.Connection(wrds_username=wrds_username)
    if freq == "D":
        ff = conn.get_table(library="ff", table="factors_daily")
    elif freq == "M":
        ff = conn.get_table(library="ff", table="factors_monthly")
    conn.close()
    return ff


def load_fama_french_factors(data_dir=RAW_DATA_DIR, freq="D"):
    if freq == "D":
        path = Path(data_dir) / "FF_FACTORS_D.parquet"
    elif freq == "M":
        path = Path(data_dir) / "FF_FACTORS_M.parquet"
    ff = pd.read_parquet(path)
    return ff


def load_CRSP_Comp_Link_Table(data_dir=RAW_DATA_DIR):
    path = Path(data_dir) / "CRSP_Comp_Link_Table.parquet"
    ccm = pd.read_parquet(path)
    return ccm



if __name__ == "__main__":

    ff_dly = pull_fama_french_factors(wrds_username=WRDS_USERNAME, freq="D")
    ff_dly.to_parquet(RAW_DATA_DIR / "FF_FACTORS_D.parquet")

    ff_mth = pull_fama_french_factors(wrds_username=WRDS_USERNAME, freq="M")
    ff_mth.to_parquet(RAW_DATA_DIR / "FF_FACTORS_M.parquet")