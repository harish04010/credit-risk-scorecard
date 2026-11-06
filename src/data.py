"""
Loading, validating and documenting the Taiwan credit card default data.

30,000 credit card accounts from a Taiwanese bank, April to September 2005.
Target: default on the October 2005 payment. Yeh, I. C., & Lien, C. H. (2009),
Expert Systems with Applications 36(2). UCI Machine Learning Repository,
CC BY 4.0, doi:10.24432/C55S3H.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
XLS = ROOT / "data" / "default_of_credit_card_clients.xls"
SOURCE_URL = ("https://raw.githubusercontent.com/robertofranceschi/Default-Credit-Card-Prediction/master/"
              "dataset/default%20of%20credit%20card%20clients.xls")

MONTHS = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"]  # suffix 1..6, most recent first


def load_raw() -> pd.DataFrame:
    if not XLS.exists():
        import urllib.request
        XLS.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE_URL, XLS)
    df = pd.read_excel(XLS, header=1)
    # The file calls the most recent status PAY_0 but the bills BILL_AMT1. Make them line up.
    return df.rename(columns={"PAY_0": "PAY_1", "default payment next month": "default"})


def quality_report(df: pd.DataFrame) -> list[dict]:
    """Every rule the data breaks, how often, and what the pipeline does about it."""
    pay = df[[f"PAY_{i}" for i in range(1, 7)]]
    bills = df[[f"BILL_AMT{i}" for i in range(1, 7)]]
    return [
        {"check": "Rows and unique IDs", "count": int(df["ID"].nunique()), "of": len(df),
         "treatment": "One row per account, no duplicates."},
        {"check": "EDUCATION codes 0, 5, 6 (dictionary defines only 1 to 4)", "count": int(df["EDUCATION"].isin([0, 5, 6]).sum()),
         "of": len(df), "treatment": "Grouped with 4 (other). Education isn't a scorecard input, so this only affects reporting."},
        {"check": "MARRIAGE code 0 (undefined)", "count": int((df["MARRIAGE"] == 0).sum()), "of": len(df),
         "treatment": "Kept as unknown. Marital status is a prohibited basis under ECOA and is excluded from the model."},
        {"check": "Repayment status -2 or 0 (not in the documentation)", "count": int(pay.isin([-2, 0]).any(axis=1).sum()),
         "of": len(df), "treatment": "Read as -2 = no balance to pay, -1 = paid in full, 0 = revolving (minimum paid). "
                                     "These are ordered and binned, not treated as delinquency."},
        {"check": "Accounts with a negative bill in any month (credit balance)", "count": int((bills < 0).any(axis=1).sum()),
         "of": len(df), "treatment": "Kept. Utilization is floored at zero."},
        {"check": "Accounts over their credit limit in September", "count": int((df["BILL_AMT1"] > df["LIMIT_BAL"]).sum()),
         "of": len(df), "treatment": "Kept. Utilization is capped at 150% so a few extreme values don't drive bins."},
        {"check": "Missing values", "count": int(df.isna().sum().sum()), "of": int(df.size), "treatment": "None found."},
        {"check": "Accounts with no balance in any of the six months, yet labeled as defaulting",
         "count": int((inactive(df) & (df["default"] == 1)).sum()), "of": int(inactive(df).sum()),
         "treatment": f"These {int(inactive(df).sum())} inactive accounts default at {df.loc[inactive(df), 'default'].mean():.0%}, "
                      "above the book's average, which is implausible for accounts that owe nothing. They're excluded from "
                      "development as indeterminate, the usual practice for behavior scorecards, and flagged for the data owner."},
    ]


def inactive(df: pd.DataFrame) -> pd.Series:
    """No balance owed in any of the six statement months."""
    return (df[[f"BILL_AMT{i}" for i in range(1, 7)]] <= 0).all(axis=1)


def features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Behavioral features a card issuer would build from six months of statements.
    Sex, marital status and age are deliberately left out: sex and marital
    status are prohibited bases under the Equal Credit Opportunity Act, and age
    can only be used under narrow conditions. Sex is kept aside to test fairness.
    """
    f = pd.DataFrame(index=df.index)
    lim = df["LIMIT_BAL"].to_numpy(dtype=float)
    bills = np.column_stack([df[f"BILL_AMT{i}"] for i in range(1, 7)]).astype(float)
    pays = np.column_stack([df[f"PAY_AMT{i}"] for i in range(1, 7)]).astype(float)
    status = np.column_stack([df[f"PAY_{i}"] for i in range(1, 7)])

    util = np.clip(bills / lim[:, None], 0, 1.5)
    f["limit"] = lim
    f["utilization"] = util[:, 0]
    f["avg_utilization_6m"] = util.mean(axis=1)
    f["utilization_trend"] = util[:, 0] - util[:, 5]
    f["status_now"] = status[:, 0]
    f["status_last_month"] = status[:, 1]
    f["worst_status_6m"] = status.max(axis=1)
    f["months_delinquent_6m"] = (status > 0).sum(axis=1)
    # Payment in month t against the bill from month t+1 (the statement it pays).
    owed = np.clip(bills[:, 1:], 1, None)
    ratio = np.clip(pays[:, :5] / owed, 0, 1.5)
    ratio[bills[:, 1:] <= 0] = 1.0  # nothing owed counts as fully paid
    f["payment_ratio"] = ratio[:, 0]
    f["avg_payment_ratio_5m"] = ratio.mean(axis=1)
    f["months_paid_nothing_5m"] = ((pays[:, :5] == 0) & (bills[:, 1:] > 0)).sum(axis=1)
    return f


def protected(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"sex": df["SEX"].map({1: "Male", 2: "Female"}),
                         "age_band": pd.cut(df["AGE"], [0, 29, 44, 59, 200], labels=["Under 30", "30-44", "45-59", "60+"])})
