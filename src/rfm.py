"""
src/rfm.py
----------
Reusable RFM (Recency, Frequency, Monetary) scoring module.
Used by notebooks 02 and beyond.
"""

import pandas as pd
import numpy as np
from datetime import timedelta


def compute_rfm(df: pd.DataFrame,
                customer_col: str = 'Customer ID',
                invoice_col:  str = 'Invoice',
                date_col:     str = 'InvoiceDate',
                revenue_col:  str = 'TotalRevenue',
                snapshot_date=None) -> pd.DataFrame:
    """
    Compute raw RFM values from a transactions dataframe.

    Parameters
    ----------
    df            : Clean transactions dataframe
    customer_col  : Column name for customer ID
    invoice_col   : Column name for invoice/order ID
    date_col      : Column name for transaction date
    revenue_col   : Column name for revenue per row
    snapshot_date : Reference date for recency (default: max date + 1 day)

    Returns
    -------
    pd.DataFrame with columns: Customer ID, Recency, Frequency, Monetary
    """
    if snapshot_date is None:
        snapshot_date = df[date_col].max() + timedelta(days=1)

    rfm = (
        df.groupby(customer_col)
        .agg(
            Recency   = (date_col,    lambda x: (snapshot_date - x.max()).days),
            Frequency = (invoice_col, 'nunique'),
            Monetary  = (revenue_col, 'sum')
        )
        .reset_index()
    )

    return rfm


def score_rfm(rfm: pd.DataFrame, q: int = 5) -> pd.DataFrame:
    """
    Add quintile-based scores for R, F, M dimensions.

    Recency  : lower days = higher score (inverted)
    Frequency: higher count = higher score
    Monetary : higher spend = higher score

    Parameters
    ----------
    rfm : DataFrame with Recency, Frequency, Monetary columns
    q   : Number of quantile bins (default: 5)

    Returns
    -------
    DataFrame with added R_Score, F_Score, M_Score, RFM_Score, RFM_Label
    """
    rfm = rfm.copy()

    labels = list(range(1, q + 1))

    rfm['R_Score'] = pd.qcut(
        rfm['Recency'],
        q=q,
        labels=labels[::-1]   # invert: low recency = high score
    ).astype(int)

    rfm['F_Score'] = pd.qcut(
        rfm['Frequency'].rank(method='first'),
        q=q,
        labels=labels
    ).astype(int)

    rfm['M_Score'] = pd.qcut(
        rfm['Monetary'].rank(method='first'),
        q=q,
        labels=labels
    ).astype(int)

    rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
    rfm['RFM_Label'] = (
        rfm['R_Score'].astype(str) +
        rfm['F_Score'].astype(str) +
        rfm['M_Score'].astype(str)
    )

    return rfm


def label_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Apply rule-based segment labels based on R, F, M scores.

    Segments:
        Champions         — recent, frequent, high spend
        Loyal Customers   — consistent buyers, good spend
        Potential Loyalists — recent buyers with growth potential
        New Customers     — bought recently but only once
        Promising         — moderate recency, low frequency
        At Risk           — used to buy often but haven't recently
        Cannot Lose Them  — high-value but at risk
        Hibernating       — low activity across all dimensions
        Lost              — lowest scores, inactive

    Parameters
    ----------
    rfm : DataFrame with R_Score, F_Score, M_Score columns

    Returns
    -------
    DataFrame with added Segment_RuleBased column
    """
    def _label(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']

        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3 and m >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f <= 2:
            return 'New Customers'
        elif r >= 3 and f >= 2 and m >= 2:
            return 'Potential Loyalists'
        elif r == 3 and f <= 2:
            return 'Promising'
        elif r <= 2 and f >= 4 and m >= 4:
            return 'Cannot Lose Them'
        elif r <= 2 and f >= 3 and m >= 3:
            return 'At Risk'
        elif r <= 2 and f <= 2 and m <= 2:
            return 'Lost'
        else:
            return 'Hibernating'

    rfm = rfm.copy()
    rfm['Segment_RuleBased'] = rfm.apply(_label, axis=1)
    return rfm


def segment_summary(rfm: pd.DataFrame,
                    segment_col: str = 'Segment_RuleBased') -> pd.DataFrame:
    """
    Generate a business summary table per segment.

    Returns
    -------
    DataFrame with: Customers, Avg_Recency, Avg_Frequency,
                    Avg_Monetary, Total_Revenue, Revenue_Share_%
    """
    total_rev = rfm['Monetary'].sum()

    summary = (
        rfm.groupby(segment_col)
        .agg(
            Customers     = ('Monetary',  'count'),
            Avg_Recency   = ('Recency',   'mean'),
            Avg_Frequency = ('Frequency', 'mean'),
            Avg_Monetary  = ('Monetary',  'mean'),
            Total_Revenue = ('Monetary',  'sum'),
        )
        .round(1)
        .sort_values('Total_Revenue', ascending=False)
    )

    summary['Revenue_Share_%'] = (
        summary['Total_Revenue'] / total_rev * 100
    ).round(1)

    return summary


def full_rfm_pipeline(df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """
    Run the complete RFM pipeline in one call:
        compute_rfm → score_rfm → label_segments

    Parameters
    ----------
    df            : Clean transactions dataframe
    snapshot_date : Optional reference date

    Returns
    -------
    Complete RFM dataframe with scores and segment labels
    """
    rfm = compute_rfm(df, snapshot_date=snapshot_date)
    rfm = score_rfm(rfm)
    rfm = label_segments(rfm)
    return rfm
