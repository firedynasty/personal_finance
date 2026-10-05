"""Streamlit front end for categorize_transactions: upload a CSV, review, download.

Run with: streamlit run categorize_app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from categorize_transactions import (
    DEFAULT_RULES_FILE,
    PAYMENT,
    UNCATEGORIZED,
    add_keyword,
    apply_rules,
    category_labels,
    load_rules,
    load_transactions,
    parse_rules,
    suggest_keyword,
    summarize,
    summarize_subcategories,
)

RULES_FILE = DEFAULT_RULES_FILE


def unknown_merchants(result: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct unknown merchant, with a suggested keyword and empty category."""
    unknown = result[result["category"] == UNCATEGORIZED]
    rows = {}
    for description in unknown["description"]:
        keyword = suggest_keyword(description)
        rows.setdefault(keyword, {"keyword": keyword, "example": description, "count": 0})
        rows[keyword]["count"] += 1
    table = pd.DataFrame(list(rows.values()), columns=["keyword", "example", "count"])
    table["category"] = pd.Series(dtype="object")
    return table


def edit_rules_file() -> None:
    """Sidebar editor for the categories text file; saves only if it still parses."""
    with st.sidebar.expander("Edit categories"):
        st.caption("[Category] or [Category/Subcategory] headers, one keyword per line.")
        text = st.text_area("categories.txt", RULES_FILE.read_text(), height=400)
        if st.button("Save categories file"):
            try:
                parse_rules(text)
            except ValueError as exc:
                st.error(str(exc))
            else:
                RULES_FILE.write_text(text if text.endswith("\n") else text + "\n")
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="Transaction Categorizer", layout="wide")
    st.title("🏷️ Transaction Categorizer")
    st.caption("Upload a bank/credit-card CSV with Date, Description, Debit and Credit columns.")

    edit_rules_file()
    uploaded = st.file_uploader("Transactions CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to begin.")
        return

    try:
        transactions = load_transactions(uploaded)
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not read this file: {exc}")
        return

    rules = load_rules(RULES_FILE)
    result = apply_rules(transactions, rules)

    # --- Unknown merchants: pick a category once, remember it ---
    unknown = unknown_merchants(result)
    if not unknown.empty:
        st.subheader(f"{len(unknown)} unknown merchant(s)")
        st.caption("Pick a category and edit the keyword if needed. Rules apply to future files.")
        edited = st.data_editor(
            unknown,
            hide_index=True,
            use_container_width=True,
            disabled=["example", "count"],
            column_config={
                "category": st.column_config.SelectboxColumn(
                    "category", options=category_labels(rules)
                ),
            },
        )
        if st.button("Save categories"):
            chosen = edited.dropna(subset=["category"])
            for _, row in chosen.iterrows():
                add_keyword(RULES_FILE, row["category"], row["keyword"])
            st.rerun()

    # --- Summary ---
    st.subheader("Spending by category")
    summary = summarize(result)
    left, right = st.columns([1, 2])
    left.dataframe(summary, use_container_width=True)
    totals = summary.drop(index="Total")["total"]
    totals = totals[totals > 0]  # a pie can't show negative (net-refund) categories
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(totals, labels=totals.index, autopct="%1.0f%%", startangle=90, counterclock=False)
    ax.axis("equal")
    right.pyplot(fig)
    st.caption(f"Card payments ({PAYMENT}) are excluded; refunds net against spending.")

    subcategories = summarize_subcategories(result)
    if not subcategories.empty:
        st.subheader("Subcategories")
        st.dataframe(subcategories, use_container_width=True)

    # --- Transactions + download ---
    st.subheader("Transactions")
    categories = sorted(result["category"].unique())
    selected = st.multiselect("Filter by category", categories, default=categories)
    st.dataframe(result[result["category"].isin(selected)], hide_index=True,
                 use_container_width=True)

    download = result.assign(date=result["date"].dt.strftime("%Y-%m-%d"))
    st.download_button("Download categorized CSV", download.to_csv(index=False),
                       file_name=f"{Path(uploaded.name).stem}_categorized.csv", mime="text/csv")


if __name__ == "__main__":
    main()
