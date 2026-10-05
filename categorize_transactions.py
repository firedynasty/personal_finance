"""Categorize bank/credit-card transactions from a CSV using merchant keyword rules.

Categories and keywords live in a plain-text file (default: categories.txt next to this script):

    [Food]
    STARBUCKS
    [Bill/Entertainment]
    NETFLIX.COM

Usage:
    python categorize_transactions.py transactions.csv
    python categorize_transactions.py transactions.csv -o categorized.csv --interactive

Keywords match case-insensitively as whole words; the longest matching keyword wins.
With --interactive, unknown merchants are asked about once and the answer is appended to the
rules file so it is never asked again.
"""
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

UNCATEGORIZED = "Uncategorized"
PAYMENT = "Payment"  # card payments are excluded from spending
SEPARATOR = "/"  # section names are "Category" or "Category/Subcategory"
DEFAULT_RULES_FILE = Path(__file__).with_name("categories.txt")


def parse_rules(text: str) -> Dict[str, List[str]]:
    """Parse rules text into {label: [UPPER-CASE KEYWORDS]}, keeping section order.

    Parameters:
        text: Contents of a rules file. ``[Label]`` lines start a section, other non-blank,
            non-``#`` lines are keywords for the current section.

    Returns:
        Mapping of section label (e.g. "Bill/Entertainment") to its sorted unique keywords.

    Raises:
        ValueError: If a keyword appears before any ``[Label]`` header.
    """
    rules: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            if not current:
                raise ValueError(f"Line {number}: empty section name")
            rules.setdefault(current, [])
        elif current is None:
            raise ValueError(f"Line {number}: keyword '{line}' appears before any [Category]")
        else:
            rules[current].append(line.upper())
    return {label: sorted(set(words)) for label, words in rules.items()}


def load_rules(rules_file: Path = DEFAULT_RULES_FILE) -> Dict[str, List[str]]:
    """Read and parse the rules file."""
    if not rules_file.exists():
        raise FileNotFoundError(f"Rules file not found: {rules_file}")
    return parse_rules(rules_file.read_text())


def category_labels(rules: Dict[str, List[str]]) -> List[str]:
    """Labels the user can assign, in file order (the special Payment section is excluded)."""
    return [label for label in rules if label != PAYMENT]


def add_keyword(rules_file: Path, label: str, keyword: str) -> None:
    """Append ``keyword`` under ``[label]`` in the rules file, keeping comments and layout.

    Creates the section at the end of the file if it doesn't exist yet.
    """
    lines = rules_file.read_text().splitlines()
    header = f"[{label}]"
    keyword = keyword.strip().upper()
    if header not in [line.strip() for line in lines]:
        lines += ["", header, keyword]
    else:
        index = [line.strip() for line in lines].index(header) + 1
        while index < len(lines) and lines[index].strip() and not lines[index].startswith("["):
            index += 1
        lines.insert(index, keyword)
    rules_file.write_text("\n".join(lines) + "\n")


def keyword_pattern(keyword: str) -> re.Pattern:
    """Regex matching ``keyword`` only when it is not embedded in a larger word."""
    return re.compile(rf"(?<![A-Z0-9]){re.escape(keyword)}(?![A-Z0-9])")


def build_matchers(rules: Dict[str, List[str]]) -> List[tuple]:
    """Compile rules into (regex, label, keyword) tuples, longest keyword first.

    A keyword only matches when it is not embedded in a larger word, so ``ROSS`` will not
    match ``CROSSFIT``.
    """
    matchers = []
    for label, keywords in rules.items():
        for keyword in keywords:
            matchers.append((keyword_pattern(keyword), label, keyword))
    return sorted(matchers, key=lambda m: len(m[2]), reverse=True)


def categorize_description(description: str, matchers: List[tuple]) -> str:
    """Return the label of the longest keyword found in ``description``."""
    text = str(description).upper()
    for pattern, label, _ in matchers:
        if pattern.search(text):
            return label
    return UNCATEGORIZED


US_STATES = set(
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ "
    "NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split()
)
# Multi-word cities that appear before a state code; longest are tried first.
KNOWN_CITIES = [
    "SOUTH SAN FRANCISCO", "SAN FRANCISCO", "SAN JOSE", "SAN DIEGO", "SAN MATEO", "SAN BRUNO",
    "MOUNTAIN VIEW", "PALO ALTO", "DALY CITY", "REDWOOD CITY", "SANTA CLARA", "SANTA CRUZ",
    "LOS ANGELES", "LAS VEGAS", "NEW YORK CITY", "NEW YORK", "SALT LAKE CITY",
]
PROCESSOR_PREFIX = re.compile(r"^(?:(?:SQ|TST|PY|PP|PAYPAL|SP|DD)\s*\*\s*|SP\s+|THE\s+)")
CARD_MASK = re.compile(r"\bNULL\b|X{4,}\d*")
DOMAIN_LIKE = re.compile(r"^[A-Z0-9-]+\.(?:COM|ORG|NET|IO|AI|CO|APP|TV|EDU)$")
STREET_WORDS = {"ST", "AVE", "BLVD", "RD", "DR", "HWY"}


def _strip_location(words: List[str]) -> List[str]:
    """Drop a trailing "<city> <state>" from a list of words, keeping at least one word."""
    if len(words) < 2 or words[-1] not in US_STATES:
        return words
    words = words[:-1]
    for city in sorted(KNOWN_CITIES, key=len, reverse=True):
        city_words = city.split()
        if len(words) > len(city_words) and words[-len(city_words):] == city_words:
            return words[:-len(city_words)]
    return words[:-1] if len(words) > 1 else words  # unknown city: drop the word before the state


def suggest_keyword(description: str) -> str:
    """Suggest a short, reusable keyword for a raw transaction description.

    Strips card masks, payment-processor prefixes (``SQ *``, ``TST*``, ``SP``), a leading
    "THE", store numbers and everything after them, and a trailing "city state". Domain-like
    names (``RESUME.IO``) are kept whole; anything else is cut to the first two words. The
    suggestion is guaranteed to match ``description``, otherwise the first two words are used.
    """
    text = str(description).upper()
    cleaned = PROCESSOR_PREFIX.sub("", CARD_MASK.sub(" ", text).strip())
    words = cleaned.split()
    # Cut at the first store number / phone number / reference code (never the first word).
    for index, word in enumerate(words):
        if index > 0 and any(c.isdigit() for c in word):
            words = words[:index]
            break
    words = _strip_location(words)
    words = [w for w in words if re.search(r"[A-Z0-9]", w) and w not in STREET_WORDS]

    if not words:
        words = text.split()
    keyword = words[0] if words and DOMAIN_LIKE.match(words[0]) else " ".join(words[:2])
    if not keyword_pattern(keyword).search(text):
        keyword = " ".join(text.split()[:2])
    return keyword


def ask_category(description: str, labels: List[str]) -> Optional[str]:
    """Prompt until the user picks a label by number; return None to skip."""
    menu = "\n".join(f"  {i}: {label}" for i, label in enumerate(labels, start=1))
    while True:
        answer = input(f"\n{description}\n{menu}\n  [number, Enter to skip] > ").strip()
        if answer == "":
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(labels):
            return labels[int(answer) - 1]
        print(f"  Please enter 1-{len(labels)} or press Enter to skip.")


def learn_unknown_merchants(df: pd.DataFrame, rules_file: Path) -> None:
    """Interactively ask about each unknown merchant once and append the answers to the file."""
    labels = category_labels(load_rules(rules_file))
    seen = set()
    for description in df.loc[df["category"] == UNCATEGORIZED, "description"]:
        suggestion = suggest_keyword(description)
        if suggestion in seen:
            continue
        seen.add(suggestion)
        label = ask_category(description, labels)
        if label is None:
            continue
        keyword = input(f"  Keyword to remember [{suggestion}] > ").strip() or suggestion
        add_keyword(rules_file, label, keyword)


def load_transactions(csv_path: Path) -> pd.DataFrame:
    """Read a transactions CSV and normalize it to date/description/debit/credit columns.

    Parameters:
        csv_path: CSV with at least Date and Description columns; Debit and Credit optional.

    Returns:
        DataFrame with lower-case column names, a datetime ``date`` and numeric amounts.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"date", "description"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")

    df["date"] = pd.to_datetime(df["date"])
    for col in ("debit", "credit"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    return df


def apply_rules(df: pd.DataFrame, rules: Dict[str, List[str]]) -> pd.DataFrame:
    """Categorize a frame from ``load_transactions`` and return it in output column order.

    Output columns: date, description, category, subcategory, kind, debit, credit, amount, where
    ``amount`` is spending minus refunds (credits are negative in card exports).
    """
    df = df.copy()
    matchers = build_matchers(rules)
    labels = df["description"].apply(lambda d: categorize_description(d, matchers))
    parts = labels.str.partition(SEPARATOR)
    df["category"], df["subcategory"] = parts[0], parts[2]
    df["kind"] = "onetime"
    df["amount"] = df["debit"].fillna(0) + df["credit"].fillna(0)
    columns = [
        "date", "description", "category", "subcategory", "kind", "debit", "credit", "amount",
    ]
    return df[columns].sort_values(["date", "description"], ascending=[False, True])


def categorize(
    csv_path: Path, rules_file: Path = DEFAULT_RULES_FILE, interactive: bool = False
) -> pd.DataFrame:
    """Categorize every transaction in ``csv_path``, optionally asking about unknown merchants."""
    df = load_transactions(csv_path)
    result = apply_rules(df, load_rules(rules_file))
    if interactive and (result["category"] == UNCATEGORIZED).any():
        learn_unknown_merchants(result, rules_file)
        result = apply_rules(df, load_rules(rules_file))
    return result


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Return total spending per category, excluding card payments, with a Total row."""
    spending = df[df["category"] != PAYMENT]
    summary = (
        spending.groupby("category")["amount"].agg(total="sum", transactions="count")
        .sort_values("total", ascending=False)
        .round(2)
    )
    summary.loc["Total"] = [summary["total"].sum().round(2), summary["transactions"].sum()]
    return summary.astype({"transactions": int})


def summarize_subcategories(df: pd.DataFrame) -> pd.DataFrame:
    """Return spending per category/subcategory for categories that have subcategories."""
    spending = df[df["category"] != PAYMENT]
    has_subs = spending.groupby("category")["subcategory"].transform(lambda s: (s != "").any())
    spending = spending[has_subs]
    return (
        spending.assign(subcategory=spending["subcategory"].replace("", "(general)"))
        .groupby(["category", "subcategory"])["amount"].agg(total="sum", transactions="count")
        .sort_values("total", ascending=False)
        .round(2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Categorize transactions in a CSV.")
    parser.add_argument("input", type=Path, help="transactions CSV")
    parser.add_argument("-o", "--output", type=Path,
                        help="output CSV (default: <input>_categorized.csv)")
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_FILE,
                        help="categories text file (default: categories.txt)")
    parser.add_argument("--interactive", action="store_true",
                        help="ask about unknown merchants")
    args = parser.parse_args()

    output = args.output or args.input.with_name(f"{args.input.stem}_categorized.csv")
    result = categorize(args.input, args.rules, args.interactive)
    result.assign(date=result["date"].dt.strftime("%Y-%m-%d")).to_csv(output, index=False)

    print(summarize(result).to_string())
    unknown = int((result["category"] == UNCATEGORIZED).sum())
    print(f"\nWrote {len(result)} rows to {output} ({unknown} uncategorized)")


if __name__ == "__main__":
    main()
