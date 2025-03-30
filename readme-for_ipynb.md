# Financial Data Analysis Toolkit

## Overview

This Python toolkit provides a collection of functions for analyzing personal financial data from CSV files. It focuses on expense tracking, spending pattern analysis, and budget planning, with special features for recurring expenses.

## Key Features

- **Monthly Spending Analysis**: Track and project spending over a month
- **Spending Reduction Analysis**: Find opportunities to reduce expenses
- **Weekly Spending Analysis**: Analyze patterns by week and day of week
- **Day/Week Column Generation**: Add time-based columns to financial data
- **Recurring Expense Tracking**: Extract and track non-one-time expenses

## Requirements

- Python 3.6+
- pandas
- numpy
- matplotlib
- scikit-learn (for prediction models)

## Function Documentation

### print_non_onetime_rows

```python
print_non_onetime_rows(csv_file, output_file='recurring.csv')
```

Identifies and extracts all non-one-time expenses from a financial CSV file, saving them to a dedicated tracking file.

**Parameters:**
- `csv_file`: Path to the CSV file containing financial data
- `output_file`: Path where the output CSV will be saved (default: 'recurring.csv')

**Returns:**
- A pandas DataFrame with the filtered rows

**Features:**
- Filters out rows where the "kind" column equals "onetime"
- Prints detailed information about each non-one-time expense
- Calculates and displays the total amount for all recurring expenses
- Appends results to an existing tracking file or creates a new one
- Automatically removes duplicate entries when updating the tracking file

### analyze_monthly_spending

```python
analyze_monthly_spending(csv_file, month_days=None, output_path=None)
```

Analyzes monthly spending patterns and makes projections for the remainder of the month.

### analyze_spending_reduction

```python
analyze_spending_reduction(csv_file, discretionary_categories=None, essential_categories=None, reduction_target=0.2, output_path=None)
```

Analyzes spending data and provides recommendations for reducing expenses.

### analyze_weekly_spending

```python
analyze_weekly_spending(csv_file, discretionary_categories=None, weekly_savings_target=0.10, daily_reduction_target=0.20, category_reduction_target=0.25, output_path=None)
```

Analyzes spending patterns by week and day of week, providing reduction strategies.

### add_day_week_columns

```python
add_day_week_columns(csv_file, date_format='%m/%d/%y', output_file='added_column_finances.csv', print_preview=True)
```

Adds day and week columns to financial data and saves to a new CSV file.

## Usage Examples

### Basic Usage

```python
# Track recurring expenses
print_non_onetime_rows('finances_march.csv')

# Analyze monthly spending
results = analyze_monthly_spending('finances_march.csv')

# Find ways to reduce spending
results = analyze_spending_reduction('finances_march.csv')

# Analyze weekly patterns
results = analyze_weekly_spending('finances_march.csv')

# Add time-based columns
result_df = add_day_week_columns('finances_march.csv')
```

### Advanced Usage

```python
# Custom output path for recurring expenses
print_non_onetime_rows('finances_march.csv', output_file='custom_recurring.csv')

# Custom categories and reduction targets
results = analyze_spending_reduction(
    'finances_march.csv',
    discretionary_categories=['Dining', 'Entertainment', 'Shopping', 'Subscriptions'],
    essential_categories=['Groceries', 'Utilities', 'Rent', 'Transportation'],
    reduction_target=0.25,
    output_path='./reports'
)
```

## CSV File Format

The toolkit expects CSV files with the following columns:
- `Date`: Transaction date (e.g., "03/15/24")
- `description`: Transaction description
- `category`: Spending category (e.g., "Dining", "Groceries")
- `kind`: Transaction type ("onetime" or other values for recurring)
- `Debit`: Transaction amount (positive values)

Other columns may be present and will be preserved in the output files.

## Notes

- The toolkit automatically handles date parsing and formats
- Visual charts are generated for spending analysis when matplotlib is available
- Duplicate detection prevents redundant entries in the recurring expenses tracker
- All functions provide detailed console output for transparency

---

Created with ❤️ for personal finance management
