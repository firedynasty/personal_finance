# Personal Finance Analyzer

A Streamlit application for analyzing personal financial data, spotting trends, and finding opportunities to reduce expenses.

## Features

- **Monthly Spending Analysis**: Visualize spending patterns and project your end-of-month total
- **Spending Reduction**: Identify discretionary vs. essential spending and get personalized savings recommendations
- **Weekly Analysis**: Track spending patterns by week and day to find optimization opportunities
- **Data Utilities**: Enhance your financial data with day/week information and extract recurring expenses

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/personal-finance-analyzer.git
   cd personal-finance-analyzer
   ```

2. Install the required dependencies:
   ```
   pip install streamlit pandas matplotlib numpy scikit-learn
   ```

3. Run the app:
   ```
   streamlit run finance_analyzer.py
   ```

## Preparing Your Data

This app works with financial data in CSV format. To get the most out of the analysis:

1. **Download transactions from your bank**
   - Most banks offer CSV export of your transactions
   - Typically found in your online banking portal under "Transaction History" or "Statement"

2. **Add required columns**
   - **kind**: Categorizes the transaction frequency
     - `onetime`: One-time purchases or non-recurring expenses
     - `monthly`: Bills that occur every month (subscriptions, rent, utilities)
     - `yearly`: Annual expenses (insurance premiums, memberships)
   
   - **category**: Classifies the transaction type
     - Examples: Entertainment, Food, Bills, Transportation, Groceries, Shopping
     - Be consistent with your categories for better analysis

3. **Data format requirements**
   - The app expects these columns:
     - `Date`: Transaction date
     - `Debit`: Amount spent (positive number)
     - `category`: Your spending categories
     - `kind`: Transaction frequency (onetime, monthly, yearly)

## Example Data Format

| Date       | Description                | category       | kind     | Debit   | Credit |
|------------|----------------------------|----------------|----------|---------|--------|
| 3/15/2024  | UBER RIDE                  | Transportation | onetime  | 24.50   | 0      |
| 3/15/2024  | NETFLIX SUBSCRIPTION       | Entertainment  | monthly  | 15.99   | 0      |
| 3/16/2024  | GROCERY OUTLET             | Groceries      | onetime  | 87.32   | 0      |
| 3/17/2024  | RENT PAYMENT               | Housing        | monthly  | 1800.00 | 0      |
| 3/18/2024  | STARBUCKS                  | Food           | onetime  | 5.75    | 0      |
| 3/20/2024  | CAR INSURANCE              | Insurance      | yearly   | 120.00  | 0      |

## Tips for Better Analysis

1. **Consistent categorization**: Use the same category names for similar expenses
2. **Regular updates**: Add new transactions regularly for the most accurate projections
3. **Verify recurring expenses**: Ensure all subscriptions and regular bills are marked as `monthly` or `yearly`
4. **Review uncategorized items**: Make sure all transactions have appropriate categories

## Troubleshooting

- **CSV format issues**: Make sure your CSV has the correct column names and formatting
- **Date parsing errors**: Try adjusting the date format in the "Add Day/Week Columns" utility
- **Missing data**: Check that required columns (Date, Debit) are present in your CSV

## License

This project is licensed under the MIT License - see the LICENSE file for details.
