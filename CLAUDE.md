# Personal Finance Analyzer Guidelines

## Running the Application
- Start application: `streamlit run personal_finance_1.py`
- Debug mode: `streamlit run personal_finance_1.py --logger.level=debug`
- Test with sample data: `streamlit run personal_finance_1.py -- --demo`

## Code Style Guidelines
- **Imports**: Standard libraries first, then third-party packages, then local modules
- **Formatting**: Follow PEP 8 (4 spaces indentation, max 100 chars per line)
- **Types**: Use type hints for function parameters and return values
- **Naming**:
  - Classes: CamelCase (e.g., `FinancialAnalyzer`)
  - Functions/methods: snake_case (e.g., `analyze_monthly_spending`)
  - Variables: snake_case (e.g., `monthly_data`)
  - Constants: UPPERCASE (e.g., `DEFAULT_DATE_FORMAT`)
- **Documentation**: Use docstrings with parameters and return values
- **Error Handling**: Use try/except for expected errors, validate inputs with descriptive error messages

## Data Processing Conventions
- Use pandas for all data manipulation
- Always make a copy of dataframes before modifying them
- Convert 'Date' column to datetime when loading data
- Handle missing values explicitly
- Use meaningful column names with snake_case