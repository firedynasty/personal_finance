import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
from sklearn.linear_model import LinearRegression
import io
import base64


class FinancialAnalyzer:
    def __init__(self):
        self.data = None
        self.file_name = None
    
    def load_data(self, file):
        """Load data from uploaded file or file path"""
        if isinstance(file, str):  # File path
            self.data = pd.read_csv(file)
            self.file_name = os.path.basename(file)
        else:  # Uploaded file
            self.data = pd.read_csv(file)
            self.file_name = file.name
        
        # Convert Date column to datetime
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        return self.data
    
    def analyze_monthly_spending(self, month_days=None, output_path=None):
        """
        Analyze monthly spending data and project future spending
        
        Parameters:
        -----------
        month_days : int, optional
            Number of days in the month being analyzed. If None, will be determined from data
        output_path : str, optional
            Path to save visualizations. If None, will display without saving
            
        Returns:
        --------
        dict
            Dictionary containing analysis results
        """
        # Make a copy of the data to avoid modifying the original
        monthly_data = self.data.copy()
        
        # Determine days in month if not provided
        if month_days is None:
            # Get the month from the data
            current_month = monthly_data['Date'].dt.month.value_counts().idxmax()
            current_year = monthly_data['Date'].dt.year.value_counts().idxmax()
            
            # Calculate days in this month
            if current_month in [4, 6, 9, 11]:
                month_days = 30
            elif current_month == 2:
                # Check for leap year
                if (current_year % 4 == 0 and current_year % 100 != 0) or (current_year % 400 == 0):
                    month_days = 29
                else:
                    month_days = 28
            else:
                month_days = 31
        
        # Add day-of-month and day-of-week features
        monthly_data['day'] = monthly_data['Date'].dt.day
        monthly_data['day_of_week'] = monthly_data['Date'].dt.dayofweek
        monthly_data['day_name'] = monthly_data['Date'].dt.day_name()

        # Analyze spending by day
        daily_spending = monthly_data.groupby('day')['Debit'].sum().reset_index()
        
        # Calculate cumulative spending
        daily_spending['cumulative'] = daily_spending['Debit'].cumsum()
        
        # Initialize results dictionary
        results = {
            "total_spending": daily_spending['Debit'].sum(),
            "daily_average": daily_spending['Debit'].mean(),
            "transaction_count": len(monthly_data),
            "days_analyzed": len(daily_spending),
            "highest_day": daily_spending.loc[daily_spending['Debit'].idxmax(), 'day']
        }
        
        # Simple prediction for remaining days (if not end of month)
        current_day = daily_spending['day'].max()
        
        if current_day < month_days:
            # Fit a simple trend line to existing data
            X = daily_spending[['day']]
            y = daily_spending['cumulative']
            model = LinearRegression().fit(X, y)
            
            # Predict for remaining days
            future_days = np.array(range(current_day + 1, month_days + 1)).reshape(-1, 1)
            predicted_cumulative = model.predict(future_days)
            
            # Projected spending at month end
            projected_total = predicted_cumulative[-1] if len(predicted_cumulative) > 0 else daily_spending['cumulative'].iloc[-1]
            additional_spending = projected_total - daily_spending['cumulative'].iloc[-1]
            
            # Add projection results
            results["projected_total"] = projected_total
            results["current_spending"] = daily_spending['cumulative'].iloc[-1]
            results["projected_additional"] = additional_spending
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(daily_spending['day'], daily_spending['cumulative'], 'bo-', label='Actual Spending')
            ax.plot(future_days, predicted_cumulative, 'ro--', label='Predicted Spending')
            ax.set_xlabel('Day of Month')
            ax.set_ylabel('Cumulative Spending ($)')
            ax.set_title('Monthly Spending Projection')
            ax.legend()
            ax.grid(True)
            
            # Return the figure for Streamlit to display
            results["projection_chart"] = fig
        else:
            results["projected_total"] = daily_spending['cumulative'].iloc[-1]
            results["current_spending"] = daily_spending['cumulative'].iloc[-1]
            results["projected_additional"] = 0
        
        # Category-based analysis
        if 'category' in monthly_data.columns:
            category_spending = monthly_data.groupby('category')['Debit'].agg(['sum', 'count', 'mean']).reset_index()
            category_spending = category_spending.sort_values('sum', ascending=False)
            
            # Add category insights
            results["top_categories"] = category_spending.head(3)[['category', 'sum']].to_dict('records')
            results["category_breakdown"] = category_spending.to_dict('records')
            
            # Create category visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(category_spending.head(10)['category'], category_spending.head(10)['sum'])
            ax.set_xlabel('Category')
            ax.set_ylabel('Total Spending ($)')
            ax.set_title('Top 10 Spending Categories')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            results["category_chart"] = fig
        
        # Day of week analysis
        day_of_week_spending = monthly_data.groupby(['day_of_week', 'day_name'])['Debit'].sum().reset_index()
        day_of_week_spending = day_of_week_spending.sort_values('day_of_week')
        results["day_of_week_spending"] = day_of_week_spending.to_dict('records')
        
        # Create day of week visualization
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_data = day_of_week_spending.set_index('day_name')['Debit'].reindex(day_order).fillna(0)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(day_data.index, day_data.values)
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Total Spending ($)')
        ax.set_title('Spending by Day of Week')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        results["day_of_week_chart"] = fig
        
        # Generate weekly analysis if we have enough data
        if len(daily_spending) >= 7:
            # Add week number
            monthly_data['week'] = ((monthly_data['day'] - 1) // 7) + 1
            weekly_spending = monthly_data.groupby('week')['Debit'].sum().reset_index()
            results["weekly_spending"] = weekly_spending.to_dict('records')
            
            # Create weekly spending visualization
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(weekly_spending['week'], weekly_spending['Debit'])
            ax.set_xlabel('Week of Month')
            ax.set_ylabel('Total Spending ($)')
            ax.set_title('Weekly Spending Pattern')
            ax.set_xticks(weekly_spending['week'])
            ax.grid(axis='y')
            
            results["weekly_chart"] = fig
        
        return results
    
    def analyze_spending_reduction(self, discretionary_categories=None, essential_categories=None, reduction_target=0.2):
        """
        Analyze spending data and provide recommendations for reducing expenses
        
        Parameters:
        -----------
        discretionary_categories : list, optional
            List of category names considered discretionary spending
        essential_categories : list, optional
            List of category names considered essential spending
        reduction_target : float, optional
            Target percentage reduction for discretionary spending (0.2 = 20%)
            
        Returns:
        --------
        dict
            Dictionary containing analysis results and savings recommendations
        """
        # Default discretionary and essential categories if not provided
        if discretionary_categories is None:
            discretionary_categories = ['Dining', 'Entertainment', 'Shopping']
        
        if essential_categories is None:
            essential_categories = ['Groceries', 'Utilities', 'Rent', 'Bills']
        
        # Make a copy of the data to avoid modifying the original
        monthly_data = self.data.copy()
        
        # Add day of week
        monthly_data['day_of_week'] = monthly_data['Date'].dt.dayofweek
        monthly_data['day_name'] = monthly_data['Date'].dt.day_name()
        
        # Identify top spending categories
        if 'category' in monthly_data.columns:
            category_totals = monthly_data.groupby('category')['Debit'].sum()
            top_categories = category_totals.sort_values(ascending=False)
            
            # Find discretionary vs. essential spending
            # Filter to only include categories that exist in the data
            existing_disc_categories = [cat for cat in discretionary_categories if cat in category_totals.index]
            existing_ess_categories = [cat for cat in essential_categories if cat in category_totals.index]
            
            # If none of the predefined categories exist in the data, identify top 3 as discretionary
            if not existing_disc_categories:
                existing_disc_categories = top_categories.head(3).index.tolist()
                st.warning(f"None of the predefined discretionary categories found. Using top 3 categories instead: {existing_disc_categories}")
            
            # Calculate discretionary and essential spending
            discretionary_spending = monthly_data[monthly_data['category'].isin(existing_disc_categories)]['Debit'].sum()
            essential_spending = monthly_data[monthly_data['category'].isin(existing_ess_categories)]['Debit'].sum()
        else:
            st.warning("No 'category' column found in data. Unable to analyze spending by category.")
            # Create empty values if category doesn't exist
            category_totals = pd.Series(dtype='float64')
            top_categories = pd.Series(dtype='float64')
            existing_disc_categories = []
            existing_ess_categories = []
            discretionary_spending = 0
            essential_spending = 0
        
        # Identify spending patterns by day of week
        spending_by_day = monthly_data.groupby(['day_of_week', 'day_name'])['Debit'].sum().reset_index()
        spending_by_day = spending_by_day.sort_values('day_of_week')
        
        # Calculate potential savings
        potential_savings = discretionary_spending * reduction_target
        
        # Create result dictionary
        results = {
            "total_spending": monthly_data['Debit'].sum(),
            "discretionary_spending": discretionary_spending,
            "essential_spending": essential_spending,
            "potential_savings": potential_savings,
            "discretionary_categories": existing_disc_categories,
            "category_breakdown": top_categories.to_dict() if not top_categories.empty else {},
            "spending_by_day": spending_by_day.set_index('day_name')['Debit'].to_dict() if not spending_by_day.empty else {},
            "savings_recommendations": []
        }
        
        # Generate visualization of spending distribution and savings targets
        if 'category' in monthly_data.columns and not top_categories.empty:
            # Pie chart of spending categories
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
            
            # Create custom labels with both percentage and amount
            def make_autopct(values):
                def my_autopct(pct):
                    total = sum(values)
                    val = pct*total/100.0
                    return f'{pct:.1f}%\n(${val:.2f})'
                return my_autopct
        
            top_categories.head(5).plot.pie(autopct=make_autopct(top_categories.head(5)), ax=ax1)
            ax1.set_title('Top 5 Spending Categories')
            ax1.set_ylabel('')
        
            # Bar chart of day-of-week spending patterns
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            spending_by_dow = spending_by_day.set_index('day_name')['Debit']
            spending_by_dow = spending_by_dow.reindex(day_order, fill_value=0)
            
            if not spending_by_dow.empty:
                spending_by_dow.plot.bar(ax=ax2)
                ax2.set_title('Spending by Day of Week')
                ax2.set_xlabel('')
                ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            results["spending_distribution_chart"] = fig
        
        # Generate savings recommendations for each discretionary category
        category_savings = []
        for category in existing_disc_categories:
            cat_total = monthly_data[monthly_data['category'] == category]['Debit'].sum()
            suggested_reduction = cat_total * reduction_target
            target_amount = cat_total - suggested_reduction
            
            category_savings.append({
                "category": category,
                "current_amount": cat_total,
                "target_amount": target_amount,
                "savings": suggested_reduction
            })
        
        results["category_savings"] = category_savings
        
        # Find high-spending days
        average_daily = spending_by_day['Debit'].mean() if not spending_by_day.empty else 0
        high_spending_days = []
        
        for _, row in spending_by_day.iterrows():
            if row['Debit'] > average_daily:
                high_spending_days.append({
                    "day": row['day_name'],
                    "amount": row['Debit'],
                    "above_average": row['Debit'] - average_daily
                })
        
        results["high_spending_days"] = high_spending_days
        
        return results
    
    def analyze_weekly_spending(self, discretionary_categories=None, weekly_savings_target=0.10,
                               daily_reduction_target=0.20, category_reduction_target=0.25):
        """
        Analyze spending patterns by week and day of week, providing reduction strategies
        
        Parameters:
        -----------
        discretionary_categories : list, optional
            List of category names considered discretionary spending
        weekly_savings_target : float, optional
            Target percentage reduction for weekly spending (0.10 = 10%)
        daily_reduction_target : float, optional
            Target percentage reduction for highest spending day (0.20 = 20%)
        category_reduction_target : float, optional
            Target percentage reduction for top discretionary category (0.25 = 25%)
            
        Returns:
        --------
        dict
            Dictionary containing analysis results and reduction strategies
        """
        # Default discretionary categories if not provided
        if discretionary_categories is None:
            discretionary_categories = ['Dining', 'Entertainment', 'Shopping']
        
        # Make a copy of the data to avoid modifying the original
        monthly_data = self.data.copy()
        
        # Add week number and day of week
        monthly_data['day_of_week'] = monthly_data['Date'].dt.dayofweek
        monthly_data['day_name'] = monthly_data['Date'].dt.day_name()
        
        # Calculate week of month (ensuring first week of month is week 1)
        try:
            # Try the isocalendar method first (newer pandas versions)
            first_day_of_month = monthly_data['Date'].dt.to_period('M').dt.start_time
            monthly_data['week_of_month'] = monthly_data['Date'].dt.isocalendar().week - \
                                            first_day_of_month.dt.isocalendar().week + 1
        except AttributeError:
            # Fallback for older pandas versions
            monthly_data['week_of_month'] = ((monthly_data['Date'].dt.day - 1) // 7) + 1
        
        # Weekly spending analysis
        weekly_spending = monthly_data.groupby('week_of_month')['Debit'].sum()
        weekly_avg = weekly_spending.mean() if not weekly_spending.empty else 0
        
        # Day of week analysis
        daily_spending = monthly_data.groupby(['day_of_week', 'day_name'])['Debit'].sum().reset_index()
        daily_spending = daily_spending.sort_values('day_of_week')
        daily_avg = daily_spending['Debit'].mean() if not daily_spending.empty else 0
        
        # Category spending by week (if category column exists)
        if 'category' in monthly_data.columns:
            category_by_week = monthly_data.pivot_table(
                index='category', 
                columns='week_of_month', 
                values='Debit', 
                aggfunc='sum', 
                fill_value=0
            ).reset_index()
        else:
            category_by_week = pd.DataFrame()
        
        # Create results dictionary
        results = {
            "total_spending": monthly_data['Debit'].sum(),
            "weekly_spending": weekly_spending.to_dict() if not weekly_spending.empty else {},
            "weekly_average": weekly_avg,
            "daily_spending": daily_spending.set_index('day_name')['Debit'].to_dict() if not daily_spending.empty else {},
            "daily_average": daily_avg,
            "category_by_week": category_by_week.to_dict() if not category_by_week.empty else {},
            "high_spending_weeks": [],
            "high_spending_days": [],
        }
        
        # Create visualizations
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        
        # Weekly spending
        if not weekly_spending.empty:
            axs[0].bar(weekly_spending.index, weekly_spending.values)
            axs[0].axhline(y=weekly_avg, color='r', linestyle='-', label=f'Weekly Avg: ${weekly_avg:.2f}')
            axs[0].set_xlabel('Week of Month')
            axs[0].set_ylabel('Total Spending ($)')
            axs[0].set_title('Weekly Spending Pattern')
            axs[0].legend()
        
        # Daily spending
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        # Create a mapping from day_name to Debit for easy plotting in correct order
        day_mapping = daily_spending.set_index('day_name')['Debit'].to_dict() if not daily_spending.empty else {}
        # Get values for all days, using 0 for days not in the data
        day_values = [day_mapping.get(day, 0) for day in day_order]
        
        axs[1].bar(day_order, day_values)
        axs[1].set_xlabel('Day of Week')
        axs[1].set_ylabel('Total Spending ($)')
        axs[1].set_title('Spending by Day of Week')
        axs[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        results["weekly_analysis_chart"] = fig
        
        # Find high-spending weeks
        high_weeks = []
        for week, amount in weekly_spending.items():
            if amount > weekly_avg:
                # Find categories driving higher spending
                if 'category' in monthly_data.columns:
                    high_week_categories = monthly_data[monthly_data['week_of_month'] == week].groupby('category')['Debit'].sum().sort_values(ascending=False).head(3)
                    top_cats = high_week_categories.to_dict()
                else:
                    top_cats = {}
                
                high_weeks.append({
                    "week": int(week),
                    "amount": amount,
                    "above_average": amount - weekly_avg,
                    "top_categories": top_cats
                })
        
        results["high_spending_weeks"] = high_weeks
        
        # Find high-spending days
        high_days = []
        for _, row in daily_spending.sort_values('Debit', ascending=False).iterrows():
            day_name = row['day_name']
            amount = row['Debit']
            
            if amount > daily_avg:
                # Find categories driving higher spending on this day
                if 'category' in monthly_data.columns:
                    day_categories = monthly_data[monthly_data['day_name'] == day_name].groupby('category')['Debit'].sum().sort_values(ascending=False).head(3)
                    top_cats = day_categories.to_dict()
                else:
                    top_cats = {}
                
                high_days.append({
                    "day": day_name,
                    "amount": amount,
                    "above_average": amount - daily_avg,
                    "top_categories": top_cats
                })
        
        results["high_spending_days"] = high_days
        
        # Calculate reduction strategies
        target_weekly_saving = weekly_avg * weekly_savings_target
        results["weekly_savings_target"] = target_weekly_saving
        
        # Focus on highest day for reduction
        if not daily_spending.empty:
            highest_day = daily_spending.sort_values('Debit', ascending=False).iloc[0]
            highest_day_name = highest_day['day_name']
            highest_day_amount = highest_day['Debit']
            
            results["day_reduction_strategy"] = {
                "day": highest_day_name,
                "current_amount": highest_day_amount,
                "target_amount": highest_day_amount * (1 - daily_reduction_target),
                "savings": highest_day_amount * daily_reduction_target
            }
        
        # Get top discretionary category (if category column exists)
        if 'category' in monthly_data.columns:
            discretionary_cats_in_data = [cat for cat in discretionary_categories if cat in monthly_data['category'].unique()]
            
            if discretionary_cats_in_data:
                disc_spending = monthly_data[monthly_data['category'].isin(discretionary_cats_in_data)]
                if not disc_spending.empty:
                    top_disc = disc_spending.groupby('category')['Debit'].sum().sort_values(ascending=False).head(1)
                    if not top_disc.empty:
                        top_disc_cat = top_disc.index[0]
                        top_disc_amount = top_disc.values[0]
                        weekly_disc_amount = top_disc_amount / len(weekly_spending) if not weekly_spending.empty and len(weekly_spending) > 0 else top_disc_amount
                        
                        results["category_reduction_strategy"] = {
                            "category": top_disc_cat,
                            "total_amount": top_disc_amount,
                            "weekly_amount": weekly_disc_amount,
                            "target_weekly": weekly_disc_amount * (1 - category_reduction_target),
                            "weekly_savings": weekly_disc_amount * category_reduction_target
                        }
        
        return results
    
    def print_non_onetime_rows(self, output_file='recurring.csv'):
        """
        Find all rows from a CSV file where the 'kind' column is not equal to 'onetime'
        and save them to an output CSV file, removing duplicates
        
        Parameters:
        -----------
        output_file : str, optional
            Path where the output CSV will be saved (default: 'recurring.csv')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with filtered rows
        """
        # Make a copy of the data to avoid modifying the original
        df = self.data.copy()
        
        # Check if 'kind' column exists
        if 'kind' not in df.columns:
            st.error("'kind' column not found in the CSV file.")
            return pd.DataFrame()
            
        # Filter rows where kind is not 'onetime'
        filtered_df = df[df['kind'] != 'onetime']
        
        # If no matching rows found
        if filtered_df.empty:
            st.warning("No rows found where 'kind' is not 'onetime'.")
            return pd.DataFrame()
        
        # Calculate total of Debit column if it exists
        total_amount = 0
        if 'Debit' in filtered_df.columns:
            total_amount = filtered_df['Debit'].sum()
        
        # Return the filtered dataframe in case it's needed for further processing
        return filtered_df
    
    def add_day_week_columns(self, date_format='%m/%d/%y'):
        """
        Add day and week columns to financial data
        
        Parameters:
        -----------
        date_format : str, optional
            Format string for parsing the date column
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with added day and week columns
        """
        # Make a copy of the data to avoid modifying the original
        df = self.data.copy()
        
        # Convert Date column to datetime format if it isn't already
        if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'], format=date_format)
        
        # Extract day of the week and prefix with a number for sorting
        df['day'] = df['Date'].dt.dayofweek + 1  # Monday=0, so add 1 to make Monday=1
        day_map = {
            1: '1Monday',
            2: '2Tuesday',
            3: '3Wednesday',
            4: '4Thursday',
            5: '5Friday',
            6: '6Saturday',
            7: '7Sunday'
        }
        df['day'] = df['day'].map(day_map)
        
        # Calculate week of month
        # Method: Week 1 starts on the 1st of the month, Week 2 starts on the 8th, etc.
        df['week'] = ((df['Date'].dt.day - 1) // 7) + 1
        
        return df
    
    def get_download_link(self, df, filename="data.csv", text="Download CSV"):
        """Generate a download link for a dataframe"""
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
        return href

def main():
    st.set_page_config(page_title="Personal Finance Analyzer", layout="wide")
    
    analyzer = FinancialAnalyzer()
    
    # Define welcome HTML content
    welcome_html = """
        <div style="text-align: left; max-width: 800px; margin: 0 auto;">
            <h1>Personal Finance Analyzer</h1>
            <p style="font-size: 1.1rem;">A Streamlit application for analyzing personal financial data, spotting trends, and finding opportunities to reduce expenses.</p>
            <p style="font-size: 1.1rem;"><strong>GitHub Repository:</strong> <a href="https://github.com/firedynasty/personal_finance" target="_blank">https://github.com/firedynasty/personal_finance</a></p>
            <h2>Features</h2>
            <ul style="font-size: 1.05rem; line-height: 1.6;">
                <li><strong>Monthly Spending Analysis</strong>: Visualize spending patterns and project your end-of-month total</li>
                <li><strong>Spending Reduction</strong>: Identify discretionary vs. essential spending and get personalized savings recommendations</li>
                <li><strong>Weekly Analysis</strong>: Track spending patterns by week and day to find optimization opportunities</li>
                <li><strong>Data Utilities</strong>: Enhance your financial data with day/week information and extract recurring expenses</li>
            </ul>
            
            <h2>Preparing Your Data</h2>
            <p>This app works with financial data in CSV format. To get the most out of the analysis:</p>
            <ol style="font-size: 1.05rem; line-height: 1.6;">
                <li>
                    <strong>Download transactions from your bank</strong>
                    <ul>
                        <li>Most banks offer CSV export of your transactions</li>
                        <li>Typically found in your online banking portal under "Transaction History" or "Statement"</li>
                    </ul>
                </li>
                <li>
                    <strong>Add required columns</strong>
                    <ul>
                        <li><strong>kind</strong>: Categorizes the transaction frequency
                            <ul>
                                <li><code>onetime</code>: One-time purchases or non-recurring expenses</li>
                                <li><code>monthly</code>: Bills that occur every month (subscriptions, rent, utilities)</li>
                                <li><code>yearly</code>: Annual expenses (insurance premiums, memberships)</li>
                            </ul>
                        </li>
                        <li><strong>category</strong>: Classifies the transaction type
                            <ul>
                                <li>Examples: Entertainment, Food, Bills, Transportation, Groceries, Shopping</li>
                                <li>Be consistent with your categories for better analysis</li>
                            </ul>
                        </li>
                    </ul>
                </li>
                <li>
                    <strong>Data format requirements</strong>
                    <ul>
                        <li>The app expects these columns:
                            <ul>
                                <li><code>Date</code>: Transaction date</li>
                                <li><code>Debit</code>: Amount spent (positive number)</li>
                                <li><code>category</code>: Your spending categories</li>
                                <li><code>kind</code>: Transaction frequency (onetime, monthly, yearly)</li>
                            </ul>
                        </li>
                    </ul>
                </li>
            </ol>
            
            <h2>Example Data Format</h2>
            <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                <thead>
                    <tr style="background-color: #f2f2f2;">
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Date</th>
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Description</th>
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">category</th>
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">kind</th>
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Debit</th>
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Credit</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px;">3/15/2024</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">UBER RIDE</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">Transportation</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">onetime</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">24.50</td>
                        <td style="border: 1px solid #ddd; padding: 8px;"></td>
                    </tr>
                    <tr style="background-color: #f2f2f2;">
                        <td style="border: 1px solid #ddd; padding: 8px;">3/15/2024</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">NETFLIX SUBSCRIPTION</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">Entertainment</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">monthly</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">15.99</td>
                        <td style="border: 1px solid #ddd; padding: 8px;"></td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px;">3/16/2024</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">GROCERY OUTLET</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">Groceries</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">onetime</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">87.32</td>
                        <td style="border: 1px solid #ddd; padding: 8px;"></td>
                    </tr>
                    <tr style="background-color: #f2f2f2;">
                        <td style="border: 1px solid #ddd; padding: 8px;">3/17/2024</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">RENT PAYMENT</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">Housing</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">monthly</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">1800.00</td>
                        <td style="border: 1px solid #ddd; padding: 8px;"></td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px;">3/18/2024</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">STARBUCKS</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">Food</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">onetime</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">5.75</td>
                        <td style="border: 1px solid #ddd; padding: 8px;"></td>
                    </tr>
                    <tr style="background-color: #f2f2f2;">
                        <td style="border: 1px solid #ddd; padding: 8px;">3/20/2024</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">CAR INSURANCE</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">Insurance</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">yearly</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">120.00</td>
                        <td style="border: 1px solid #ddd; padding: 8px;"></td>
                    </tr>
                </tbody>
            </table>
            
            <h2>Tips for Better Analysis</h2>
            <ol style="font-size: 1.05rem; line-height: 1.6;">
                <li><strong>Consistent categorization</strong>: Use the same category names for similar expenses</li>
                <li><strong>Regular updates</strong>: Add new transactions regularly for the most accurate projections</li>
                <li><strong>Verify recurring expenses</strong>: Ensure all subscriptions and regular bills are marked as <code>monthly</code> or <code>yearly</code></li>
                <li><strong>Review uncategorized items</strong>: Make sure all transactions have appropriate categories</li>
            </ol>
            
            <h2>Troubleshooting</h2>
            <ul style="font-size: 1.05rem; line-height: 1.6;">
                <li><strong>CSV format issues</strong>: Make sure your CSV has the correct column names and formatting</li>
                <li><strong>Date parsing errors</strong>: Try adjusting the date format in the "Add Day/Week Columns" utility</li>
                <li><strong>Missing data</strong>: Check that required columns (Date, Debit) are present in your CSV</li>
            </ul>
            
            <h2>License</h2>
            <p>This project is licensed under the MIT License - see the LICENSE file for details.</p>
            
            <hr style="margin: 30px 0;">
            <p style="font-style: italic; text-align: center;">Please upload your financial data CSV file using the sidebar to begin analysis.</p>
        </div>
    """
    
    # Initialize session state for tracking if welcome page has been shown
    if 'show_welcome' not in st.session_state:
        st.session_state.show_welcome = True
    
    # Function to toggle welcome screen
    def toggle_welcome_screen():
        st.session_state.show_welcome = not st.session_state.show_welcome
    
    # Function to hide welcome screen when file browser is opened
    def on_file_browser_interaction():
        st.session_state.show_welcome = False
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload Financial Data")
        
        uploaded_file = st.file_uploader(
            "Upload a CSV file with your financial data",
            type=["csv"],
            help="Your file should have at least Date and Debit columns",
            on_change=on_file_browser_interaction  # Hide instructions when user interacts with file uploader
        )
        
        # Add a button to show/hide instructions
        if st.session_state.show_welcome:
            button_text = "Hide Instructions"
        else:
            button_text = "Show Instructions"
            
        st.button(button_text, on_click=toggle_welcome_screen)
        
        # Show data info when file is uploaded
        if uploaded_file is not None:
            # Load the data
            data = analyzer.load_data(uploaded_file)
            st.success(f"Loaded {len(data)} rows from {uploaded_file.name}")
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            # Data validation checks
            if 'Date' not in data.columns:
                st.error("Required 'Date' column not found in data!")
            
            if 'Debit' not in data.columns:
                st.error("Required 'Debit' column not found in data!")
    
    # Show the welcome screen if toggle is on
    if st.session_state.show_welcome:
        # Display welcome HTML using components.v1.html for proper rendering
        st.components.v1.html(welcome_html, height=1800, scrolling=True)
    
    # Always show analysis content when a file is uploaded (regardless of welcome screen state)
    if uploaded_file is not None and 'Date' in analyzer.data.columns and 'Debit' in analyzer.data.columns:
        # Display the app title
        st.title("💰 Personal Finance Analyzer")
        
        # Display content in tabs
        tabs = st.tabs(["Monthly Analysis", "Spending Reduction", "Weekly Analysis", "Data Utilities"])
        
        # --- Monthly Spending Analysis Tab ---
        with tabs[0]:
            st.header("Monthly Spending Analysis")
            
            # Run the analysis
            results = analyzer.analyze_monthly_spending()
            
            # Display results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Monthly Summary")
                st.metric("Total Spending", f"${results['total_spending']:.2f}")
                st.metric("Daily Average", f"${results['daily_average']:.2f}")
                st.metric("Transaction Count", results['transaction_count'])
                
                if 'projected_total' in results:
                    st.metric("Projected Total", f"${results['projected_total']:.2f}", 
                              f"${results['projected_additional']:.2f}")
            
            with col2:
                if 'category_breakdown' in results and results['category_breakdown']:
                    st.subheader("Top Spending Categories")
                    for cat in results.get('top_categories', []):
                        st.metric(cat['category'], f"${cat['sum']:.2f}")
            
            # Show charts
            st.subheader("Spending Projections")
            if 'projection_chart' in results:
                st.pyplot(results['projection_chart'])
                
            col3, col4 = st.columns(2)
            
            with col3:
                if 'category_chart' in results:
                    st.subheader("Category Breakdown")
                    st.pyplot(results['category_chart'])
            
            with col4:
                if 'day_of_week_chart' in results:
                    st.subheader("Spending by Day of Week")
                    st.pyplot(results['day_of_week_chart'])
            
            if 'weekly_chart' in results:
                st.subheader("Weekly Spending Pattern")
                st.pyplot(results['weekly_chart'])
        
        # --- Spending Reduction Tab ---
        with tabs[1]:
            st.header("Spending Reduction Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Configure Analysis")
                
                # Get actual categories from data
                available_categories = list(analyzer.data['category'].unique()) if 'category' in analyzer.data.columns else []
                
                # Default discretionary and essential categories
                default_discretionary = ['Dining', 'Entertainment', 'Shopping']
                default_essential = ['Groceries', 'Utilities', 'Rent', 'Bills']
                
                # Filter defaults to only include categories that exist in the data
                available_disc = [cat for cat in default_discretionary if cat in available_categories]
                available_ess = [cat for cat in default_essential if cat in available_categories]
                
                # If none of the defaults exist, use empty lists as defaults
                disc_default = available_disc if available_disc else []
                ess_default = available_ess if available_ess else []
                
                discretionary_cats = st.multiselect(
                    "Discretionary Categories",
                    options=available_categories,
                    default=disc_default
                )
                
                essential_cats = st.multiselect(
                    "Essential Categories",
                    options=available_categories,
                    default=ess_default
                )
                
                reduction_target = st.slider("Reduction Target (%)", 5, 50, 20) / 100
                
                run_reduction = st.button("Run Spending Reduction Analysis")
            
            with col2:
                st.subheader("About Spending Reduction")
                st.write("""
                This analysis identifies discretionary vs. essential spending and recommends ways to reduce expenses.
                - **Discretionary Categories**: Non-essential spending that can be reduced
                - **Essential Categories**: Necessary expenses that are harder to reduce
                - **Reduction Target**: The percentage by which to reduce discretionary spending
                """)
            
            with col3:
                st.image("https://cdn.pixabay.com/photo/2016/10/09/19/19/coins-1726618_960_720.jpg", width=200)
                st.caption("Set a spending reduction goal that is challenging but achievable")
            
            if run_reduction or 'reduction_results' in st.session_state:
                # If first time or button pressed, run the analysis
                if run_reduction or 'reduction_results' not in st.session_state:
                    reduction_results = analyzer.analyze_spending_reduction(
                        discretionary_categories=discretionary_cats,
                        essential_categories=essential_cats,
                        reduction_target=reduction_target
                    )
                    st.session_state.reduction_results = reduction_results
                else:
                    # Use cached results
                    reduction_results = st.session_state.reduction_results
                
                st.divider()
                
                # Display results in a nice format
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Spending", f"${reduction_results['total_spending']:.2f}")
                
                with col2:
                    st.metric("Discretionary Spending", 
                             f"${reduction_results['discretionary_spending']:.2f}",
                             f"{reduction_results['discretionary_spending']/reduction_results['total_spending']*100:.1f}% of total")
                
                with col3:
                    st.metric("Potential Monthly Savings", 
                             f"${reduction_results['potential_savings']:.2f}",
                             f"{reduction_target*100:.0f}% of discretionary")
                
                # Charts
                if 'spending_distribution_chart' in reduction_results:
                    st.pyplot(reduction_results['spending_distribution_chart'])
                
                # Savings recommendations
                st.subheader("Savings Recommendations by Category")
                if reduction_results['category_savings']:
                    savings_df = pd.DataFrame(reduction_results['category_savings'])
                    
                    # Create a cleaner display version
                    display_df = savings_df.copy()
                    display_df.columns = ['Category', 'Current Amount', 'Target Amount', 'Savings']
                    display_df['Current Amount'] = display_df['Current Amount'].map('${:.2f}'.format)
                    display_df['Target Amount'] = display_df['Target Amount'].map('${:.2f}'.format)
                    display_df['Savings'] = display_df['Savings'].map('${:.2f}'.format)
                    
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.info("No category savings recommendations available. Make sure you have category data in your CSV.")
                
                # High spending days
                if reduction_results['high_spending_days']:
                    st.subheader("High-Spending Days to Be Cautious About")
                    days_df = pd.DataFrame(reduction_results['high_spending_days'])
                    
                    # Create a cleaner display version
                    display_days = days_df.copy()
                    display_days.columns = ['Day', 'Amount', 'Above Average']
                    display_days['Amount'] = display_days['Amount'].map('${:.2f}'.format)
                    display_days['Above Average'] = display_days['Above Average'].map('${:.2f}'.format)
                    
                    st.dataframe(display_days, use_container_width=True)
                
                # Additional strategies
                st.subheader("Additional Strategies")
                
                strategies = []
                
                if reduction_results['high_spending_days']:
                    highest_day = max(reduction_results['high_spending_days'], key=lambda x: x['amount'])
                    strategies.append(f"1. Consider implementing a budget cap for {highest_day['day']}s")
                
                if len(reduction_results['category_savings']) > 0:
                    top_saving_cat = max(reduction_results['category_savings'], key=lambda x: x['savings'])
                    strategies.append(f"2. Focus on reducing {top_saving_cat['category']} expenses for biggest impact")
                
                for strategy in strategies:
                    st.markdown(f"- {strategy}")
                
                if not strategies:
                    st.info("No additional strategies available based on your data.")
        
        # --- Weekly Analysis Tab ---
        with tabs[2]:
            st.header("Weekly Spending Analysis")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Configure Analysis")
                
                # Get actual categories from data
                available_categories = list(analyzer.data['category'].unique()) if 'category' in analyzer.data.columns else []
                
                # Default discretionary and essential categories
                default_discretionary = ['Dining', 'Entertainment', 'Shopping']
                default_essential = ['Groceries', 'Utilities', 'Rent', 'Bills']
                
                # Filter defaults to only include categories that exist in the data
                available_disc = [cat for cat in default_discretionary if cat in available_categories]
                available_ess = [cat for cat in default_essential if cat in available_categories]
                
                # If none of the defaults exist, use empty lists as defaults
                disc_default = available_disc if available_disc else []
                ess_default = available_ess if available_ess else []
                
                disc_cats_weekly = st.multiselect(
                    "Discretionary Categories",
                    options=available_categories,
                    default=disc_default,
                    key="disc_cats_weekly"
                )
                
                weekly_savings = st.slider("Weekly Savings Target (%)", 5, 30, 10, key="weekly_savings") / 100
                daily_reduction = st.slider("Daily Reduction Target (%)", 5, 50, 20, key="daily_reduction") / 100
                category_reduction = st.slider("Category Reduction Target (%)", 5, 50, 25, key="category_reduction") / 100
                
                run_weekly = st.button("Run Weekly Analysis")
            
            with col2:
                st.subheader("About Weekly Analysis")
                st.write("""
                This analysis examines your spending patterns by week and day of week to identify:
                
                - **Weekly Patterns**: How your spending varies across weeks of the month
                - **Daily Patterns**: Which days of the week have the highest spending
                - **Category Trends**: How category spending changes throughout the month
                
                The analysis will provide targeted strategies to reduce spending based on these patterns.
                """)
            
            if run_weekly or 'weekly_results' in st.session_state:
                # If first time or button pressed, run the analysis
                if run_weekly or 'weekly_results' not in st.session_state:
                    weekly_results = analyzer.analyze_weekly_spending(
                        discretionary_categories=disc_cats_weekly,
                        weekly_savings_target=weekly_savings,
                        daily_reduction_target=daily_reduction,
                        category_reduction_target=category_reduction
                    )
                    st.session_state.weekly_results = weekly_results
                else:
                    # Use cached results
                    weekly_results = st.session_state.weekly_results
                
                st.divider()
                
                # Display visualizations
                if 'weekly_analysis_chart' in weekly_results:
                    st.pyplot(weekly_results['weekly_analysis_chart'])
                
                # Weekly insights section
                st.subheader("Weekly Spending Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Average Weekly Spending", f"${weekly_results['weekly_average']:.2f}")
                    
                    if weekly_results['high_spending_weeks']:
                        st.subheader("High-Spending Weeks")
                        for week_info in weekly_results['high_spending_weeks']:
                            with st.expander(f"Week {week_info['week']} (${week_info['amount']:.2f})"):
                                st.write(f"Exceeds weekly average by ${week_info['above_average']:.2f}")
                                if week_info['top_categories']:
                                    st.write("Top spending categories this week:")
                                    for cat, cat_amount in week_info['top_categories'].items():
                                        st.write(f"- {cat}: ${cat_amount:.2f}")
                
                with col2:
                    st.metric("Average Daily Spending", f"${weekly_results['daily_average']:.2f}")
                    
                    if weekly_results['high_spending_days']:
                        st.subheader("High-Spending Days")
                        for day_info in weekly_results['high_spending_days']:
                            with st.expander(f"{day_info['day']} (${day_info['amount']:.2f})"):
                                st.write(f"Exceeds daily average by ${day_info['above_average']:.2f}")
                                if day_info['top_categories']:
                                    st.write(f"Top {day_info['day']} categories:")
                                    for cat, cat_amount in day_info['top_categories'].items():
                                        st.write(f"- {cat}: ${cat_amount:.2f}")
                
                # Weekly reduction plan
                st.subheader("Weekly Spending Reduction Plan")
                st.metric("Weekly Savings Target", f"${weekly_results['weekly_savings_target']:.2f}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if "day_reduction_strategy" in weekly_results:
                        day_strategy = weekly_results["day_reduction_strategy"]
                        st.write(f"### Strategy 1: Reduce {day_strategy['day']} Spending")
                        st.write(f"Current: ${day_strategy['current_amount']:.2f}")
                        st.write(f"Target: ${day_strategy['target_amount']:.2f}")
                        st.metric("Weekly Savings", f"${day_strategy['savings']:.2f}")
                
                with col2:
                    if "category_reduction_strategy" in weekly_results:
                        cat_strategy = weekly_results["category_reduction_strategy"]
                        st.write(f"### Strategy 2: Reduce {cat_strategy['category']} Spending")
                        st.write(f"Current weekly: ${cat_strategy['weekly_amount']:.2f}")
                        st.write(f"Target weekly: ${cat_strategy['target_weekly']:.2f}")
                        st.metric("Weekly Savings", f"${cat_strategy['weekly_savings']:.2f}")
                
                # Download weekly analysis report
                weekly_report = f"""
                # Weekly Spending Analysis Report
                
                ## Summary Statistics
                - Total Spending: ${weekly_results['total_spending']:.2f}
                - Average Weekly Spending: ${weekly_results['weekly_average']:.2f}
                - Average Daily Spending: ${weekly_results['daily_average']:.2f}
                - Weekly Savings Target: ${weekly_results['weekly_savings_target']:.2f}
                
                ## Reduction Strategies
                """
                
                if "day_reduction_strategy" in weekly_results:
                    day_strategy = weekly_results["day_reduction_strategy"]
                    weekly_report += f"""
                    ### Strategy 1: Reduce {day_strategy['day']} Spending
                    - Current: ${day_strategy['current_amount']:.2f}
                    - Target: ${day_strategy['target_amount']:.2f}
                    - Weekly Savings: ${day_strategy['savings']:.2f}
                    """
                
                if "category_reduction_strategy" in weekly_results:
                    cat_strategy = weekly_results["category_reduction_strategy"]
                    weekly_report += f"""
                    ### Strategy 2: Reduce {cat_strategy['category']} Spending
                    - Current weekly: ${cat_strategy['weekly_amount']:.2f}
                    - Target weekly: ${cat_strategy['target_weekly']:.2f}
                    - Weekly Savings: ${cat_strategy['weekly_savings']:.2f}
                    """
                
                # Download report button placeholder
                st.download_button(
                    "Download Weekly Analysis Report",
                    weekly_report,
                    file_name="weekly_spending_analysis.txt",
                    mime="text/plain"
                )
        
        # --- Data Utilities Tab ---
        with tabs[3]:
            st.header("Data Utilities")
            
            # Subtabs for different utilities
            subtabs = st.tabs(["Add Day/Week Columns", "Extract Recurring Expenses"])
            
            # Add Day/Week Columns
            with subtabs[0]:
                st.subheader("Add Day and Week Columns")
                st.write("""
                This utility adds two helpful columns to your financial data:
                - **day**: Day of the week with a prefix for sorting (e.g., '1Monday', '2Tuesday')
                - **week**: Week of the month (1-5)
                """)
                
                date_format = st.text_input("Date Format", value="%m/%d/%y", 
                                          help="Format of your date column, e.g., %m/%d/%y for MM/DD/YY")
                
                if st.button("Add Day/Week Columns"):
                    enhanced_df = analyzer.add_day_week_columns(date_format=date_format)
                    
                    # Display preview with improved row count visibility
                    st.subheader("Preview of Enhanced Data")
                    st.write(f"Total rows in dataset: {len(enhanced_df)}")
                    
                    # Show all rows directly without slider
                    st.dataframe(enhanced_df)
                    
                    # Add download button with row count
                    st.markdown(
                        analyzer.get_download_link(enhanced_df, 
                                                 filename=f"{os.path.splitext(analyzer.file_name)[0]}_enhanced.csv",
                                                 text=f"📥 Download Enhanced CSV ({len(enhanced_df)} rows)"),
                        unsafe_allow_html=True
                    )
            
            # Extract Recurring Expenses
            with subtabs[1]:
                st.subheader("Extract Recurring Expenses")
                st.write("""
                This utility extracts all expenses that are not marked as 'onetime' in the 'kind' column.
                It helps identify recurring payments and subscriptions.
                
                Note: Your data must have a 'kind' column with values like 'recurring', 'subscription', etc.
                """)
                
                if st.button("Extract Recurring Expenses"):
                    if 'kind' in analyzer.data.columns:
                        recurring_df = analyzer.print_non_onetime_rows()
                        
                        if not recurring_df.empty:
                            # Calculate total
                            total_amount = recurring_df['Debit'].sum() if 'Debit' in recurring_df.columns else 0
                            
                            # Display stats
                            st.metric("Recurring Expenses Count", len(recurring_df))
                            st.metric("Total Recurring Amount", f"${total_amount:.2f}")
                            
                            # Display data with improved row count visibility
                            st.write(f"Total recurring expenses: {len(recurring_df)} rows")
                            
                            # Show all rows directly without slider
                            st.dataframe(recurring_df)
                            
                            # Add download button with row count
                            st.markdown(
                                analyzer.get_download_link(recurring_df, 
                                                         filename="recurring_expenses.csv",
                                                         text=f"📥 Download Recurring Expenses ({len(recurring_df)} rows)"),
                                unsafe_allow_html=True
                            )
                        else:
                            st.warning("No recurring expenses found in your data.")
                    else:
                        st.error("Your data does not have a 'kind' column which is required for this analysis.")
                        st.info("The 'kind' column should categorize expenses as 'recurring', 'subscription', 'onetime', etc.")

if __name__ == "__main__":
    main()