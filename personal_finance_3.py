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
        # Check if data exists
        if self.data is None:
            st.error("No data loaded. Please upload a CSV file first.")
            return {}
            
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
        # Check if data exists
        if self.data is None:
            st.error("No data loaded. Please upload a CSV file first.")
            return {}
            
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
        # Check if data exists
        if self.data is None:
            st.error("No data loaded. Please upload a CSV file first.")
            return {}
            
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
        # Check if data exists
        if self.data is None:
            st.error("No data loaded. Please upload a CSV file first.")
            return pd.DataFrame()
            
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
        # Check if data exists
        if self.data is None:
            st.error("No data loaded. Please upload a CSV file first.")
            return pd.DataFrame()
            
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
    st.title("💰 Personal Finance Analyzer")
    
    # Initialize session state for tracking if file is uploaded
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    analyzer = FinancialAnalyzer()
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload Financial Data")
        
        uploaded_file = st.file_uploader(
            "Upload a CSV file with your financial data",
            type=["csv"],
            help="Your file should have at least Date and Debit columns"
        )
        
        # Process the uploaded file
        if uploaded_file is not None:
            # Load the data
            data = analyzer.load_data(uploaded_file)
            
            # Set session state to indicate data is loaded
            if 'Date' in data.columns and 'Debit' in data.columns:
                st.session_state.data_loaded = True
            
            st.success(f"Loaded {len(data)} rows from {uploaded_file.name}")
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            # Data validation checks
            if 'Date' not in data.columns:
                st.error("Required 'Date' column not found in data!")
                st.session_state.data_loaded = False
            
            if 'Debit' not in data.columns:
                st.error("Required 'Debit' column not found in data!")
                st.session_state.data_loaded = False
        else:
            # If the file is removed, reset the analyzer's data
            if hasattr(analyzer, 'data') and analyzer.data is not None:
                analyzer.data = None
                analyzer.file_name = None
                st.session_state.data_loaded = False
        
        # Reset button
        if st.button("Reset App"):
            st.session_state.data_loaded = False
            # Clear the analyzer data
            if hasattr(analyzer, 'data'):
                analyzer.data = None
                analyzer.file_name = None
            st.session_state.clear()
            st.experimental_rerun()
    
    # Main content area - show welcome screen or analysis tabs based on data_loaded state
    if not st.session_state.data_loaded:
        # Show welcome screen with HTML instead of Markdown
        welcome_html = """
        <div style="text-align: left; max-width: 800px; margin: 0 auto;">
            <h1>Personal Finance Analyzer</h1>
            <p style="font-size: 1.1rem;">A Streamlit application for analyzing personal financial data, spotting trends, and finding opportunities to reduce expenses.</p>
            
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
        
        # Display the HTML welcome screen
        st.html(welcome_html, height=800)
    else:
        # Check if analyzer still has data loaded
        if analyzer.data is None and uploaded_file is not None:
            data = analyzer.load_data(uploaded_file)</li>
            </ul>
            
            <h2>License</h2>
            <p>This project is licensed under the MIT License - see the LICENSE file for details.</p>
            
            <hr style="margin: 30px 0;">
            <p style="font-style: italic; text-align: center;">Please upload your financial data CSV file using the sidebar to begin analysis.</p>
        </div>
        """

if __name__ == "__main__":
    main()
