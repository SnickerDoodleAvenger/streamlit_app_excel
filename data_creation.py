import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set a seed for reproducibility
np.random.seed(42)

# Create date range for the past 3 years
start_date = datetime(2022, 1, 1)
end_date = datetime(2024, 12, 31)
date_range = pd.date_range(start=start_date, end=end_date, freq='MS')  # Monthly start

# Product categories
categories = ['Electronics', 'Clothing', 'Home Goods', 'Sporting Goods', 'Beauty']
regions = ['North', 'South', 'East', 'West', 'Central']
channels = ['Online', 'Retail', 'Wholesale', 'Direct Sales']
product_lines = ['Premium', 'Standard', 'Budget', 'Exclusive']

# Create empty lists to hold data
dates = []
product_cats = []
region_list = []
channel_list = []
product_line_list = []
revenue = []
cogs = []
marketing_expense = []
fulfillment_expense = []
admin_expense = []
customer_counts = []
transaction_counts = []
return_rates = []

# Generate records
for date in date_range:
    for category in categories:
        for region in regions:
            # Base values for this segment
            base_revenue = random.randint(50000, 200000)

            # Seasonal factor (higher in Q4, lower in Q1)
            month = date.month
            if month in [10, 11, 12]:  # Q4
                seasonal_factor = np.random.uniform(1.2, 1.5)
            elif month in [1, 2, 3]:  # Q1
                seasonal_factor = np.random.uniform(0.7, 0.9)
            elif month in [7, 8, 9]:  # Q3
                seasonal_factor = np.random.uniform(0.8, 1.0)
            else:  # Q2
                seasonal_factor = np.random.uniform(0.9, 1.1)

            # Growth over time
            years_passed = (date.year - start_date.year) + (date.month - start_date.month) / 12
            growth_factor = 1 + (0.05 * years_passed)  # 5% yearly growth

            # Regional factor
            if region == 'North':
                regional_factor = 1.1
            elif region == 'South':
                regional_factor = 0.9
            elif region == 'East':
                regional_factor = 1.2
            elif region == 'West':
                regional_factor = 1.3
            else:  # Central
                regional_factor = 0.85

            # Category factor
            if category == 'Electronics':
                category_factor = 1.4
                avg_margin = 0.4  # 40% margin
            elif category == 'Clothing':
                category_factor = 1.2
                avg_margin = 0.6  # 60% margin
            elif category == 'Home Goods':
                category_factor = 0.9
                avg_margin = 0.55  # 55% margin
            elif category == 'Sporting Goods':
                category_factor = 0.8
                avg_margin = 0.5  # 50% margin
            else:  # Beauty
                category_factor = 1.1
                avg_margin = 0.7  # 70% margin

            for channel in channels:
                if channel == 'Online':
                    channel_factor = 1.3
                    expense_ratio = 0.2  # Fulfillment is higher for online
                elif channel == 'Retail':
                    channel_factor = 0.8
                    expense_ratio = 0.15
                elif channel == 'Wholesale':
                    channel_factor = 0.9
                    expense_ratio = 0.1
                else:  # Direct Sales
                    channel_factor = 1.1
                    expense_ratio = 0.12

                for product_line in product_lines:
                    if product_line == 'Premium':
                        line_factor = 1.5
                        margin_adjust = 0.1  # Higher margin
                        vol_factor = 0.7  # Lower volume
                    elif product_line == 'Standard':
                        line_factor = 1.0
                        margin_adjust = 0
                        vol_factor = 1.0
                    elif product_line == 'Budget':
                        line_factor = 0.7
                        margin_adjust = -0.1  # Lower margin
                        vol_factor = 1.3  # Higher volume
                    else:  # Exclusive
                        line_factor = 1.8
                        margin_adjust = 0.15
                        vol_factor = 0.5

                    # Calculate revenue with random noise
                    segment_revenue = base_revenue * seasonal_factor * growth_factor * \
                                      regional_factor * category_factor * channel_factor * \
                                      line_factor * np.random.uniform(0.9, 1.1)

                    # Calculate other metrics
                    segment_cogs = segment_revenue * (1 - (avg_margin + margin_adjust)) * \
                                   np.random.uniform(0.95, 1.05)

                    segment_marketing = segment_revenue * 0.15 * np.random.uniform(0.8, 1.2)
                    segment_fulfillment = segment_revenue * expense_ratio * np.random.uniform(0.9, 1.1)
                    segment_admin = segment_revenue * 0.05 * np.random.uniform(0.95, 1.05)

                    avg_order_value = {
                        'Premium': np.random.uniform(200, 350),
                        'Standard': np.random.uniform(100, 200),
                        'Budget': np.random.uniform(50, 100),
                        'Exclusive': np.random.uniform(300, 500)
                    }[product_line]

                    segment_trans_count = int(segment_revenue / avg_order_value)
                    segment_customer_count = int(
                        segment_trans_count * np.random.uniform(0.7, 0.9))  # Some customers make multiple purchases

                    segment_return_rate = {
                        'Electronics': np.random.uniform(0.05, 0.15),
                        'Clothing': np.random.uniform(0.1, 0.25),
                        'Home Goods': np.random.uniform(0.03, 0.08),
                        'Sporting Goods': np.random.uniform(0.04, 0.1),
                        'Beauty': np.random.uniform(0.02, 0.07)
                    }[category]

                    # Append data
                    dates.append(date)
                    product_cats.append(category)
                    region_list.append(region)
                    channel_list.append(channel)
                    product_line_list.append(product_line)
                    revenue.append(round(segment_revenue, 2))
                    cogs.append(round(segment_cogs, 2))
                    marketing_expense.append(round(segment_marketing, 2))
                    fulfillment_expense.append(round(segment_fulfillment, 2))
                    admin_expense.append(round(segment_admin, 2))
                    customer_counts.append(segment_customer_count)
                    transaction_counts.append(segment_trans_count)
                    return_rates.append(segment_return_rate)

# Create DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Product_Category': product_cats,
    'Region': region_list,
    'Channel': channel_list,
    'Product_Line': product_line_list,
    'Revenue': revenue,
    'COGS': cogs,
    'Marketing_Expense': marketing_expense,
    'Fulfillment_Expense': fulfillment_expense,
    'Administrative_Expense': admin_expense,
    'Customer_Count': customer_counts,
    'Transaction_Count': transaction_counts,
    'Return_Rate': return_rates
})

# Calculate derived metrics
df['Gross_Profit'] = df['Revenue'] - df['COGS']
df['Gross_Margin_Pct'] = df['Gross_Profit'] / df['Revenue']
df['Total_Expenses'] = df['Marketing_Expense'] + df['Fulfillment_Expense'] + df['Administrative_Expense']
df['Net_Profit'] = df['Gross_Profit'] - df['Total_Expenses']
df['Net_Margin_Pct'] = df['Net_Profit'] / df['Revenue']
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df['Average_Order_Value'] = df['Revenue'] / df['Transaction_Count']

# Save to Excel
df.to_excel('sample_financial_data.xlsx', index=False)

print(f"Generated {len(df)} rows of financial data and saved to 'sample_financial_data.xlsx'")