
# ==========================================
# OIBSIP TASK 1: EDA ON RETAIL SALES DATA
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("retail_sales_dataset.csv")


print("\n--- Dataset Information ---")
df.info()

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- First 5 Rows ---")
print(df.head())


numeric_data = df.select_dtypes(include='number')

print("\n--- Mean ---")
print(numeric_data.mean())

print("\n--- Median ---")
print(numeric_data.median())

print("\n--- Mode ---")
print(numeric_data.mode())

print("\n--- Standard Deviation ---")
print(numeric_data.std())


df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)


df['Month'] = df['Date'].dt.to_period('M')
monthly_sales = df.groupby('Month')['Total Amount'].sum()

print("\n--- Monthly Sales ---")
print(monthly_sales)


plt.figure(figsize=(12, 5))
monthly_sales.plot(kind='line', marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


df['Quarter'] = df['Date'].dt.to_period('Q')
quarterly_sales = df.groupby('Quarter')['Total Amount'].sum()

print("\n--- Quarterly Sales ---")
print(quarterly_sales)


plt.figure(figsize=(10, 5))
quarterly_sales.plot(kind='bar')
plt.title("Quarterly Sales Analysis")
plt.xlabel("Quarter")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


df['Age Group'] = pd.cut(
    df['Age'],
    bins=[0, 18, 25, 35, 45, 55, 100],
    labels=['Below 18', '18-25', '26-35',
            '36-45', '46-55', 'Above 55']
)

age_group = df['Age Group'].value_counts().sort_index()

print("\n--- Customer Age Group ---")
print(age_group)

plt.figure(figsize=(10, 5))
age_group.plot(kind='bar')
plt.title("Customer Age Group Distribution")
plt.xlabel("Age Group")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


gender_count = df['Gender'].value_counts()

print("\n--- Gender Distribution ---")
print(gender_count)

plt.figure(figsize=(7, 5))
gender_count.plot(kind='bar')
plt.title("Male vs Female Customers")
plt.xlabel("Gender")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


category_sales = df.groupby('Product Category')['Total Amount'].sum()

print("\n--- Revenue by Product Category ---")
print(category_sales)

plt.figure(figsize=(9, 5))
category_sales.plot(kind='bar')
plt.title("Revenue by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


top_categories = category_sales.sort_values(
    ascending=False
).head(10)

print("\n--- Top Categories ---")
print(top_categories)

plt.figure(figsize=(9, 5))
top_categories.plot(kind='bar')
plt.title("Top Product Categories by Revenue")
plt.xlabel("Product Category")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


correlation = numeric_data.corr()

print("\n--- Correlation Matrix ---")
print(correlation)

plt.figure(figsize=(10, 6))
sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

print("\n--- EDA Analysis Completed ---")



gender_sales = df.groupby('Gender')['Total Amount'].mean()

print("\n--- Average Sales by Gender ---")
print(gender_sales)

plt.figure(figsize=(8, 5))
gender_sales.plot(kind='bar')

plt.title("Average Sales by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Sales")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()



print("\n========== BUSINESS RECOMMENDATIONS ==========")

print("1. Sales Performance:")
print("Analyze monthly sales trends and offer discounts during low-sales months.")

print("\n2. Product Category:")
print("Focus on high-revenue product categories and improve their availability.")

print("\n3. Customer Behaviour:")
print("Use customer gender and age-group analysis to create targeted marketing campaigns.")