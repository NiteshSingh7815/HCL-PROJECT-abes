"""
=============================================================================
HOTEL BOOKING & OCCUPANCY ANALYSIS
Author: Python Data Analyst
Course Project: Hospitality & Tourism Analytics
Dataset: 13_hotel_bookings.csv
Libraries: Pandas, NumPy, Matplotlib, Seaborn
=============================================================================
"""

# =============================================================================
# 0. IMPORT LIBRARIES AND SETTINGS
# =============================================================================
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.titleweight"] = "bold"

print("=" * 80)
print("             HOTEL BOOKING & OCCUPANCY ANALYTICS PIPELINE")
print("=" * 80)

# =============================================================================
# 1. DATA LOADING
# =============================================================================
# Update file path if the CSV is in a specific folder, e.g., '13_hotel_bookings.csv'
file_path = "13_hotel_bookings.csv"

# Fallback check for user's Downloads or working directory
if not os.path.exists(file_path):
    user_download_path = os.path.expanduser(
        r"~\Downloads\13_hotel_bookings.csv"
    )
    if os.path.exists(user_download_path):
        file_path = user_download_path

print(f"\n[Step 1] Loading dataset from: '{file_path}'...")
df = pd.read_csv(file_path)
print(">> Dataset successfully loaded!")

# =============================================================================
# 2. DATASET OVERVIEW & EXPLORATION
# =============================================================================
print("\n" + "=" * 80)
print("[Step 2] DATASET OVERVIEW")
print("=" * 80)
print(f"Shape of Dataset (Rows, Columns): {df.shape}")
print("\n--- Column Data Types and Non-Null Counts ---")
print(df.info())

print("\n--- First 5 Records ---")
print(df.head())

print("\n--- Summary Statistics (Raw Numerical Columns) ---")
print(df.describe().T)

# =============================================================================
# 3. IDENTIFY MISSING VALUES & DUPLICATES
# =============================================================================
print("\n" + "=" * 80)
print("[Step 3] MISSING VALUES & DUPLICATE IDENTIFICATION")
print("=" * 80)

missing_counts = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame(
    {"Missing_Count": missing_counts, "Percentage (%)": missing_percent}
)
print("--- Missing Values Summary ---")
print(missing_df[missing_df["Missing_Count"] > 0])

duplicate_id_count = df["Booking_ID"].duplicated().sum()
print(f"\nNumber of duplicate 'Booking_ID' records found: {duplicate_id_count}")

# =============================================================================
# 4. DATA CLEANING & IMPUTATION
# =============================================================================
print("\n" + "=" * 80)
print("[Step 4] DATA CLEANING & IMPUTATION")
print("=" * 80)

# 4.1 Remove duplicate Booking_IDs (keeping the first occurrence)
if duplicate_id_count > 0:
    df.drop_duplicates(subset=["Booking_ID"], keep="first", inplace=True)
    print(
        f">> Removed {duplicate_id_count} duplicate booking IDs. New shape: {df.shape}"
    )

# 4.2 Handle missing Lead_Time_Days using Median (robust against right-skew)
median_lead_time = df["Lead_Time_Days"].median()
df["Lead_Time_Days"] = df["Lead_Time_Days"].fillna(median_lead_time)
print(
    f">> Imputed missing Lead_Time_Days with median: {median_lead_time:.1f} days"
)

# 4.3 Handle missing Room_Rate using Room_Type group median
# Different room categories (Standard vs Deluxe vs Suite) have vastly different price tiers.
df["Room_Rate"] = df.groupby("Room_Type")["Room_Rate"].transform(
    lambda group: group.fillna(group.median())
)

# Fallback imputation if any overall NaN remains
if df["Room_Rate"].isnull().sum() > 0:
    df["Room_Rate"].fillna(df["Room_Rate"].median(), inplace=True)

print(
    ">> Imputed missing Room_Rate using Room_Type group medians (Standard, Deluxe, Suite)."
)
print(
    f">> Remaining missing values across dataset: {df.isnull().sum().sum()} (Cleaned!)"
)

# =============================================================================
# 5. DATETIME CONVERSION & FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 80)
print("[Step 5] DATE CONVERSION & FEATURE EXTRACTION")
print("=" * 80)

# Convert Checkin_Date to datetime
df["Checkin_Date"] = pd.to_datetime(df["Checkin_Date"])

# Extract Arrival Month and Weekday
df["Arrival_Month"] = df["Checkin_Date"].dt.month_name()
df["Arrival_Month_Num"] = df["Checkin_Date"].dt.month
df["Arrival_Weekday"] = df["Checkin_Date"].dt.day_name()

# Calculate Total Booking Revenue: Nights * Room_Rate
df["Total_Revenue"] = df["Nights"] * df["Room_Rate"]

# Create an ordered categorical month for natural chronological plotting
month_order = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
df["Arrival_Month"] = pd.Categorical(
    df["Arrival_Month"], categories=month_order, ordered=True
)

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
df["Arrival_Weekday"] = pd.Categorical(
    df["Arrival_Weekday"], categories=weekday_order, ordered=True
)

print(
    ">> Successfully extracted 'Arrival_Month', 'Arrival_Weekday', and 'Total_Revenue'."
)
print(
    df[
        [
            "Booking_ID",
            "Checkin_Date",
            "Arrival_Month",
            "Arrival_Weekday",
            "Nights",
            "Room_Rate",
            "Total_Revenue",
        ]
    ].head()
)

# =============================================================================
# 6. CORE HOSPITALITY METRICS
# =============================================================================
print("\n" + "=" * 80)
print("[Step 6] KEY PERFORMANCE INDICATORS (KPIs)")
print("=" * 80)

total_bookings = len(df)
total_room_nights = df["Nights"].sum()

canceled_bookings = (df["Booking_Status"] == "Canceled").sum()
confirmed_bookings = (df["Booking_Status"] == "Confirmed").sum()
overall_cancellation_pct = (canceled_bookings / total_bookings) * 100

# Non-canceled (Confirmed) bookings dataset
df_confirmed = df[df["Booking_Status"] == "Confirmed"].copy()
total_net_revenue = df_confirmed["Total_Revenue"].sum()
gross_potential_revenue = df["Total_Revenue"].sum()
lost_revenue = gross_potential_revenue - total_net_revenue

print(f"1. Total Bookings Registered:           {total_bookings:,}")
print(f"2. Total Room Nights Demanded:          {total_room_nights:,} nights")
print(
    f"3. Confirmed Bookings:                  {confirmed_bookings:,} ({100 - overall_cancellation_pct:.2f}%)"
)
print(
    f"4. Canceled Bookings:                   {canceled_bookings:,} ({overall_cancellation_pct:.2f}%)"
)
print(f"5. Total Net Realized Revenue:          ${total_net_revenue:,.2f}")
print(f"6. Potential Revenue Lost to Cancel:    ${lost_revenue:,.2f}")

# =============================================================================
# 7. ROOM TYPE & CHANNEL ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("[Step 7] ROOM TYPE & BOOKING CHANNEL ANALYSIS")
print("=" * 80)

# 7.1 Room Type Comparison (Avg Rate, Total Nights, Bookings, Revenue)
room_type_summary = (
    df.groupby("Room_Type")
    .agg(
        Total_Bookings=("Booking_ID", "count"),
        Avg_Room_Rate=("Room_Rate", "mean"),
        Median_Room_Rate=("Room_Rate", "median"),
        Total_Room_Nights=("Nights", "sum"),
    )
    .reset_index()
)

# Realized revenue by room type (from confirmed bookings)
room_rev = (
    df_confirmed.groupby("Room_Type")["Total_Revenue"]
    .sum()
    .rename("Realized_Revenue")
)
room_type_summary = room_type_summary.merge(room_rev, on="Room_Type")
room_type_summary["Occupancy_Share_%"] = (
    room_type_summary["Total_Room_Nights"]
    / room_type_summary["Total_Room_Nights"].sum()
) * 100

print("--- Room Type Performance Table ---")
print(room_type_summary.to_string(index=False))

# 7.2 Booking Channel Ranking by Volume & Realized Revenue
channel_volume = df.groupby("Booking_Channel")["Booking_ID"].count()
channel_rev = df_confirmed.groupby("Booking_Channel")["Total_Revenue"].sum()
channel_cancel_rate = (
    df.groupby("Booking_Channel")["Booking_Status"].apply(
        lambda s: (s == "Canceled").mean() * 100
    )
)

channel_summary = pd.DataFrame(
    {
        "Total_Bookings": channel_volume,
        "Confirmed_Revenue": channel_rev,
        "Cancellation_Rate_%": channel_cancel_rate,
    }
).sort_values(by="Total_Bookings", ascending=False)

print("\n--- Booking Channel Ranking Table ---")
print(channel_summary.to_string())

# =============================================================================
# 8. LEAD-TIME CATEGORIZATION (pd.cut) & CANCELLATION BEHAVIOR
# =============================================================================
print("\n" + "=" * 80)
print("[Step 8] LEAD-TIME SEGMENTATION & CANCELLATION ANALYSIS")
print("=" * 80)

lead_bins = [-1, 7, 30, 60, 90, 1000]
lead_labels = [
    "Last Minute (0-7 d)",
    "Short Lead (8-30 d)",
    "Medium Lead (31-60 d)",
    "Long Lead (61-90 d)",
    "Very Long Lead (90+ d)",
]

df["Lead_Time_Category"] = pd.cut(
    df["Lead_Time_Days"], bins=lead_bins, labels=lead_labels
)

lead_summary = (
    df.groupby("Lead_Time_Category", observed=False)
    .agg(
        Total_Bookings=("Booking_ID", "count"),
        Canceled_Bookings=(
            "Booking_Status",
            lambda x: (x == "Canceled").sum(),
        ),
        Cancellation_Rate=(
            "Booking_Status",
            lambda x: (x == "Canceled").mean() * 100,
        ),
    )
    .reset_index()
)

print("--- Lead-Time Category Cancellation Analysis ---")
print(lead_summary.to_string(index=False))

# =============================================================================
# 9. SEASONALITY & TRAVELER SEGMENTATION
# =============================================================================
print("\n" + "=" * 80)
print("[Step 9] SEASONAL PEAKS & TRAVELER SEGMENTATION")
print("=" * 80)

# Top 3 peak revenue months (confirmed bookings)
monthly_revenue = (
    df_confirmed.groupby("Arrival_Month", observed=False)["Total_Revenue"]
    .sum()
    .reset_index()
    .sort_values(by="Total_Revenue", ascending=False)
)

print("--- Top 3 Peak Revenue Months ---")
top_3_months = monthly_revenue.head(3)
for idx, row in top_3_months.iterrows():
    print(f"Rank {row['Arrival_Month']}: ${row['Total_Revenue']:,.2f}")

# Compare stay duration: Leisure vs Business travelers
guest_stay = (
    df.groupby("Guest_Type")
    .agg(
        Avg_Stay_Nights=("Nights", "mean"),
        Median_Stay_Nights=("Nights", "median"),
        Std_Stay_Nights=("Nights", "std"),
        Total_Guests=("Booking_ID", "count"),
    )
    .reset_index()
)

print("\n--- Average Stay Duration by Traveler Segment ---")
print(guest_stay.to_string(index=False))

# =============================================================================
# 10. STATISTICAL ANALYSIS (NUMPY & IQR METHOD)
# =============================================================================
print("\n" + "=" * 80)
print("[Step 10] STATISTICAL METRICS & IQR OUTLIER DETECTION")
print("=" * 80)

room_rates = df["Room_Rate"].to_numpy()

# NumPy calculations
mean_rate = np.mean(room_rates)
median_rate = np.median(room_rates)
std_rate = np.std(room_rates)
min_rate = np.min(room_rates)
max_rate = np.max(room_rates)

print("--- NumPy Descriptive Statistics for Room Rates ---")
print(f"Mean Room Rate:               ${mean_rate:.2f}")
print(f"Median Room Rate:             ${median_rate:.2f}")
print(f"Standard Deviation:           ${std_rate:.2f}")
print(f"Minimum Room Rate:            ${min_rate:.2f}")
print(f"Maximum Room Rate:            ${max_rate:.2f}")

# IQR Method for Outlier Detection in Room Rates
Q1 = np.percentile(room_rates, 25)
Q3 = np.percentile(room_rates, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df["Room_Rate"] < lower_bound) | (df["Room_Rate"] > upper_bound)]

print("\n--- IQR Outlier Detection (Room_Rate) ---")
print(f"25th Percentile (Q1):         ${Q1:.2f}")
print(f"75th Percentile (Q3):         ${Q3:.2f}")
print(f"Interquartile Range (IQR):    ${IQR:.2f}")
print(f"Lower Bound (Q1 - 1.5*IQR):   ${lower_bound:.2f}")
print(f"Upper Bound (Q3 + 1.5*IQR):   ${upper_bound:.2f}")
print(f"Total Unusual / Outlier Rows: {len(outliers)}")

if len(outliers) > 0:
    print(
        "\nSample of Outlier Records (Extremely High / Luxury Suite Rates):"
    )
    print(
        outliers[
            [
                "Booking_ID",
                "Room_Type",
                "Room_Rate",
                "Guest_Type",
                "Booking_Status",
            ]
        ].head()
    )

# =============================================================================
# 11. DATA VISUALIZATIONS (7 EXPECTED CHARTS)
# =============================================================================
print("\n" + "=" * 80)
print("[Step 11] GENERATING VISUALIZATIONS...")
print("=" * 80)

# Output directory for saving plots
os.makedirs("visualizations", exist_ok=True)

# -----------------------------------------------------------------------------
# Chart 1: Monthly Revenue Trend
# -----------------------------------------------------------------------------
monthly_rev_all = (
    df_confirmed.groupby("Arrival_Month", observed=False)["Total_Revenue"]
    .sum()
    .reset_index()
)

plt.figure(figsize=(12, 6))
sns.lineplot(
    data=monthly_rev_all,
    x="Arrival_Month",
    y="Total_Revenue",
    marker="o",
    linewidth=3,
    color="#1f77b4",
    markersize=8,
)
plt.title(
    "1. Monthly Revenue Trend (Confirmed Bookings)", fontsize=14, pad=15
)
plt.xlabel("Arrival Month", fontweight="bold")
plt.ylabel("Realized Revenue ($)", fontweight="bold")
plt.xticks(rotation=45)
plt.gca().yaxis.set_major_formatter("${x:,.0f}")
plt.grid(True, linestyle="--", alpha=0.6)
for i, row in monthly_rev_all.iterrows():
    plt.annotate(
        f"${row['Total_Revenue']/1000:,.1f}k",
        (row["Arrival_Month"], row["Total_Revenue"]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=9,
        fontweight="bold",
    )
plt.tight_layout()
plt.savefig("visualizations/1_monthly_revenue_trend.png", dpi=300)
plt.show()

# -----------------------------------------------------------------------------
# Chart 2: Room Type Revenue Comparison
# -----------------------------------------------------------------------------
plt.figure(figsize=(9, 6))
bar_colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]
ax = sns.barplot(
    data=room_type_summary,
    x="Room_Type",
    y="Realized_Revenue",
    palette=bar_colors,
    edgecolor="black",
)
plt.title("2. Realized Revenue Comparison by Room Type", fontsize=14, pad=15)
plt.xlabel("Room Category", fontweight="bold")
plt.ylabel("Total Realized Revenue ($)", fontweight="bold")
plt.gca().yaxis.set_major_formatter("${x:,.0f}")
for p in ax.patches:
    height = p.get_height()
    ax.annotate(
        f"${height:,.0f}",
        (p.get_x() + p.get_width() / 2.0, height),
        ha="center",
        va="bottom",
        xytext=(0, 5),
        textcoords="offset points",
        fontweight="bold",
    )
plt.tight_layout()
plt.savefig("visualizations/2_room_type_revenue.png", dpi=300)
plt.show()

# -----------------------------------------------------------------------------
# Chart 3: Booking Channel Analysis (Volume & Revenue Dual Subplot)
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Channel Volume
sns.barplot(
    x=channel_summary.index,
    y=channel_summary["Total_Bookings"],
    ax=ax1,
    palette="Blues_r",
    edgecolor="black",
)
ax1.set_title("Booking Volume by Channel", fontweight="bold")
ax1.set_xlabel("Booking Channel", fontweight="bold")
ax1.set_ylabel("Total Bookings Count", fontweight="bold")
ax1.tick_params(axis="x", rotation=30)
for p in ax1.patches:
    ax1.annotate(
        f"{int(p.get_height()):,}",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="bottom",
        xytext=(0, 4),
        textcoords="offset points",
        fontweight="bold",
    )

# Channel Revenue
sns.barplot(
    x=channel_summary.index,
    y=channel_summary["Confirmed_Revenue"],
    ax=ax2,
    palette="Greens_r",
    edgecolor="black",
)
ax2.set_title("Confirmed Revenue by Channel", fontweight="bold")
ax2.set_xlabel("Booking Channel", fontweight="bold")
ax2.set_ylabel("Total Net Revenue ($)", fontweight="bold")
ax2.yaxis.set_major_formatter("${x:,.0f}")
ax2.tick_params(axis="x", rotation=30)
for p in ax2.patches:
    ax2.annotate(
        f"${p.get_height():,.0f}",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="bottom",
        xytext=(0, 4),
        textcoords="offset points",
        fontweight="bold",
    )

fig.suptitle(
    "3. Booking Channel Analysis: Volume vs Revenue",
    fontsize=16,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("visualizations/3_booking_channel_analysis.png", dpi=300)
plt.show()

# -----------------------------------------------------------------------------
# Chart 4: Cancellation Rate by Booking Channel
# -----------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
channel_summary_sorted = channel_summary.sort_values(
    by="Cancellation_Rate_%", ascending=True
)
bars = plt.barh(
    channel_summary_sorted.index,
    channel_summary_sorted["Cancellation_Rate_%"],
    color="#d62728",
    edgecolor="black",
    height=0.6,
)
plt.title("4. Cancellation Rate by Booking Channel (%)", fontsize=14, pad=15)
plt.xlabel("Cancellation Rate (%)", fontweight="bold")
plt.ylabel("Booking Channel", fontweight="bold")
plt.xlim(0, max(channel_summary["Cancellation_Rate_%"]) + 10)
for bar in bars:
    w = bar.get_width()
    plt.text(
        w + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f"{w:.2f}%",
        va="center",
        fontweight="bold",
    )
plt.tight_layout()
plt.savefig("visualizations/4_cancellation_rate_by_channel.png", dpi=300)
plt.show()

# -----------------------------------------------------------------------------
# Chart 5: Monthly Occupancy Trend (Room Nights)
# -----------------------------------------------------------------------------
monthly_nights = (
    df_confirmed.groupby("Arrival_Month", observed=False)["Nights"]
    .sum()
    .reset_index()
)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=monthly_nights,
    x="Arrival_Month",
    y="Nights",
    palette="Spectral",
    edgecolor="black",
)
plt.title(
    "5. Monthly Occupancy Trend (Confirmed Room Nights)", fontsize=14, pad=15
)
plt.xlabel("Arrival Month", fontweight="bold")
plt.ylabel("Total Room Nights Occupied", fontweight="bold")
plt.xticks(rotation=45)
for p in plt.gca().patches:
    plt.gca().annotate(
        f"{int(p.get_height()):,}",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="bottom",
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
    )
plt.tight_layout()
plt.savefig("visualizations/5_monthly_occupancy_trend.png", dpi=300)
plt.show()

# -----------------------------------------------------------------------------
# Chart 6: Average Room Rate Comparison (Boxplot & Mean Bar)
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df,
    x="Room_Type",
    y="Room_Rate",
    palette="Pastel1",
    showmeans=True,
    meanprops={
        "marker": "o",
        "markerfacecolor": "red",
        "markeredgecolor": "black",
        "markersize": "9",
    },
)
plt.title(
    "6. Room Rate Distribution and Mean by Room Type\n(Red Dot = Mean Rate)",
    fontsize=14,
    pad=15,
)
plt.xlabel("Room Category", fontweight="bold")
plt.ylabel("Nightly Room Rate ($)", fontweight="bold")
plt.gca().yaxis.set_major_formatter("${x:,.0f}")
plt.tight_layout()
plt.savefig("visualizations/6_avg_room_rate_comparison.png", dpi=300)
plt.show()

# -----------------------------------------------------------------------------
# Chart 7: Lead-Time Category Analysis (Cancellation Rate & Volume)
# -----------------------------------------------------------------------------
plt.figure(figsize=(11, 6))
ax1 = sns.barplot(
    data=lead_summary,
    x="Lead_Time_Category",
    y="Cancellation_Rate",
    palette="Reds_r",
    edgecolor="black",
)
plt.title(
    "7. Cancellation Rate by Lead-Time Category", fontsize=14, pad=15
)
plt.xlabel("Lead-Time Window", fontweight="bold")
plt.ylabel("Cancellation Rate (%)", fontweight="bold")
plt.xticks(rotation=20)
for p in ax1.patches:
    ax1.annotate(
        f"{p.get_height():.1f}%",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="bottom",
        xytext=(0, 4),
        textcoords="offset points",
        fontweight="bold",
    )
plt.tight_layout()
plt.savefig("visualizations/7_lead_time_cancellation_analysis.png", dpi=300)
plt.show()

print("\n>> All 7 visualizations successfully generated and saved to disk!")
print("=" * 80)
print("             ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 80)