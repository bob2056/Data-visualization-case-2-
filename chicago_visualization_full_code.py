


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.units import inch

# --------- Configuration ---------
CLEANED_CSV = "/mnt/data/Chicago_Crimes_2012_2017_cleaned_sample100k.csv"
OUTPUT_DIR = "/mnt/data/"
PDF_OUTPUT = OUTPUT_DIR + "Chicago_Crimes_Visual_Report_Detailed_from_code.pdf"

# --------- Load data ---------
df = pd.read_csv(CLEANED_CSV, low_memory=False, parse_dates=['IncidentDate'], infer_datetime_format=True)

# Ensure temporal fields
if 'Year' not in df.columns and 'IncidentDate' in df.columns:
    df['Year'] = df['IncidentDate'].dt.year
    df['Month'] = df['IncidentDate'].dt.month
    df['Day'] = df['IncidentDate'].dt.day
    df['Hour'] = df['IncidentDate'].dt.hour
    df['Weekday'] = df['IncidentDate'].dt.day_name()

# Detect lat/lon columns
lat_col = next((c for c in df.columns if c.lower().startswith('lat')), None)
lon_col = next((c for c in df.columns if c.lower().startswith('lon')), None)

# Helper to save figures
def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

# --------- Visualization 1: Top 10 Primary Types ---------
top_primary = df['Primary Type'].value_counts().head(10)
fig1, ax1 = plt.subplots(figsize=(10,6))
ax1.bar(top_primary.index, top_primary.values)
ax1.set_title("Top 10 Primary Crime Types (sample)")
ax1.set_ylabel("Count")
ax1.set_xticklabels(top_primary.index, rotation=45, ha='right')
save_fig(fig1, OUTPUT_DIR + "vis1_top10_primary.png")

# --------- Visualization 2: Arrest Rate by Primary Type ---------
top10 = top_primary.index.tolist()
arrest_rate = df[df['Primary Type'].isin(top10)].groupby('Primary Type')['Arrest'].mean().loc[top10]
fig2, ax2 = plt.subplots(figsize=(10,6))
ax2.bar(arrest_rate.index, arrest_rate.values)
ax2.set_title("Arrest Rate by Primary Type (top 10)")
ax2.set_ylabel("Arrest Rate (proportion)")
ax2.set_xticklabels(arrest_rate.index, rotation=45, ha='right')
save_fig(fig2, OUTPUT_DIR + "vis2_arrest_rate.png")

# --------- Visualization 3: Monthly Crime Counts Time Series ---------
df_ts = df.dropna(subset=['IncidentDate']).copy()
df_ts['YearMonth'] = df_ts['IncidentDate'].dt.to_period('M').dt.to_timestamp()
monthly = df_ts.groupby('YearMonth').size().sort_index()
fig3, ax3 = plt.subplots(figsize=(12,4))
ax3.plot(monthly.index, monthly.values)
ax3.set_title("Monthly Crime Counts (sample)")
ax3.set_ylabel("Count")
fig3.autofmt_xdate()
save_fig(fig3, OUTPUT_DIR + "vis3_monthly_ts.png")

# --------- Visualization 4: Hourly Distribution ---------
fig4, ax4 = plt.subplots(figsize=(10,4))
ax4.hist(df['Hour'].dropna().astype(int), bins=24, range=(0,24))
ax4.set_title("Hourly Distribution of Incidents")
ax4.set_xlabel("Hour of Day")
ax4.set_ylabel("Count")
save_fig(fig4, OUTPUT_DIR + "vis4_hourly_hist.png")

# --------- Visualization 5: Weekday Counts ---------
weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
weekday_counts = df['Weekday'].value_counts().reindex(weekday_order).fillna(0)
fig5, ax5 = plt.subplots(figsize=(10,4))
ax5.bar(weekday_counts.index, weekday_counts.values)
ax5.set_title("Incidents by Weekday")
ax5.set_ylabel("Count")
save_fig(fig5, OUTPUT_DIR + "vis5_weekday.png")

# --------- Visualization 6: Top 10 Location Descriptions ---------
loc_top10 = df['Location Description'].value_counts().head(10)
fig6, ax6 = plt.subplots(figsize=(10,6))
ax6.bar(loc_top10.index, loc_top10.values)
ax6.set_title("Top 10 Location Descriptions (sample)")
ax6.set_ylabel("Count")
ax6.set_xticklabels(loc_top10.index, rotation=45, ha='right')
save_fig(fig6, OUTPUT_DIR + "vis6_top10_locations.png")

# --------- Visualization 7: Spatial Scatter ---------
if lat_col and lon_col:
    sample_scatter = df.dropna(subset=[lat_col, lon_col]).sample(n=min(20000, len(df)), random_state=1)
    fig7, ax7 = plt.subplots(figsize=(8,8))
    ax7.scatter(sample_scatter[lon_col], sample_scatter[lat_col], s=1)
    ax7.set_title("Spatial Scatter of Incidents (sampled)")
    ax7.set_xlabel("Longitude")
    ax7.set_ylabel("Latitude")
    save_fig(fig7, OUTPUT_DIR + "vis7_scatter_latlon.png")

    # --------- Visualization 8: Hexbin Density ---------
    fig8, ax8 = plt.subplots(figsize=(8,8))
    ax8.hexbin(sample_scatter[lon_col], sample_scatter[lat_col], gridsize=80)
    ax8.set_title("Hexbin Spatial Density of Incidents (sampled)")
    ax8.set_xlabel("Longitude")
    ax8.set_ylabel("Latitude")
    save_fig(fig8, OUTPUT_DIR + "vis8_hexbin.png")

# --------- Visualization 9: Cumulative Crimes Over Time ---------
cum = monthly.cumsum()
fig9, ax9 = plt.subplots(figsize=(12,4))
ax9.plot(cum.index, cum.values)
ax9.set_title("Cumulative Crime Count Over Time (sample)")
ax9.set_ylabel("Cumulative Count")
fig9.autofmt_xdate()
save_fig(fig9, OUTPUT_DIR + "vis9_cumulative.png")

# --------- Visualization 10: Boxplot - Daily Counts by Month ---------
df_ts['DateOnly'] = df_ts['IncidentDate'].dt.date
daily_counts = df_ts.groupby(['Year','Month','DateOnly']).size().reset_index(name='daily_count')
monthly_groups = [daily_counts[daily_counts['Month']==m]['daily_count'].values for m in range(1,13)]
fig10, ax10 = plt.subplots(figsize=(12,6))
ax10.boxplot(monthly_groups, labels=[str(m) for m in range(1,13)])
ax10.set_title("Distribution of Daily Incident Counts by Month")
ax10.set_xlabel("Month")
ax10.set_ylabel("Daily Incident Count")
save_fig(fig10, OUTPUT_DIR + "vis10_boxplot_monthly.png")



