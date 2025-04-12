import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import plotly.express as px

sns.set(style='whitegrid')
data=pd.read_csv("oci_dataset_from08122005_to31122009.csv")
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data=data.dropna(subset=['Date'])


#basic EDA prints
print("\n First 10 records of dataset")
print(data.head(10))
print("\n Last 10 records of dataset")
print(data.tail(10))
print("\n Statistic summary ")
print(data.describe())
print("\n summary information")
print(data.info())
print("\n Number of rows and columns(shape)")
print(data.shape)
print("Check for missing values")
print(data.isnull().sum())


#correlation and covariance
df=data.drop(columns=['Country','Mission','Date','OCI - Enquiries'])
print("\nCovariance:",df.cov())
corr=df.corr()
print("\nCorrelation:",df.corr())
#heatmap to visualize correlation between different columns of table
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#dbe9f4')
ax.set_facecolor('#ffffff')
sns.set(style="whitegrid")
sns.heatmap(corr,annot=True,cmap='coolwarm',fmt='.2f',linewidths=0.5)
plt.title("Correlation Matrix",fontsize=16,fontweight='bold')
plt.tight_layout()
plt.show()


#Outlier detection
#column-1
q1=np.percentile(df['OCI - Registered'],25)
q3=np.percentile(df['OCI - Registered'],75)
IQR =q3-q1
lower_bound= q1-1.5*IQR
upper_bound= q3+1.5*IQR
outliers_iqr = df[(df['OCI - Registered']<lower_bound)|(df['OCI - Registered']>upper_bound)]
print("\noutlier for ['OCI - Registered']:",outliers_iqr)
#column-2
q1=np.percentile(df['OCI - Issued'],25)
q3=np.percentile(df['OCI - Issued'],75)
IQR =q3-q1
lower_bound= q1-1.5*IQR
upper_bound= q3+1.5*IQR
outliers_iqr = df[(df['OCI - Issued']<lower_bound)|(df['OCI - Issued']>upper_bound)]
print("\noutlier for ['OCI - Issued']:",outliers_iqr)
#column-3
q1=np.percentile(df['Image - Scanned'],25)
q3=np.percentile(df['Image - Scanned'],75)
IQR =q3-q1
lower_bound= q1-1.5*IQR
upper_bound= q3+1.5*IQR
outliers_iqr = df[(df['Image - Scanned']<lower_bound)|(df['Image - Scanned']>upper_bound)]
print("\noutlier for ['Image - Scanned']:",outliers_iqr)
#column-4
q1=np.percentile(df['OCI - Granted'],25)
q3=np.percentile(df['OCI - Granted'],75)
IQR =q3-q1
lower_bound= q1-1.5*IQR
upper_bound= q3+1.5*IQR
outliers_iqr = df[(df['OCI - Granted']<lower_bound)|(df['OCI - Granted']>upper_bound)]
print("\noutlier for ['OCI - Granted']:",outliers_iqr)
#column-5
q1=np.percentile(df['OCI - Despatched to mission'],25)
q3=np.percentile(df['OCI - Despatched to mission'],75)
IQR =q3-q1
lower_bound= q1-1.5*IQR
upper_bound= q3+1.5*IQR
outliers_iqr = df[(df['OCI - Despatched to mission']<lower_bound)|(df['OCI - Despatched to mission']>upper_bound)]
print("\noutlier for ['OCI - Despatched to mission']:",outliers_iqr)
#boxplot visualization for outlier detection
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#dbe9f4')
ax.set_facecolor('#ffffff')
sns.set(style="whitegrid")
sns.boxplot(data=df)
plt.title("Boxplot Visualization for Outlier Detection",fontsize=16,fontweight='bold')
plt.tight_layout()
plt.show()


#1st object:-   Top 10 countries with maximum OCI registration
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#dbe9f4')
ax.set_facecolor('#ffffff')
sns.set(style="whitegrid")
g=((data.groupby('Country',as_index=False)['OCI - Registered'].sum()).sort_values(by='OCI - Registered',ascending=False)).head(10)
sns.barplot(y=g['Country'],x=g['OCI - Registered'],errorbar=None,palette='viridis_r',hue=g['Country'],legend=False)
plt.title("Top 10 Countries by OCI Registration",fontsize=16,fontweight='bold')
plt.xlabel('OCI - Registered',fontsize=12, fontweight='bold')
plt.ylabel('Country',fontsize=12, fontweight='bold')
for index, value in enumerate(g['OCI - Registered']):
    plt.text(value + max(g['OCI - Registered']) * 0.01, index, f'{int(value):,}', va='center', fontsize=10)
plt.tight_layout()
plt.show()


#2nd objective:- how OCI-registration and OCI-Granted relatively change over years
pivot_table=data.pivot_table(values=['OCI - Registered','OCI - Granted'],index='Date',aggfunc='sum')
yearly_data=pivot_table.resample('YE').sum()
yearly_data = yearly_data.reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#dbe9f4')
ax.set_facecolor('#ffffff')
sns.set(style="whitegrid")
sns.lineplot(data=yearly_data, x='Date', y='OCI - Registered',label='OCI - Registered', color='royalblue', marker='o', linewidth=2.5)
sns.lineplot(data=yearly_data, x='Date', y='OCI - Granted', label='OCI - Granted',color='darkorange', marker='s', linewidth=2.5)
for i in range(len(yearly_data)):
    plt.text(x=yearly_data['Date'][i],y=yearly_data['OCI - Registered'][i]+100,s=str(yearly_data['OCI - Registered'][i]),color='royalblue',fontsize=10,ha='right',va='bottom')
    plt.text(x=yearly_data['Date'][i],y=yearly_data['OCI - Granted'][i]+100,s=str(yearly_data['OCI - Granted'][i]),color='darkorange',fontsize=10,ha='left',va='top')
plt.title("Yearly OCI Registered and Granted (2005-2009)",fontsize=16,fontweight='bold')
plt.xlabel("Year",fontsize=12, fontweight='bold')
plt.ylabel("Count",fontsize=12, fontweight='bold')
plt.xticks(ticks=yearly_data['Date'], labels=yearly_data['Date'].dt.year,rotation=45)
plt.tight_layout()
plt.show()


#3rd objective:-  Top 10 missions with maximum OCI Registration
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#dbe9f4')
ax.set_facecolor('#ffffff')
sns.set(style="whitegrid")
g=((data.groupby('Mission',as_index=False)['OCI - Registered'].sum()).sort_values(by='OCI - Registered',ascending=False)).head(10)
sns.barplot(y=g['Mission'],x=g['OCI - Registered'],errorbar=None,palette='plasma',hue=g['Mission'],legend=False)
for index, value in enumerate(g['OCI - Registered']):
    plt.text(value + max(g['OCI - Registered']) * 0.01, index, f'{int(value):,}', va='center', fontsize=10)
plt.title("Top 10 Mission by OCI Registration",fontsize=16,fontweight='bold')
plt.tight_layout()
plt.show()


#4th objective:- how months or season affect registration,issuing and granting process
pivot_table=data.pivot_table(values=['OCI - Registered','OCI - Issued','OCI - Granted'],index='Date',aggfunc='sum')
monthly_data = pivot_table.resample('ME').sum()
monthly_data['Month'] = monthly_data.index.month_name().str[:3]
monthly_avg = monthly_data.groupby('Month').sum()
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun','Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_avg = monthly_avg.loc[month_order]
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#dbe9f4')
ax.set_facecolor('#ffffff')
sns.set(style="whitegrid")
plt.plot(monthly_avg.index, monthly_avg['OCI - Registered'], label='Sum of OCI - Registered', color='orangered', marker='o')
plt.plot(monthly_avg.index, monthly_avg['OCI - Issued'], label='Sum of OCI - Issued', color='gold', marker='o')
plt.plot(monthly_avg.index, monthly_avg['OCI - Granted'], label='Sum of OCI - Granted', color='green', marker='o')
plt.title("Monthly Analysis of OCI", fontsize=16, fontweight='bold')
plt.xlabel("Month",fontsize=12, fontweight='bold')
plt.ylabel("Total",fontsize=12, fontweight='bold')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
plt.tight_layout()
plt.show()


#5th objective:- To visualize total OCI-registration Recorded per year
data['Year'] = data['Date'].dt.year
oci_per_year = data.groupby('Year')["OCI - Registered"].sum()
colors = plt.cm.tab20.colors[:len(oci_per_year)]
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor('#dbe9f4')
ax.set_facecolor('#ffffff')
sns.set(style="whitegrid")
plt.pie(oci_per_year,labels=oci_per_year.index,autopct=lambda p: f'{p:.1f}%\n({int(p*oci_per_year.sum()/100):,})',startangle=90,colors=colors)
plt.title("Total OCI Registered Records Per Year",fontsize=16,fontweight='bold')
plt.tight_layout()
plt.show()


#6th objective:- (Granted v/s Dispatched) changes monthly
pivot_table=data.pivot_table(values=['OCI - Granted','OCI - Despatched to mission'],index='Date',aggfunc='sum')
monthly_data = pivot_table.resample('ME').sum()
monthly_data['Month'] = monthly_data.index.month_name().str[:3]
monthly_avg = monthly_data.groupby('Month').sum()
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun','Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_avg = monthly_avg.loc[month_order]
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#dbe9f4')
ax.set_facecolor('#ffffff')
sns.set(style="whitegrid")
plt.plot(monthly_avg.index, monthly_avg['OCI - Granted'], label='Sum of OCI - Granted', color='brown', marker='o',linewidth=2)
plt.bar(monthly_avg.index, monthly_avg['OCI - Despatched to mission'],label='OCI - Despatched to mission',color='lightgreen')
plt.title("Granted v/s Dispatched", fontsize=16, fontweight='bold')
plt.ylabel("Total",fontsize=12, fontweight='bold')
plt.xlabel("Month",fontsize=12, fontweight='bold')
plt.show()


#7th objective:- funnel chart to show how the people get reduced process
d=pd.DataFrame({'Stage':df.columns,'Count':df.sum()})
colors = ['#00FFFF', '#ff7f0e', '#FFC0CB','#2ca02c',  '#9467bd']
fig = px.funnel(d, x='Count', y='Stage',title='OCI Process Funnel',color='Stage',color_discrete_sequence=colors)
fig.update_layout(title_font_size=30,title_font_family='Arial Black',title_font_color='black')
fig.show()
