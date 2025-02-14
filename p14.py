# importting Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')

#Reading the dataset
df=pd.read_csv("C:\\Users\\Sambhaji\\numpy2\\numpy\\winequality-red.csv")
print(df.head())
#analyzing the data
#shape of the data
print(df.shape)

#data information
print(df.info())

#describing of the data
print(df.describe())

#checking for coulumns
#column to list
print(df.columns.tolist())

#checking for missing values
print(df.isnull().sum())

#checking for duplicate values
print(df.nunique())

#step 4 :Univariate Analysis


# Assuming 'df' is your DataFrame
quality_counts = df['quality'].value_counts()

# Using Matplotlib to create a count plot
plt.figure(figsize=(8, 6))
plt.bar(quality_counts.index, quality_counts, color='deeppink')
plt.title('Count Plot of Quality')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.show()

#kernal Density plot
# Set seaborn style
sns.set_style=("Darkgrid")

# Identify numerical columns
numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns

# Plot distribution of each numerical feature
plt.figure(figsize=(14, len(numerical_columns) * 3))
for idx, feature in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 2, idx)
    sns.histplot(df[feature], kde=True)
    plt.title(f"{feature} | Skewness: {round(df[feature].skew(), 2)}")

# Adjust layout and show plots
plt.tight_layout()
plt.show()


#Swarm Plot
# Assuming 'df' is your DataFrame
plt.figure(figsize=(10, 8))

# Using Seaborn to create a swarm plot
sns.swarmplot(x="quality", y="alcohol", data=df, palette='viridis')

plt.title('Swarm Plot for Quality and Alcohol')
plt.xlabel('Quality')
plt.ylabel('Alcohol')
plt.show()

#Step 5: Bivariate Analysis
#Pair Plot

# Set the color palette
sns.set_palette("Pastel1")

# Assuming 'df' is your DataFrame
plt.figure(figsize=(10, 6))

# Using Seaborn to create a pair plot with the specified color palette
sns.pairplot(df)

plt.suptitle('Pair Plot for DataFrame')
plt.show()

#Violin Plot

# Assuming 'df' is your DataFrame
df['quality'] = df['quality'].astype(str)  # Convert 'quality' to categorical

plt.figure(figsize=(10, 8))

# Using Seaborn to create a violin plot
sns.violinplot(x="quality", y="alcohol", data=df, palette={
               '3': 'lightcoral', '4': 'lightblue', '5': 'lightgreen', '6': 'gold', '7': 'lightskyblue', '8': 'lightpink'}, alpha=0.7)

plt.title('Violin Plot for Quality and Alcohol')
plt.xlabel('Quality')
plt.ylabel('Alcohol')
plt.show()

#Box Plot

#plotting box plot between alcohol and quality
sns.boxplot(x='quality', y='alcohol', data=df)


#Step 6: Multivariate Analysis
#Correlation Matrix

# Assuming 'df' is your DataFrame
plt.figure(figsize=(15, 10))

# Using Seaborn to create a heatmap
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='Pastel2', linewidths=2)

plt.title('Correlation Heatmap')
plt.show()

#2 EDA titanic dataset
# Load the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("C:\\Users\\Sambhaji\\numpy2\\numpy\\titanic.csv")

# View the data
print(df.head())


#1 basic information about data-EDA
print(df.info())

#Describe the data
print(df.describe())

#2 Duplicate values
print(df.duplicated().sum())

#3 Unique values in the data
#unique values

print(df['Pclass'].unique())

print(df['Survived'].unique())

print(df['Sex'].unique())

import numpy as np

# Array of integers
array1 = np.array([3, 1, 2], dtype=np.int64)
print("Array 1:", array1)

# Array of integers (two values)
array2 = np.array([0, 1], dtype=np.int64)
print("Array 2:", array2)

# Array of objects (strings)
array3 = np.array(['male', 'female'], dtype=object)
print("Array 3:", array3)

# 4. Visualize the Unique counts
#plot the unique values
# Assuming df is your DataFrame
unique_values = df['Pclass'].unique()
print("Unique values in Pclass:", unique_values)

#Now plot the countplot
sns.countplot(x='Pclass', data=df)
plt.show()

#5 Find the Null values
#Find null values

print(df.isnull().sum())

#6.Replace the Null values
#Replace null values
print(df.replace(np.nan,'0',inplace=True))


#check the changes now
print(df.isnull().sum())

#7 Know the datatypes
#Datatypes
print(df.dtypes)

#8. Filter the Data
#Filter data
print(df[df['Pclass']==1].head())

#9 A quick box plot
#Boxplot
print(df[['Fare']].boxplot())
plt.show()

#10 Correlation plot-

# Create a correlation matrix
# Ensure only numeric data is processed
numeric_df = df.select_dtypes(include=['float64', 'int64'])
print(numeric_df)

corr_matrix = numeric_df.corr()

# Plot the heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1.0)

# Display the plot
plt.show()


#3 EDA loading the data
#Read in the csv file and convert to a Pandas dataframe

df = pd.read_csv("https://raw.githubusercontent.com/siglimumuni/Datasets/master/customer-data.csv")

# Display the first few rows of the dataframe
print(df.head())

#Viewing the dataframe
#Return number of rows and coulumns
print(df.shape)

#Return first 5 rows of the dataset
print(df.head())

#Return info on the dataset
print(df.info())

#Display number missing values per column
print(df.isnull().sum())

#checking the mean credit score for each income gro
print(df.groupby(by="income")["credit_score"].mean())

#Create a function to impute missing values based on mean credit score for each income group
def impute_creditscore(df, income_classes):
    """This function takes a DataFrame and a list of income groups and imputes the missing values
    of credit scores for each group based on the mean credit score of that group"""

    # Calculate the mean credit score for each income group and store them in a dictionary
    income_means = df.groupby('income')['credit_score'].mean().to_dict()

    # Iterate through each income group
    for income_class in income_classes:
        # Create a subset mask
        mask = df['income'] == income_class
        
        # Get the mean credit score for the income group
        mean_credit_score = income_means.get(income_class, None)
        
        # Check if mean_credit_score is not NaN
        if mean_credit_score is not None:
            # Fill the missing values with the mean credit score of the group
            df.loc[mask, 'credit_score'] = df.loc[mask, 'credit_score'].fillna(mean_credit_score)
        else:
            print(f"No valid mean credit score available for income class: {income_class}")

    return df

#Apply the function to the dataframe
income_groups = ["poverty","upper class","middle class","working class"]

#Define the income groups
df=impute_creditscore(df,income_groups)

#check for missing values
print(df.isnull().sum())

#Check the mean annual mileage for the different driving experience groups
print(df.groupby(by="driving_experience")["annual_mileage"].mean())#Check the mean annual mileage for the different driving experience groups

#Calculate mean for annual_mileage column
mean_mileage = df["annual_mileage"].mean()

#Fill in null values using the column mean
print(df["annual_mileage"].fillna(mean_mileage,inplace=True))

#Check for null values
print(df.isna().sum())

#Delete the id and postal_code columns
print(df.drop(["id","postal_code"],axis=1,inplace=True))

#check the count for each category in the "gender" column
print(df["gender"].value_counts())

#Create a countplot to visualize the count of each category in the gender column.
sns.countplot(data=df,x="gender")
plt.title("Number of Clients per Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Clients")
plt.show()

#Define plot size
plt.figure(figsize=[6,6])

#Define column to use
data = df["income"].value_counts(normalize=True)

#Define labels
labels = ["upper class","middle class","poverty","working class"]

#Define color palette
colors = sns.color_palette('pastel')

#Create pie chart
plt.pie(data,labels=labels,colors=colors, autopct='%.0f%%')
plt.title("Proportion of Clients by Income Group")
plt.show()

#Create a countplot to visualize the count of each category in the education column 
plt.figure(figsize=[8,5])
sns.countplot(data=df,x="education",order=["university","high school","none"],color="orange")
plt.title("Number of Clients per Education Level")
plt.show()

#Return summary statistics for the "credit_score" column
print(df["credit_score"].describe())

#Plot a histogram using the "credit_score" column
plt.figure(figsize=[8,5])
sns.histplot(data=df,x="credit_score",bins=40).set(title="Distribution of credit scores",ylabel="Number of clients")
plt.show()

#Plot a histogram using the "annual_mileage" column
plt.figure(figsize=[8,5])
sns.histplot(data=df,x="annual_mileage",bins=20,kde=True).set(title="Distribution of Annual Mileage",ylabel="Number of clients")
plt.show()

#Create a scatter plot to. show relationship between "annual_mileage" and "speeding_violations"
plt.figure(figsize=[8,5])
plt.scatter(data=df,x="annual_mileage",y="speeding_violations")
plt.title("Annual Mileage vrs Speeding Violations")
plt.ylabel("Speeding Violations")
plt.xlabel("Annual Mileage")
plt.show()

#Create a correlation matrix to show relationship between select variables
corr_matrix = df[["speeding_violations","DUIs","past_accidents"]].corr()
print(corr_matrix)

#Create a heatmap to visualize correlation
plt.figure(figsize=[8,5])
sns.heatmap(corr_matrix,annot=True,cmap='Reds')
plt.title("Correlation between Selected Variables")
plt.show()

#Check the mean annual mileage per category in the outcome column
print(df.groupby('outcome')['annual_mileage'].mean())

#Plot two boxplots to compare dispersion
sns.boxplot(data=df,x='outcome', y='annual_mileage')
plt.title("Distribution of Annual Mileage per Outcome")
plt.show()

#Create histograms to compare distribution 
sns.histplot(df,x="credit_score",hue="outcome",element="step",stat="density")
plt.title("Distribution of Credit Score per Outcome")
plt.show()

#Create a new "claim rate" column
df['claim_rate'] = np.where(df['outcome']==True,1,0)
df['claim_rate'].value_counts()

#Plot the average claim rate per age group
plt.figure(figsize=[8,5])
df.groupby('age')['claim_rate'].mean().plot(kind="bar")
plt.title("Claim Rate by Age Group")
plt.show()

#Plot the average claim rate per vehicle year category
plt.figure(figsize=[8,5])
df.groupby('vehicle_year')['claim_rate'].mean().plot(kind="bar")
plt.title("Claim Rate by Vehicle Year")
plt.xlabel("Vehicle Year")
plt.ylabel("Vehicle Year")
plt.show()

#Create an empty figure object
fig, axes = plt.subplots(1,2,figsize=(12,4))

#Plot two probability graphs for education and income
for i,col in enumerate(["education","income"]):
    sns.histplot(df, ax=axes[i],x=col, hue="outcome",stat="probability", multiple="fill", shrink=.8,alpha=0.7)
    axes[i].set(title="Claim Probability by "+ col,ylabel=" ",xlabel=" ")

#Create a pivot table for education and income with average claim rate as values
edu_income = pd.pivot_table(data=df,index='education',columns='income',values='claim_rate',aggfunc='mean')
print(edu_income)

#Create a heatmap to visualize income, education and claim rate
plt.figure(figsize=[8,5])
sns.heatmap(edu_income,annot=True,cmap='coolwarm',center=0.117)
plt.title("Education Level and Income Class")
plt.show()

#Create pivot table for driving experience and marital status with average claim rate as values
driv_married = pd.pivot_table(data=df,index='driving_experience',columns='married',values='claim_rate')

#Create a heatmap to visualize driving experience, marital status and claim rate
plt.figure(figsize=[8,5])
sns.heatmap(driv_married,annot=True,cmap='coolwarm', center=0.117)
plt.title("Driving Experience and Marital Status")
plt.show()

#Create pivot table for gender and family status with average claim rate as values
gender_children = pd.pivot_table(data=df,index='gender',columns='children',values='claim_rate')

#Create a heatmap to visualize gender, family status and claim rate
plt.figure(figsize=[8,5])
sns.heatmap(gender_children,annot=True,cmap='coolwarm', center=0.117)
plt.title("Gender and Family Status")
plt.show()

#EDA 3 Step 1 Load and Explore the Data
import pandas as pd
import matplotlib.pyplot as plt  # Import for plotting

# Load the dataset
data = pd.read_csv(r"C:\Users\Sambhaji\numpy2\numpy\sales_data_sample.csv", encoding='ISO-8859-1')

# Clean up column names by stripping any leading/trailing spaces
data.columns = data.columns.str.strip()

# Step 1: Print the columns in the dataset to inspect them
print("Columns in the dataset:", data.columns)

# Step 2: Display the first few rows of the dataset to inspect the data
print("First few rows of the dataset:")
print(data.head())

# Step 3: Check if 'QUANTITYORDERED' and 'ORDERDATE' columns exist
if 'QUANTITYORDERED' not in data.columns:
    print("The 'QUANTITYORDERED' column is not present in the dataset.")
else:
    if 'PRICEEACH' not in data.columns:
        print("The 'PRICEEACH' column is not present in the dataset.")
    else:
        # Proceed to create 'Sales_Amount' if both 'QUANTITYORDERED' and 'PRICEEACH' columns exist
        data['Sales_Amount'] = data['QUANTITYORDERED'] * data['PRICEEACH']
        print("Sales_Amount column created successfully.")

# Step 4: Data Cleaning
# Check for missing values and handle them
print("Missing values in the dataset:")
print(data.isnull().sum())

# Handle missing values for numeric columns by filling with the mean of that column
# Select only numeric columns for filling missing values
numeric_columns = data.select_dtypes(include=['number']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Handle missing values for non-numeric columns separately (example: 'Product' and 'Date' columns)
# For categorical columns, you can fill missing values with a placeholder like "Unknown"
non_numeric_columns = data.select_dtypes(exclude=['number']).columns
data[non_numeric_columns] = data[non_numeric_columns].fillna("Unknown")

# Step 5: Handle 'ORDERDATE' column - Convert to datetime if not already
if 'ORDERDATE' in data.columns:
    data['ORDERDATE'] = pd.to_datetime(data['ORDERDATE'], errors='coerce')
else:
    print("The 'ORDERDATE' column is not present in the dataset.")

# Step 6: Calculate Monthly Sales
# Group by Month and calculate total sales
data['Month'] = data['ORDERDATE'].dt.to_period('M')  # Convert ORDERDATE to Month period
monthly_sales = data.groupby('Month')['Sales_Amount'].sum()

# Plot Monthly Sales
monthly_sales.plot(kind='bar', title='Monthly Sales', ylabel='Sales Amount', xlabel='Month', color='skyblue')
plt.show()

# Step 7: Identify Top 5 Best-Selling Products
top_products = data.groupby('PRODUCTCODE')['QUANTITYORDERED'].sum().nlargest(5)

# Plot Top Products
top_products.plot(kind='bar', title='Top 5 Best-Selling Products', ylabel='Quantity Sold', xlabel='Product', color='orange')
plt.show()

# Step 8: Calculate Average Sale Amount per Transaction
avg_sale_amount = data['Sales_Amount'].mean()
print(f"The average sale amount per transaction is: ${avg_sale_amount:.2f}")

# Step 9: Analyze Monthly Sales Trends
# Extract Month Names
data['Month_Name'] = data['ORDERDATE'].dt.strftime('%B')

# Group by month and calculate total sales
monthly_sales_trends = data.groupby('Month_Name')['Sales_Amount'].sum()

# Sort by month order
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
monthly_sales_trends = monthly_sales_trends[month_order]

# Plot Sales Trends by Month
monthly_sales_trends.plot(kind='line', title='Sales Trends by Month', ylabel='Sales Amount', xlabel='Month', color='green')
plt.show()

