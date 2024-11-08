
# Predictive Analysis of Rental Demand for BoomBikes

Developed a regression model for BoomBikes to predict rental bike counts and analyze factors like weather, season, and holidays impacting rental demand.






## Business Context

 - Industry: Bike-Sharing 
 - Objective: BoomBikes aims to identify key factors that impact bike-sharing demand to optimize business strategies post-pandemic and boost revenue.
 - Stakeholders: Operations, marketing, finance, management, data analytics, customer service, and strategic planning teams, all of whom depend on insights into demand patterns to align service capacity, refine marketing efforts, and ensure customer satisfaction.

## Objective of Machine Learning model 
 - Key Objective : The objective of the machine learning model is to predict the daily demand for shared bikes based on various factors such as weather conditions, seasonal trends, and holiday schedules. This model will enable BoomBikes to anticipate demand fluctuations, optimize bike availability, and strategically plan to meet customer needs, thereby improving revenue and operational efficiency.

 
## Approach

- Data Cleaning: Data Cleaning involves handling missing values by using methods like mean, median, or mode imputation, and addressing outliers or anomalies to ensure data consistency and accuracy.

```bash
# Dropping the 'instant' column from the DataFrame 'df'
df.drop(columns='instant', axis=0, inplace=True)

# Displaying the DataFrame after dropping the column
df


```

- Feature Engineering : Created new columns (e.g., converting birth dates to age), merged with an additional dataset for improved insights, and saved the final dataset in CSV format for further analysis.

```bash
# Timestamp('2023-09-18 00:00:00',, tz=None)
# Deriving "days since the show started"
from datetime import date

d0 = pd.Timestamp('2018-01-01 00:00:00', tz=None)
#d0 = date(2017, 2, 28)
d1 = df.dteday

print(type(d0))
print(type(d1[0]))

delta = d1 - d0
df['day']= delta

df["day"] = df["day"].astype(str)
df["day"] = df["day"].str[:-5]
df["day"] = pd.to_numeric(df["day"], errors="coerce")
df.head()

# Mapping season numbers to their corresponding names
df_new['season'] = df_new['season'].map({1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'})
# Mapping weather situation numbers to their corresponding labels
df_new['weathersit'] = df_new['weathersit'].map({
    1: 'Clear', 
    2: 'Mist', 
    3: 'Light Rain/Snow', 
    4: 'Heavy Rain/Snow'
})
df_new

# Create dummy variables (one-hot encoding) for 'season' and 'weathersit' and convert them to 1s and 0s
dummies1 = pd.get_dummies(df_new['season'], drop_first=True).astype(int)
dummies2 = pd.get_dummies(df_new['weathersit'], drop_first=True).astype(int)

print(dummies1)
print(dummies2)
df_new=pd.concat([df_new,dummies1],axis=1)
df_new=pd.concat([df_new,dummies2],axis=1)

df_new.drop(columns=['season','weathersit'],inplace=True)
# # Display the updated dataframe
df_new.head()



```
- Exploratory Data Analysis (EDA) : Conducted univariate, segmented univariate, bivariate, and multivariate analyses to uncover key insights and created a correlation matrix.

```bash
#Displaying boxplot for num_cols
for col in num_cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df_new[col])
    plt.title(f'Distribution of {col}')
    plt.show()
    # plt.savefig(f'Distribution of {col}')

#Displaying countplot for cat_cols
for col in cat_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(y=df_new[col])
    plt.title(f'Countplot of {col}')
    plt.show()


plt.figure(figsize=(20, 12))

# Scatter plot for 'yr' vs 'cnt'
plt.subplot(3, 4, 1)
sns.scatterplot(x='yr', y='cnt', data=df_new)

# Scatter plot for 'mnth' vs 'cnt'
plt.subplot(3, 4, 2)
sns.scatterplot(x='mnth', y='cnt', data=df_new)

# Scatter plot for 'holiday' vs 'cnt'
plt.subplot(3, 4, 3)
sns.scatterplot(x='holiday', y='cnt', data=df_new)

# Scatter plot for 'weekday' vs 'cnt'
plt.subplot(3, 4, 4)
sns.scatterplot(x='weekday', y='cnt', data=df_new)

# Scatter plot for 'workingday' vs 'cnt'
plt.subplot(3, 4, 5)
sns.scatterplot(x='workingday', y='cnt', data=df_new)

# Scatter plot for 'temp' vs 'cnt'
plt.subplot(3, 4, 6)
sns.scatterplot(x='temp', y='cnt', data=df_new)

# Scatter plot for 'temp' vs 'cnt'
plt.subplot(3, 4, 7)
sns.scatterplot(x='hum', y='cnt', data=df_new)

# Scatter plot for 'temp' vs 'cnt'
plt.subplot(3, 4, 8)
sns.scatterplot(x='windspeed', y='cnt', data=df_new)

# Scatter plot for 'temp' vs 'cnt'
plt.subplot(3, 4, 9)
sns.scatterplot(x='casual', y='cnt', data=df_new)

plt.subplot(3, 4, 10)
sns.scatterplot(x='registered', y='cnt', data=df_new)

plt.subplot(3, 4, 11)
sns.scatterplot(x='day', y='cnt', data=df_new)

plt.subplot(3, 4, 12)
sns.scatterplot(x='Spring', y='cnt', data=df_new)
plt.show()


# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (16, 10))
sns.heatmap(df_new.corr(), annot = True, cmap="YlGnBu")
plt.show()

```

- Modeling: multiple regression model were built and evaluated to predict the daily demand for shared bikes. The dataset included variables related to weather conditions, seasons, holidays, and other temporal factors. During the modeling process, exploratory data analysis (EDA) was first conducted to identify trends and correlations between variables. Feature engineering was performed to enhance the dataset, such as creating new variables and normalizing certain features to improve model performance. Key modeling techniques included linear regression and regularization methods like Ridge and Lasso regression, which helped address multicollinearity and enhance the robustness of the model. Hyperparameter tuning was applied to optimize each model, and the models were assessed based on metrics like R-squared, RMSE, and MAE to select the best-performing one. The final model provides valuable insights into the factors that significantly impact bike demand, enabling BoomBikes to plan strategically for varying demand scenarios.

```bash

df_train,df_test=train_test_split(df,train_size=0.7,random_state=100)
print(df_train.shape)
print(df_test.shape)

# Initializing a MinMaxScaler to scale numerical features to a range of [0, 1]
scalar = MinMaxScaler()

# List of numerical variables to be scaled
num_vars = ['mnth', 'weekday', 'temp', 'hum', 'windspeed', 'casual', 'registered', 'cnt', 'day']

# Applying the MinMaxScaler to the specified numerical variables in the 'df_train' DataFrame
df_train[num_vars] = scalar.fit_transform(df_train[num_vars])

# Displaying the first few rows of the updated DataFrame 'df_train' to verify the scaling
df_train.head()

# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (16, 10))
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()

# Extracting the target variable 'cnt' from the 'df_train' DataFrame for training
y_train=df_train.cnt

# Defining the feature set 'X_train' by selecting specific columns from the 'df_train' DataFrame
X_train=df_train[['yr','mnth','holiday','weekday','workingday','temp','hum','windspeed','casual','registered','day','Spring','Summer','Winter','Light Rain/Snow','Mist']]

# [['yr','mnth','holiday','weekday','workingday','temp','hum','windspeed','casual','registered','day','Spring','Summer','Winter','Light Rain/Snow','Mist']]


X_train_sm6=X_train[['yr','holiday','temp','hum','windspeed','Spring','Summer','Winter','Light Rain/Snow','Mist']]

X_train_sm6=sm.add_constant(X_train_sm6)

lr6=sm.OLS(y_train,X_train_sm6)

lr_model6=lr6.fit()

print(lr_model6.params)

print(lr_model6.summary(),'\n')

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_sm6.columns
vif['VIF'] = [variance_inflation_factor(X_train_sm6.values, i) for i in range(X_train_sm6.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)
print(vif)

y_train_pred6=lr_model6.predict(X_train_sm6)
res=y_train-y_train_pred6
sns.distplot(res)

# Testing dataset
df_test[num_vars]=scalar.transform(df_test[num_vars])
df_test.head()

y_test=df_test.cnt
X_test=df_test[['yr','mnth','holiday','weekday','workingday','temp','hum','windspeed','casual','registered','day','Spring','Summer','Winter','Light Rain/Snow','Mist']]

# [['yr','mnth','holiday','weekday','workingday','temp','hum','windspeed','casual','registered','day','Spring','Summer','Winter','Light Rain/Snow','Mist']]


X_test_sm1=X_test[['yr','holiday','temp','hum','windspeed','Spring','Summer','Winter','Light Rain/Snow','Mist']]

X_test_sm1=sm.add_constant(X_test_sm1)

y_test_pred=lr_model6.predict(X_test_sm1)

print(r2_score(y_true=y_test,y_pred= y_test_pred))

plt.figure(figsize = (10, 5))
# Plotting the distribution of residuals
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
sns.histplot(res, kde=True, stat="density", color='blue', bins=20)
plt.title('Distribution of Residuals of y_train-y_train_pred')
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.grid(True)

# Creating the Q-Q plot
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
sm.qqplot(res, line='s', ax=plt.gca())
plt.title('Q-Q Plot of Residuals y_train-y_train_pred')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()

```


## Conclusion


The final model highlights that season and weather conditions are key drivers of bike demand. Adverse weather, particularly Light Rain/Snow, has a strong negative effect, reducing rentals significantly (coefficient: -2082.54). In contrast, the year variable (coefficient: 2005.31) shows a positive trend, indicating increasing demand over time. Winter also positively impacts demand (coefficient: 757.01), reflecting seasonal preferences for biking. These insights underscore the importance of considering both seasonality and weather to accurately predict bike demand and optimize resource allocation.

