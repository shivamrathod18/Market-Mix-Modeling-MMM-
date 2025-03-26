<h1> Understanding Marketing Mix Modeling </h1>

<p align="center">

<h2>Description</h2>

<h3>Introduction</h3>

The goal of this project is to explore and apply the basic concepts behind Marketing Mix Modeling.

MMM is a family of statistical tehcniques applied to historical data to understand the attribution and the “marginal contribution” of individual factors (facebook ads, amazon ads…) on our ROI. Typical goals in MMM are:

- Statistical attribution of channels
- ROI Contribution of adv channels
- Forecasting sales
- Budget Allocation

💡 It is crucial to distinguish:
- **MMM (Marketing Mix Modeling)**: ignores the cookies, and tracks how efficient is the overall investment
- **MTA (Multi-Touch Attribution)**: using cookies to track the behavior of the single user

... MMM is on the rise because the importance of offline channels combined with the recent no-cookie policies stemming from the GDPR and the CCPA are making it very tough to track the user along the various media touchpoints.

<h3>MMM in Practice</h3>

In practice, an MMM project results in a regression algorithm, whereby the aim is to forecast a given KPI (i.e. sales, impressions...) using a set of predictors including data about online and offline campaigns as well as external factors such as seasonality and competitors' initiatives. The algorithm takes the form of OLS Linear Regression or Bayesian Regression; the former is more popular, but the latter is gaining ground thanks to Google's [LightWeight-MMM Python library](https://github.com/google/lightweight_mmm).

💡 This project follows the OLS Linear Regression approach; it ignores the effect of external factors (no control variables) and focuses on sales.

***-> Sales = constant + predictors effect + control variables effect***

<h3>The Nature of Ads</h3>

While it is possible to just fit a Linear Regression model to a raw dataset containing data about sales and advertising investment on different media channels for a given period, it is important to take into account 2 concepts when dealing with MMM:

1. **AdStock:** There is a decay (or “lag”) effect in advertising, whereby an ad watched today might generate a sale in 2 weeks. This lag might be anywhere from 1 to 7 weeks, and it varies according to the media type, the industry, and the specific product. 

2. **Diminishing returns:** The marginal benefit (in terms of sales, impressions, new customers...) of increasing ad investment from $1 to $100 is greater than the one that results from increasing from $500 to $599.

These two concepts cannot be ignored. In particular, in every MMM project the ad investment for a given channel should always be transformed into AdStock using one of 4 functional forms: *classic (exponential) AdStock, Hill AdStock, Carryover, or [Weibull AdStock](https://github.com/annalectnl/weibull-adstock/blob/master/adstock_weibull_annalect.pdf)*.

💡 In this project the AdStock transformation is the most basic one: ***AdStock(t) = AdStock(t-1) * Decay_Factor + Investment(t)***

<h3>The MMM Process</h3>

This project follows a linear and simple methodology:
1. **EDA**
2. **PreProcessing**
3. **Marketing Mix Modeling**
4. **Analysis**

💡 In a real MMM project the process would be iterative:
- The decay factor in the AdStock formula would be optimized, and not guessed
- After the analysis, the budget would be optimized (here the budget is ignored)

<h2>Code</h2>

<h3> Setup & PreProcessing </h3>

```py
#LIBRARIES

#Basic Analytics
import pandas as pd
import plotly.express as px
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

#ML & Statistics
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#DATA IMPORT
df = pd.read_csv('/Users/matteo-stelluti/Desktop/Work Applications/Companies/Ekimetrics/MMM Project/adv_data_simple.csv')

df.head()
```

<img src="https://i.imgur.com/nrnpk5v.png" height="50%" width="50%" alt="df.head"/>

```py
df.shape

df = df.drop('Date', axis=1)
```

<h3> Exploratory Data Analysis </h3>

```p
corr = df.corr()
sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, annot = True, cmap = sns.diverging_palette(220, 20, as_cmap=True))
```

<img src="https://i.imgur.com/qaN0j7t.png" height="50%" width="50%" alt="heat map"/>


```py
sns.pairplot(df)
```

<img src="https://i.imgur.com/xEYIhTl.png" height="50%" width="50%" alt="Correlation plot"/>


<h3> Marketing Mix Modeling </h3>

<h4> AdStock Transformation </h4>

```py
#Define a simple function to transform the ad spend into AdStock
def adstocked(advertising,adstock_rate=0.5):
    
    adstocked_advertising = []
    for i in range(len(advertising)):
        if i == 0: 
            adstocked_advertising.append(advertising[i])
        else:
            adstocked_advertising.append(advertising[i] + adstock_rate * adstocked_advertising[i-1])            
    return adstocked_advertising
    
#Create new columns with AdStock, one for each media
df['FB_adstock']=adstocked(advertising=df['FB'])
df['TV_adstock']=adstocked(advertising=df['TV'])
df['Radio_adstock']=adstocked(advertising=df['Radio'])    

df.head()
```

<img src="https://i.imgur.com/Mufmx8j.png" height="30%" width="30%" alt="new df.head"/>

We can now visualize the AdStock plots to better understand each media. <br/>
**Note:** The "decay factor" has been set to .5 for each media, regardless of their nature, and ignoring the industry.

```py
#FACEBOOK
fig, ax = plt.subplots(figsize=(10,5), dpi=100)
plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=None)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.yaxis.grid(b=False, which='major', color='gray', linestyle='--')

ax.set_ylim([0, 1.3*max(df['FB_adstock'])])

bar = ax.bar(np.arange(len(df['FB'])), df['FB'], width=0.8, color='gray', alpha=0.7, label='Actual FB')
line = ax.plot(np.arange(len(df['FB'])), df['FB_adstock'], 
               marker='o', markersize=3, linestyle='-', linewidth=2, color='blue', alpha=0.7, label='FB AdStock')

ax.set_xlabel('Time')
ax.set_ylabel('Adstock & Advertising')
#ax.set_title('advertising adstock transformation')

ax.annotate("Advertising Adstock Transformation", (np.mean(np.arange(len(df['FB']))), 315),
            verticalalignment='bottom', horizontalalignment='center',
            fontsize=15, color='#681963')
ax.set_xticks(np.arange(len(df['FB'])))
ax.set_xticklabels(labels=np.arange(len(df['FB'])), minor=False, fontsize=7, rotation=90)
ax.legend(loc='center right')  
plt.show()

#TV
fig, ax = plt.subplots(figsize=(10,5), dpi=100)
plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=None)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.yaxis.grid(b=False, which='major', color='gray', linestyle='--')

ax.set_ylim([0, 1.3*max(df['TV_adstock'])])

bar = ax.bar(np.arange(len(df['TV'])), df['TV'], width=0.8, color='gray', alpha=0.7, label='Actual TV')
line = ax.plot(np.arange(len(df['TV'])), df['TV_adstock'], 
               marker='o', markersize=3, linestyle='-', linewidth=2, color='blue', alpha=0.7, label='TV AdStock')

ax.set_xlabel('Time')
ax.set_ylabel('Adstock & Advertising')
#ax.set_title('advertising adstock transformation')

ax.annotate("Advertising Adstock Transformation", (np.mean(np.arange(len(df['TV']))), 315),
            verticalalignment='bottom', horizontalalignment='center',
            fontsize=15, color='#681963')
ax.set_xticks(np.arange(len(df['TV'])))
ax.set_xticklabels(labels=np.arange(len(df['FB'])), minor=False, fontsize=7, rotation=90)
ax.legend(loc='center right')  
plt.show()

#RADIO
fig, ax = plt.subplots(figsize=(10,5), dpi=100)
plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=None)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.yaxis.grid(b=False, which='major', color='gray', linestyle='--')

ax.set_ylim([0, 1.3*max(df['Radio_adstock'])])

bar = ax.bar(np.arange(len(df['Radio'])), df['Radio'], width=0.8, color='gray', alpha=0.7, label='Actual Radio')
line = ax.plot(np.arange(len(df['Radio'])), df['Radio_adstock'], 
               marker='o', markersize=3, linestyle='-', linewidth=2, color='blue', alpha=0.7, label='Radio AdStock')

ax.set_xlabel('Time')
ax.set_ylabel('Adstock & Advertising')
#ax.set_title('advertising adstock transformation')

ax.annotate("Advertising Adstock Transformation", (np.mean(np.arange(len(df['Radio']))), 315),
            verticalalignment='bottom', horizontalalignment='center',
            fontsize=15, color='#681963')
ax.set_xticks(np.arange(len(df['Radio'])))
ax.set_xticklabels(labels=np.arange(len(df['FB'])), minor=False, fontsize=7, rotation=90)
ax.legend(loc='center right')  
plt.show()
```

<img src="https://i.imgur.com/8C3LXkb.png" height="50%" width="50%" alt="FB AdStock"/>
<img src="https://i.imgur.com/oEFsnkV.png" height="50%" width="50%" alt="TV AdStock"/>
<img src="https://i.imgur.com/glDERBG.png" height="50%" width="50%" alt="Radio AdStock"/>

<h4> Model Fitting: OLS Linear Regression </h4>

```py
#SETUP
predictors = ['FB','TV','Radio']

X = df[predictors]
y = df['sales']

#PARTITIONING
X_train,X_test,y_train,y_test = train_test_split(X,y,shuffle=False,test_size=.2)

#FITTING
ols = LinearRegression()
ols.fit(X_train,y_train)
```

<h4> Model Interpretation </h4>

```py
#We can get a detailed summary using statsmodel (running LM only on the training set!)
X_train = sm.add_constant(X_train) #Even if the predictors are 0, we still have sales
ols_sm = sm.OLS(y_train,X_train).fit()

ols_sm.summary()
```

<img src="https://i.imgur.com/nLM0wpf.png" height="50%" width="50%" alt="new df.head"/>

**Interpretation:**
  - The OLS equation is: *sales = 2.98 + 0.05FB + 0.1720TV - 0.013Radio*.
    - It's crucial to remember that the unit of measure is not $1 of ad spent, but it's AdStock.
    - All the coefficients are statistically significant with a 99.9% confidence interval, except Radio, which is also illogical as it's <0.
  - R2 is 0.909, meaning that the model can explain 91% of the variation in y on the training set.
  - Also noteworthy, the Durbin-Watson test returned 2.21, indicanting that there is no evidence of autocorrelation in the training set.


<h3> What are the next steps? </h3>

On the one hand, we could stop here and implement this OLS model creating a simulator in Excel, whereby the user inserts a given budget to allocate to each media channel and the model would return the sales.

On the other hand, the model needs a lot of re-work and iterations before being applicable:
  - The AdStock decaying factor needs to be optimized, since it will most likely be different for each media type.
  - We can reduce the variance of the model (i.e. make it better with unseen data) by cross validation and regularization.
  - The "Radio" variable can be removed considering it's not statistically significant and that it has a negative impact.
  - Once the model is finalized, even the budget allocation can be turned into an optimization problem.
