# Problem Set 2, due September 18 at 11:59pm

## Introduction to the assignment

For this assignment, you will be using data from the [Progresa program](http://en.wikipedia.org/wiki/Oportunidades), a government social assistance program in Mexico. This program, as well as the details of its impact, are described in the paper "[School subsidies for the poor: evaluating the Mexican Progresa poverty program](http://www.sciencedirect.com/science/article/pii/S0304387803001858)", by Paul Shultz (available on bCourses). Please familiarize yourself with the PROGRESA program before beginning this problem set, so you have a rough sense of where the data come from and how they were generated. If you just proceed into the problem set without understanding Progresa or the data, it will be very difficult!

The goal of this problem set is to implement some of the basic econometric techniques that you are learning in class to measure the impact of Progresa on secondary school enrollment rates. The timeline of the program was:

 * Baseline survey conducted in 1997
 * Intervention begins in 1998, "Wave 1" of data collected in 1998
 * "Wave 2 of data" collected in 1999
 * Evaluation ends in 2000, at which point the control villages were treated. 
 
When you are ready, download the progresa_sample.csv data from bCourses. The data are actual data collected to evaluate the impact of the Progresa program.  In this file, each row corresponds to an observation taken for a given child for a given year. There are two years of data (1997 and 1998), and just under 40,000 children who are surveyed in each year. For each child-year observation, the following variables are collected:

| Variable name | Description|
|------|------|
|year	  |year in which data is collected
|sex	  |male = 1|
|indig	  |indigenous = 1|
|dist_sec |nearest distance to a secondary school|
|sc	      |enrolled in school in year of survey|
|grc      |grade enrolled|
|fam_n    |family size|
|min_dist |	min distance to an urban center|
|dist_cap |	min distance to the capital|
|poor     |	poor = 1|
|progresa |treatment =1|
|hohedu	  |years of schooling of head of household|
|hohwag	  |monthly wages of head of household|
|welfare_index|	welfare index used to classify poor|
|hohsex	|gender of head of household (male=1)|
|hohage	|age of head of household|
|age	|years old|
|folnum	|individual id|
|village|	village id|
|sc97	|schooling in 1997|

---

## Part 1: Descriptive analysis

### 1.1	Summary Statistics

Present summary statistics (mean and standard deviation) for all of the demographic variables in the dataset (i.e., everything except year, folnum, village). Present these in a single table alphabetized by variable name. Do NOT simply expect the grader to scroll through your output!


```python
# your code here
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
progresa_df = pd.read_csv('progresa_sample.csv')
print(progresa_df.progresa)
```

    0            0
    1            0
    2            0
    3            0
    4            0
    5            0
    6        basal
    7        basal
    8        basal
    9        basal
    10       basal
    11       basal
    12       basal
    13       basal
    14       basal
    15       basal
    16       basal
    17       basal
    18       basal
    19       basal
    20       basal
    21       basal
    22       basal
    23       basal
    24       basal
    25       basal
    26       basal
    27       basal
    28       basal
    29       basal
             ...  
    77220    basal
    77221    basal
    77222    basal
    77223    basal
    77224    basal
    77225    basal
    77226    basal
    77227    basal
    77228    basal
    77229    basal
    77230    basal
    77231    basal
    77232    basal
    77233    basal
    77234    basal
    77235    basal
    77236    basal
    77237    basal
    77238    basal
    77239    basal
    77240    basal
    77241    basal
    77242    basal
    77243    basal
    77244    basal
    77245    basal
    77246    basal
    77247    basal
    77248    basal
    77249    basal
    Name: progresa, Length: 77250, dtype: object
    


```python
print('poor levels: ', progresa_df['poor'].unique().tolist())
print('progresa levels: ', progresa_df['progresa'].unique().tolist())
```

    poor levels:  ['pobre', 'no pobre']
    progresa levels:  ['0', 'basal']
    


```python
# create dictionary to convert 'poor' values to int
poor_dict = {"no pobre":0 ,"pobre":1} 
# create new variable with pobre dictionary
progresa_df['poor'] = progresa_df['poor'].replace(poor_dict)
# # convert poor_int to integer
progresa_df['poor'] = pd.to_numeric(progresa_df['poor'])


```


```python
# create dictionary to convert 'progresa' values to int
progresa_dict = {"0":0 , "basal":1}

# create new variable with pobre dictionary
progresa_df['progresa'] = progresa_df['progresa'].replace(progresa_dict)
# convert poor_int to integer
progresa_df['progresa'] = pd.to_numeric(progresa_df['progresa'])
```


```python
# drop unnecessary variables (year, folnmum, village)
progresa_demog_df = progresa_df.drop(['year', 'village', 'folnum'], 1)
```


```python
# print summary stats table
pd.options.display.float_format = '{:8.2f}'.format # Set formating for displaying tables
progresa_demog_df.describe().T[['mean','std','min','max']].sort_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>11.37</td>
      <td>3.17</td>
      <td>6.00</td>
      <td>17.00</td>
    </tr>
    <tr>
      <th>dist_cap</th>
      <td>147.67</td>
      <td>76.06</td>
      <td>9.47</td>
      <td>359.77</td>
    </tr>
    <tr>
      <th>dist_sec</th>
      <td>2.42</td>
      <td>2.23</td>
      <td>0.00</td>
      <td>14.88</td>
    </tr>
    <tr>
      <th>fam_n</th>
      <td>7.22</td>
      <td>2.35</td>
      <td>1.00</td>
      <td>24.00</td>
    </tr>
    <tr>
      <th>grc</th>
      <td>3.96</td>
      <td>2.50</td>
      <td>0.00</td>
      <td>14.00</td>
    </tr>
    <tr>
      <th>grc97</th>
      <td>3.71</td>
      <td>2.57</td>
      <td>0.00</td>
      <td>14.00</td>
    </tr>
    <tr>
      <th>hohage</th>
      <td>44.44</td>
      <td>11.62</td>
      <td>15.00</td>
      <td>98.00</td>
    </tr>
    <tr>
      <th>hohedu</th>
      <td>2.77</td>
      <td>2.66</td>
      <td>0.00</td>
      <td>20.00</td>
    </tr>
    <tr>
      <th>hohsex</th>
      <td>0.93</td>
      <td>0.26</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>hohwag</th>
      <td>586.99</td>
      <td>788.13</td>
      <td>0.00</td>
      <td>14000.00</td>
    </tr>
    <tr>
      <th>indig</th>
      <td>0.30</td>
      <td>0.46</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>min_dist</th>
      <td>103.45</td>
      <td>42.09</td>
      <td>9.47</td>
      <td>170.46</td>
    </tr>
    <tr>
      <th>poor</th>
      <td>0.85</td>
      <td>0.36</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>progresa</th>
      <td>0.62</td>
      <td>0.49</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>sc</th>
      <td>0.82</td>
      <td>0.38</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>sc97</th>
      <td>0.81</td>
      <td>0.39</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>0.51</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>welfare_index</th>
      <td>690.35</td>
      <td>139.49</td>
      <td>180.00</td>
      <td>1294.00</td>
    </tr>
  </tbody>
</table>
</div>



### 1.2 Differences at baseline?

Are the baseline (1997) demographic characteristics **for the poor**  different in treatment and control villages? Hint: Use a T-Test to determine whether there is a statistically significant difference in the average values of each of the variables in the dataset. Focus only on the data from 1997 for individuals who are poor (i.e., poor=='pobre').

Present your results in a single table with the following columns and 14 (or so) rows:

| Variable name | Average value (Treatment villages) | Average value (Control villages) | Difference (Treat - Control) | p-value |
|------|------|------|------|------|
|Male|?|?|?|?|



```python
# your code here
# Generate data frame with only 1997 data and with only poor households
pre_poor_df = progresa_df[(progresa_df['year'] == 97) & (progresa_df['poor'] == 1)]

# create treatment data frame with poor households
pre_treat = pre_poor_df[pre_poor_df['progresa'] == 1]

# create control data frame with poor households
pre_control = pre_poor_df[pre_poor_df['progresa'] == 0]
```


```python
# drop unecessary variables for demographic characteristics
pre_treat_demog = pre_treat.drop(['poor', 'progresa', 'year', 'village', 'folnum', 'sc', 'grc'], 1)
pre_control_demog = pre_control.drop(['poor', 'progresa', 'year', 'village', 'folnum', 'sc', 'grc'], 1)
```


```python
# Create data frame with p-values
means_t = []
means_c = []
diffs = []
pvalues = []
demogvars = pre_control_demog.columns.tolist()
for var in demogvars:
    #print var
    x_t = pre_treat_demog[var].dropna().mean()
    x_c = pre_control_demog[var].dropna().mean()   
    diff = x_t - x_c
    result = stats.ttest_ind(pre_treat_demog[var].dropna(), pre_control_demog[var].dropna())
    means_t.append(x_t)
    means_c.append(x_c)
    diffs.append(diff)
    pvalues.append(result[1])

mean_compare = pd.DataFrame({'Variable name':demogvars,
                             'Average value (Treatment villages)':means_t,
                             'Average value (Control villages)':means_c,
                             'Difference (Treat - Control)':diffs,
                             'p-value':pvalues})
```


```python
pd.options.display.float_format = '{:1,.4f}'.format
mean_compare[['Variable name', 'Average value (Treatment villages)', 'Average value (Control villages)', \
          'Difference (Treat - Control)', 'p-value']].sort_values('Variable name')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variable name</th>
      <th>Average value (Treatment villages)</th>
      <th>Average value (Control villages)</th>
      <th>Difference (Treat - Control)</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>age</td>
      <td>10.7170</td>
      <td>10.7420</td>
      <td>-0.0250</td>
      <td>0.4786</td>
    </tr>
    <tr>
      <th>5</th>
      <td>dist_cap</td>
      <td>150.8291</td>
      <td>153.7697</td>
      <td>-2.9407</td>
      <td>0.0008</td>
    </tr>
    <tr>
      <th>2</th>
      <td>dist_sec</td>
      <td>2.4531</td>
      <td>2.5077</td>
      <td>-0.0545</td>
      <td>0.0357</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fam_n</td>
      <td>7.2813</td>
      <td>7.3025</td>
      <td>-0.0211</td>
      <td>0.4271</td>
    </tr>
    <tr>
      <th>12</th>
      <td>grc97</td>
      <td>3.5316</td>
      <td>3.5430</td>
      <td>-0.0115</td>
      <td>0.6890</td>
    </tr>
    <tr>
      <th>10</th>
      <td>hohage</td>
      <td>43.6488</td>
      <td>44.2769</td>
      <td>-0.6281</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>hohedu</td>
      <td>2.6631</td>
      <td>2.5903</td>
      <td>0.0728</td>
      <td>0.0111</td>
    </tr>
    <tr>
      <th>9</th>
      <td>hohsex</td>
      <td>0.9247</td>
      <td>0.9229</td>
      <td>0.0017</td>
      <td>0.5712</td>
    </tr>
    <tr>
      <th>7</th>
      <td>hohwag</td>
      <td>544.3395</td>
      <td>573.1636</td>
      <td>-28.8240</td>
      <td>0.0003</td>
    </tr>
    <tr>
      <th>1</th>
      <td>indig</td>
      <td>0.3260</td>
      <td>0.3322</td>
      <td>-0.0062</td>
      <td>0.2454</td>
    </tr>
    <tr>
      <th>4</th>
      <td>min_dist</td>
      <td>107.1529</td>
      <td>103.2379</td>
      <td>3.9151</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>sc97</td>
      <td>0.8227</td>
      <td>0.8152</td>
      <td>0.0075</td>
      <td>0.0952</td>
    </tr>
    <tr>
      <th>0</th>
      <td>sex</td>
      <td>0.5193</td>
      <td>0.5051</td>
      <td>0.0143</td>
      <td>0.0122</td>
    </tr>
    <tr>
      <th>8</th>
      <td>welfare_index</td>
      <td>655.4284</td>
      <td>659.5791</td>
      <td>-4.1507</td>
      <td>0.0014</td>
    </tr>
  </tbody>
</table>
</div>



### 1.3 Interpretation

* A: Are there statistically significant differences between treatment and control villages as baseline? 
* B: Why does it matter if there are differences at baseline?
* C: What does this imply about how to measure the impact of the treatment?

* A. Yes, treatment and control villages have several statistically significant differences. There are several variables for which we observe statistically significant differences between poor households in treatment and control village at the time of the baseline. Households in PROGRESA villages differ from control households in that they more likely to be closer to the capital, closer to a secondary school, and a greater distance from an urban center. Moreover, the head of household in PROGRESA villages are on average younger, have more education, and earn a lower wage. PROGRESA households also have lower socioeconomic status, as measured by the welfare_index and children are more likely to be male.

* B. When treatment and control villages are different at baseline, it suggests that treatment assignment was not purely random. Due to random chance, we might expect that the occasional variable might not be balanced between treatment and control, even under perfect randomization. These baseline differences are important to note. It is possible that the differences could have arisen due to random chance, but unlikely. When a large number of characteristics appear to be systematically different between treatment and control (as is the case in our sample), we should be concerned that certain types of villages were systematically assigned to treatment, while others were assigned to control.

* C. When treatment assignment is not random, the simple difference in outcomes between treatment and control units is no longer an unbiased estimate of the treatment effect. Since T and C villages were different before the intervention, we might expect they would also be different after the intervention, *even in the absence of the intervention.* Thus, we would be incorrectly attributing a treatment effect to the intervention, when in reality it was due to pre-existing differences between the treatment and control villages. Therefore, when we conduct our analysis of the impact of PROGRESA we will want to control for differences in village and household characteristics at baseline.

### 1.4 Graphical exploration, part 1

For each level of household head education, compute the average enrollment rate in 1997. Create a scatterplot that shows this relationship. What do you notice?


```python
progresa_df97=progresa_df[(progresa_df['year'] == 97)]
progresa_df97_en=pd.DataFrame(progresa_df97.groupby('hohedu')['sc'].mean()).reset_index(inplace=False)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
plt.scatter(progresa_df97_en['hohedu'],progresa_df97_en['sc'])
plt.ylabel("School Enrollment")
plt.xlabel("Household Head Education")
plt.title("Average Grade Enrolled vs Household Head Education")
plt.show()
```


![png](output_18_0.png)


There is a positive linear relationship between school enrollment in 1997 and the education of the head of household. In other words, in households where the household head has more education, it is more likely that the child will be enrolled in school.  This is a common correlation found in developing and developed countries alike.

### 1.5 Graphical exploration, part 2

Visualize the distribution of village enrollment rates **among poor households in treated villages**, before and after treatment. Specifically, for each village, calculate the average rate of enrollment of poor households in treated villages in 1997, then compute the average rate of enrollment of poor households in treated villages in 1998. Create two separate histograms showing the distribution of these average enrollments rates, one histogram for 1997 and one histogram for 1998. On each histogram, draw a vertical line that intersects the x-axis at the average value (across all households). Does there appear to be a difference? Is this difference statistically significant?


```python
# Your code here
#print progresa_df.shape
poor_treatment97 = progresa_df[(progresa_df['year'] == 97) &
                               (progresa_df['progresa'] == 1) &
                               (progresa_df['poor'] == 1)]
poor_treatment98 = progresa_df[(progresa_df['year'] == 98) &
                               (progresa_df['progresa'] == 1) &
                               (progresa_df['poor'] == 1)]

poor_treatment97_en=pd.DataFrame(poor_treatment97.groupby('village')['sc'].mean())
poor_treatment98_en=pd.DataFrame(poor_treatment98.groupby('village')['sc'].mean())
# Histogram for Pre (1997)
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
plt.ylim(ymax=45)
plt.hist(poor_treatment97_en.values, color="green", bins=25)
plt.title("Village Enrollment Rates Among Poor Households, Before Treatment")
plt.xlabel("Enrollment Rate")

pre_mean = poor_treatment97_en['sc'].mean()
plt.axvline(pre_mean, color="black", linewidth=2)
plt.axis((0.3,1.1,0,40))
plt.show()

# Histogram for Post (1998)
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
plt.ylim(ymax=45)
plt.hist(poor_treatment98_en.values, color="Blue", bins=25)
plt.title("Village Enrollment Rates Among Poor Households, After Treatment")
plt.xlabel("Enrollment Rate")
post_mean = poor_treatment98_en['sc'].mean()
plt.axvline(post_mean, color="black", linewidth=2)
plt.axis((0.3,1.1,0,40))
plt.show()
```


![png](output_21_0.png)



![png](output_21_1.png)



```python
t, p = stats.ttest_ind(poor_treatment97_en, poor_treatment98_en)
print('Difference:', (post_mean - pre_mean))
print('p-value:', p)
```

    Difference: 0.0156967932034231
    p-value: [0.0442487]
    

On average, enrollment rates in treated villages increased by 1.5% between 1997 and 1998. This difference in means was statistically significant at standard levels of stastical significance (p<0.05). Note that this doesn't necessary mean that the treatment had an impact, it just says that average enrollment rates in treated villages were higher in the period post-treatment than they were in the period pre-treatment.

## Part 2: Measuring Impact

Our goal is to estimate the causal impact of the PROGRESA program on the social and economic outcomes of individuals in Mexico. We will focus on the impact of the program on the poor (those with poor=='pobre'), since only the poor were eligible to receive the PROGRESA assistance.

### 2.1 Simple differences: T-test

Begin by estimating the impact of Progresa using "simple differences." Restricting yourself to data from 1998 (after treatment), calculate the average enrollment rate among **poor** households in the Treatment villages and the average enrollment rate among **poor** households in the control villages. Use a t-test to determine if this difference is statistically significant. What do you conclude?


```python
df98 = progresa_df[progresa_df['year'] == 98]
df97 = progresa_df[progresa_df['year'] == 97]

treated98 = df98[(df98.progresa == 1) & (df98.poor == 1)]
control98 = df98[(df98.progresa == 0) & (df98.poor == 1)]
mean_t98 = treated98.sc.mean()
mean_c98 = control98.sc.mean()

print('Control average (98): {}'.format(mean_c98))
print('Treated average (98): {}'.format(mean_t98))
print('Difference: {}'.format(mean_t98-mean_c98))
tstat, pval = stats.ttest_ind(treated98.sc.dropna(), control98.sc.dropna())
print('P-value: {}'.format(pval))
```

    Control average (98): 0.807636956730308
    Treated average (98): 0.8464791213954308
    Difference: 0.0388421646651228
    P-value: 6.636344447523235e-17
    

Enrollment rates among the poor in 1998 in the treated villages were nearly four percentage points higher than average enrollment rates in the control villages. These differences are statistically significant, so the "simple differences" method makes it look like Progresa  had an effect on enrollment. However, we are skeptical of these estimates, since we know that treatment villages were systematically different from control villages, even before the PROGRESA program arrived.

### 2.2 Simple differences: Regression

Estimate the effects of Progresa on enrollment using a regression model, by regressing the 1998 enrollment rates **of the poor** on treatment assignment. For now, do not include any other variables in your regression. Discuss the following:

* Based on this model, how much did Progresa increase or decrease the likelihood of a child enrolling? Make sure you express you answer in a sentence that your grandmother could understand, using appropriate units.
* How does your regression estimate compare to your t-test estimate from part 2.1?
* Based on this regression model, can we reject the null hypothesis that the treatment effects are zero? 
* What is the counterfactual assumption underlying this regression?


```python
poor98 = df98[df98.poor == 1]
ols1 = smf.ols(formula = "sc ~ progresa", data=poor98, missing='drop').fit()
print(ols1.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                     sc   R-squared:                       0.003
    Model:                            OLS   Adj. R-squared:                  0.003
    Method:                 Least Squares   F-statistic:                     69.87
    Date:                Mon, 10 Feb 2020   Prob (F-statistic):           6.64e-17
    Time:                        17:08:52   Log-Likelihood:                -11926.
    No. Observations:               27450   AIC:                         2.386e+04
    Df Residuals:                   27448   BIC:                         2.387e+04
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      0.8076      0.004    220.676      0.000       0.800       0.815
    progresa       0.0388      0.005      8.359      0.000       0.030       0.048
    ==============================================================================
    Omnibus:                     7638.939   Durbin-Watson:                   1.734
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):            15767.534
    Skew:                          -1.767   Prob(JB):                         0.00
    Kurtosis:                       4.140   Cond. No.                         3.01
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

* The regression indicates that progresa increased enrollment of the poor by 3.88 percentage points
* Our estimate of the "simple differences" treatment effect of Progresa is identical to the estimate from 2.1
* Based on this regression model, we can reject the null hypothesis. The estimate of 3.88 percentage points is highly significant (p<0.01).
* The counterfactual assumption is that, in the absence of treatment, there would have been no statistically significant differences in schooling enrollment between the treatment and control group

### 2.3 Multiple Regression

Re-run the above regression estimated but this time include a set of control variables. Include, for instance, age, distance to a secondary school, gender, education of household head, welfare index, indigenous, etc.

* How do the controls affect the point estimate of treatment effect?
* How do the controls affect the standard error on the treatment effect? 
* How do you interpret the differences (or similarities) between your estimates of 2.2 and 2.3?


```python
formula2 = "sc ~ progresa + age + dist_sec + C(sex) + hohedu + fam_n"
ols2 = smf.ols(formula2, data=poor98, missing='drop').fit()
print(ols2.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                     sc   R-squared:                       0.264
    Model:                            OLS   Adj. R-squared:                  0.264
    Method:                 Least Squares   F-statistic:                     1644.
    Date:                Mon, 10 Feb 2020   Prob (F-statistic):               0.00
    Time:                        17:08:53   Log-Likelihood:                -7742.1
    No. Observations:               27440   AIC:                         1.550e+04
    Df Residuals:                   27433   BIC:                         1.556e+04
    Df Model:                           6                                         
    Covariance Type:            nonrobust                                         
    =================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------
    Intercept         1.5399      0.011    137.651      0.000       1.518       1.562
    C(sex)[T.1.0]     0.0311      0.004      8.013      0.000       0.023       0.039
    progresa          0.0352      0.004      8.821      0.000       0.027       0.043
    age              -0.0654      0.001    -95.546      0.000      -0.067      -0.064
    dist_sec         -0.0112      0.001    -12.892      0.000      -0.013      -0.010
    hohedu            0.0087      0.001     11.129      0.000       0.007       0.010
    fam_n            -0.0006      0.001     -0.654      0.513      -0.002       0.001
    ==============================================================================
    Omnibus:                     3016.118   Durbin-Watson:                   1.710
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             4117.802
    Skew:                          -0.941   Prob(JB):                         0.00
    Kurtosis:                       3.243   Cond. No.                         82.8
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

* Including control variables deceases the estimated coefficient on the treatment effect
* There is very little effect on the standard error
* Controlling for other observable factors decreased the estimated treatment effect - or put differently, without controlling for other factors, the estimates of the treatment effectin 2.2 were likely *over*-estimates of the true treatment effect. This is likely due to the fact that other factors were correlated with both treatment and outcomes (as we saw in 1.2).

### 2.4 Difference-in-Difference, version 1 (tabular)

Thus far, we have computed the effects of Progresa by estimating the difference in 1998 enrollment rates across villages. An alternative approach would be to compute the treatment effect using a difference-in-differences framework.

Begin by estimating the average treatment effects of the program for poor households using data from 1997 and 1998. Specifically, calculate the difference (between 1997 and 1998) in enrollment rates among poor households in treated villages; then compute the difference (between 1997 and 1998) in enrollment rates among poor households in control villages. The difference between these two differences is your estimate.

* What is your estimate of the impact, and how does it compare to your earlier (simple difference) results?
* What is the counterfactual assumption underlying this estimate? 



```python
poor_df = progresa_df[progresa_df['poor'] == 1]
poor_df_mean_sc = poor_df.groupby(['year', 'progresa'])['sc'].mean()
print(poor_df_mean_sc)
print(poor_df)
dd_estimate = (poor_df_mean_sc[3] - poor_df_mean_sc[2]) - (poor_df_mean_sc[1] - poor_df_mean_sc[0])
print(dd_estimate)
```

    year  progresa
    97    0          0.8152
          1          0.8227
    98    0          0.8076
          1          0.8465
    Name: sc, dtype: float64
           year    sex  indig  dist_sec     sc    grc  fam_n  min_dist  dist_cap  \
    0        97 0.0000 0.0000    4.4730 1.0000 7.0000      7   21.1684   21.1684   
    1        98 0.0000 0.0000    4.4730 1.0000 8.0000      7   21.1684   21.1684   
    2        97 1.0000 0.0000    4.4730 1.0000 6.0000      7   21.1684   21.1684   
    3        98 1.0000 0.0000    4.4730 1.0000 7.0000      7   21.1684   21.1684   
    4        97 0.0000 0.0000    4.4730 1.0000 2.0000      7   21.1684   21.1684   
    5        98 0.0000 0.0000    4.4730 1.0000 3.0000      7   21.1684   21.1684   
    6        97 0.0000 0.0000    3.1540 0.0000 6.0000      6  127.1148  154.1960   
    7        98 0.0000 0.0000    3.1540 0.0000 6.0000      6  127.1148  154.1960   
    8        97 1.0000 0.0000    3.3730 1.0000 2.0000      5   85.3003  105.8787   
    9        98 1.0000 0.0000    3.3730 1.0000 2.0000      5   85.3003  105.8787   
    10       97 0.0000 0.0000    3.3730 1.0000 2.0000      5   85.3003  105.8787   
    11       98 0.0000 0.0000    3.3730 1.0000 3.0000      5   85.3003  105.8787   
    12       97 0.0000 0.0000    3.3730    nan 0.0000      5   85.3003  105.8787   
    13       98 0.0000 0.0000    3.3730 1.0000 1.0000      5   85.3003  105.8787   
    14       97 1.0000 1.0000    1.9350 1.0000 2.0000      5  127.6576  333.0487   
    15       98 1.0000 1.0000    1.9350 1.0000 3.0000      5  127.6576  333.0487   
    16       97 0.0000 1.0000    1.9350 1.0000 2.0000      5  127.6576  333.0487   
    17       98 0.0000 1.0000    1.9350 1.0000 3.0000      5  127.6576  333.0487   
    18       97 1.0000 1.0000    1.9350 1.0000 5.0000     10  127.6576  333.0487   
    19       98 1.0000 1.0000    1.9350 0.0000 6.0000     10  127.6576  333.0487   
    20       97 0.0000 1.0000    1.9350 1.0000 3.0000     10  127.6576  333.0487   
    21       98 0.0000 1.0000    1.9350 1.0000 6.0000     10  127.6576  333.0487   
    22       97 0.0000 1.0000    1.9350 1.0000 2.0000      7  127.6576  333.0487   
    23       98 0.0000 1.0000    1.9350 1.0000 3.0000      7  127.6576  333.0487   
    24       97 1.0000 1.0000    1.9350 1.0000 1.0000      7  127.6576  333.0487   
    25       98 1.0000 1.0000    1.9350 1.0000 2.0000      7  127.6576  333.0487   
    26       97 0.0000 1.0000    1.9350 1.0000 3.0000      5  127.6576  333.0487   
    27       98 0.0000 1.0000    1.9350 1.0000 3.0000      5  127.6576  333.0487   
    28       97 1.0000 1.0000    1.9350 0.0000 6.0000      9  127.6576  333.0487   
    29       98 1.0000 1.0000    1.9350    nan    nan      9  127.6576  333.0487   
    ...     ...    ...    ...       ...    ...    ...    ...       ...       ...   
    77220    97 0.0000 1.0000    3.1480 0.0000 4.0000     13  137.4732  172.7708   
    77221    98 0.0000 1.0000    3.1480    nan    nan     13  137.4732  172.7708   
    77222    97 0.0000 1.0000    3.1480 0.0000 4.0000     13  137.4732  172.7708   
    77223    98 0.0000 1.0000    3.1480    nan    nan     13  137.4732  172.7708   
    77224    97 0.0000 1.0000    3.1480 1.0000 4.0000     13  137.4732  172.7708   
    77225    98 0.0000 1.0000    3.1480 1.0000 1.0000     13  137.4732  172.7708   
    77226    97 1.0000 1.0000    3.1480 1.0000 3.0000     13  137.4732  172.7708   
    77227    98 1.0000 1.0000    3.1480 1.0000 3.0000     13  137.4732  172.7708   
    77228    97 0.0000 1.0000    3.1480    nan 0.0000      3  137.4732  172.7708   
    77229    98 0.0000 1.0000    3.1480 0.0000 3.0000      3  137.4732  172.7708   
    77230    97 0.0000 1.0000    3.1480 1.0000 4.0000     16  137.4732  172.7708   
    77231    98 0.0000 1.0000    3.1480 0.0000 4.0000     16  137.4732  172.7708   
    77232    97 0.0000 1.0000    3.1480 1.0000 2.0000     16  137.4732  172.7708   
    77233    98 0.0000 1.0000    3.1480 1.0000 2.0000     16  137.4732  172.7708   
    77234    97 1.0000 1.0000    3.1480 1.0000 2.0000      8  137.4732  172.7708   
    77235    98 1.0000 1.0000    3.1480 1.0000 2.0000      8  137.4732  172.7708   
    77236    97 0.0000 1.0000    3.1480 1.0000 1.0000      8  137.4732  172.7708   
    77237    98 0.0000 1.0000    3.1480 1.0000 2.0000      8  137.4732  172.7708   
    77238    97 1.0000 1.0000    3.1480 1.0000 1.0000      8  137.4732  172.7708   
    77239    98 1.0000 1.0000    3.1480 1.0000 2.0000      8  137.4732  172.7708   
    77240    97 0.0000 1.0000    3.1480 1.0000 2.0000      8  137.4732  172.7708   
    77241    98 0.0000 1.0000    3.1480 1.0000 3.0000      8  137.4732  172.7708   
    77242    97 1.0000 1.0000    3.1480 1.0000 1.0000      9  137.4732  172.7708   
    77243    98 1.0000 1.0000    3.1480 1.0000 2.0000      9  137.4732  172.7708   
    77244    97 0.0000 1.0000    3.1480 1.0000 2.0000      6  137.4732  172.7708   
    77245    98 0.0000 1.0000    3.1480 1.0000 4.0000      6  137.4732  172.7708   
    77246    97 1.0000 1.0000    3.1480 1.0000 1.0000      6  137.4732  172.7708   
    77247    98 1.0000 1.0000    3.1480 1.0000 2.0000      6  137.4732  172.7708   
    77248    97 0.0000 1.0000    3.1480 0.0000 2.0000      3  137.4732  172.7708   
    77249    98 0.0000 1.0000    3.1480 0.0000 3.0000      3  137.4732  172.7708   
    
           poor  ...    hohedu   hohwag  welfare_index  hohsex  hohage  age  \
    0         1  ...         6   0.0000       583.0000  1.0000 35.0000   13   
    1         1  ...         6   0.0000       583.0000  1.0000 35.0000   14   
    2         1  ...         6   0.0000       583.0000  1.0000 35.0000   12   
    3         1  ...         6   0.0000       583.0000  1.0000 35.0000   13   
    4         1  ...         6   0.0000       583.0000  1.0000 35.0000    8   
    5         1  ...         6   0.0000       583.0000  1.0000 35.0000    9   
    6         1  ...         4   0.0000       684.0000  1.0000 85.0000   14   
    7         1  ...         4   0.0000       684.0000  1.0000 85.0000   15   
    8         1  ...         6 875.0000       742.1400  1.0000 26.0000    9   
    9         1  ...         6 875.0000       742.1400  1.0000 26.0000   10   
    10        1  ...         6 875.0000       742.1400  1.0000 26.0000    7   
    11        1  ...         6 875.0000       742.1400  1.0000 26.0000    8   
    12        1  ...         6 875.0000       742.1400  1.0000 26.0000    6   
    13        1  ...         6 875.0000       742.1400  1.0000 26.0000    7   
    14        1  ...         3 500.0000       552.0000  1.0000 98.0000   10   
    15        1  ...         3 500.0000       552.0000  1.0000 98.0000   11   
    16        1  ...         3 500.0000       552.0000  1.0000 98.0000    8   
    17        1  ...         3 500.0000       552.0000  1.0000 98.0000    9   
    18        1  ...         0 500.0000       660.0000  1.0000 60.0000   16   
    19        1  ...         0 500.0000       660.0000  1.0000 60.0000   17   
    20        1  ...         0 500.0000       660.0000  1.0000 60.0000   12   
    21        1  ...         0 500.0000       660.0000  1.0000 60.0000   13   
    22        1  ...         4 500.0000       471.0000  1.0000 30.0000    8   
    23        1  ...         4 500.0000       471.0000  1.0000 30.0000    9   
    24        1  ...         4 500.0000       471.0000  1.0000 30.0000    7   
    25        1  ...         4 500.0000       471.0000  1.0000 30.0000    8   
    26        1  ...         4 500.0000       530.0000  1.0000 32.0000    8   
    27        1  ...         4 500.0000       530.0000  1.0000 32.0000    9   
    28        1  ...         0 500.0000       595.6000  1.0000 50.0000   14   
    29        1  ...         0 500.0000       595.6000  1.0000 50.0000   15   
    ...     ...  ...       ...      ...            ...     ...     ...  ...   
    77220     1  ...         0 500.0000       528.3800  1.0000 41.0000   15   
    77221     1  ...         0 500.0000       528.3800  1.0000 41.0000   16   
    77222     1  ...         0 500.0000       528.3800  1.0000 41.0000   13   
    77223     1  ...         0 500.0000       528.3800  1.0000 41.0000   14   
    77224     1  ...         0 500.0000       528.3800  1.0000 41.0000   10   
    77225     1  ...         0 500.0000       528.3800  1.0000 41.0000   11   
    77226     1  ...         0 500.0000       528.3800  1.0000 41.0000    9   
    77227     1  ...         0 500.0000       528.3800  1.0000 41.0000   10   
    77228     1  ...         2 500.0000       635.3300  1.0000 50.0000   12   
    77229     1  ...         2 500.0000       635.3300  1.0000 50.0000   13   
    77230     1  ...         2   0.0000       525.3300  1.0000 61.0000   15   
    77231     1  ...         2   0.0000       525.3300  1.0000 61.0000   16   
    77232     1  ...         2   0.0000       525.3300  1.0000 61.0000    8   
    77233     1  ...         2   0.0000       525.3300  1.0000 61.0000    9   
    77234     1  ...         2 500.0000       418.0000  1.0000 32.0000   10   
    77235     1  ...         2 500.0000       418.0000  1.0000 32.0000   11   
    77236     1  ...         2 500.0000       418.0000  1.0000 32.0000    8   
    77237     1  ...         2 500.0000       418.0000  1.0000 32.0000    9   
    77238     1  ...         2 500.0000       418.0000  1.0000 32.0000    6   
    77239     1  ...         2 500.0000       418.0000  1.0000 32.0000    7   
    77240     1  ...         0 500.0000       580.0000  1.0000 56.0000   12   
    77241     1  ...         0 500.0000       580.0000  1.0000 56.0000   13   
    77242     1  ...         1 500.0000       582.5000  1.0000 45.0000    8   
    77243     1  ...         1 500.0000       582.5000  1.0000 45.0000    9   
    77244     1  ...         0   0.0000       599.0000  0.0000 67.0000   11   
    77245     1  ...         0   0.0000       599.0000  0.0000 67.0000   12   
    77246     1  ...         0   0.0000       599.0000  0.0000 67.0000    7   
    77247     1  ...         0   0.0000       599.0000  0.0000 67.0000    8   
    77248     1  ...         0 375.0000       634.0000  1.0000 38.0000   14   
    77249     1  ...         0 375.0000       634.0000  1.0000 38.0000   15   
    
           village  folnum  grc97   sc97  
    0          163       1      7 1.0000  
    1          163       1      7 1.0000  
    2          163       2      6 1.0000  
    3          163       2      6 1.0000  
    4          163       3      2 1.0000  
    5          163       3      2 1.0000  
    6          271       4      6 0.0000  
    7          271       4      6 0.0000  
    8          263       5      2 1.0000  
    9          263       5      2 1.0000  
    10         263       6      2 1.0000  
    11         263       6      2 1.0000  
    12         263       7      0    nan  
    13         263       7      0    nan  
    14         418       8      2 1.0000  
    15         418       8      2 1.0000  
    16         418       9      2 1.0000  
    17         418       9      2 1.0000  
    18         418      10      5 1.0000  
    19         418      10      5 1.0000  
    20         418      11      3 1.0000  
    21         418      11      3 1.0000  
    22         418      12      2 1.0000  
    23         418      12      2 1.0000  
    24         418      13      1 1.0000  
    25         418      13      1 1.0000  
    26         418      14      3 1.0000  
    27         418      14      3 1.0000  
    28         418      15      6 0.0000  
    29         418      15      6 0.0000  
    ...        ...     ...    ...    ...  
    77220      348   38611      4 0.0000  
    77221      348   38611      4 0.0000  
    77222      348   38612      4 0.0000  
    77223      348   38612      4 0.0000  
    77224      348   38613      4 1.0000  
    77225      348   38613      4 1.0000  
    77226      348   38614      3 1.0000  
    77227      348   38614      3 1.0000  
    77228      348   38615      0    nan  
    77229      348   38615      0    nan  
    77230      348   38616      4 1.0000  
    77231      348   38616      4 1.0000  
    77232      348   38617      2 1.0000  
    77233      348   38617      2 1.0000  
    77234      348   38618      2 1.0000  
    77235      348   38618      2 1.0000  
    77236      348   38619      1 1.0000  
    77237      348   38619      1 1.0000  
    77238      348   38620      1 1.0000  
    77239      348   38620      1 1.0000  
    77240      348   38621      2 1.0000  
    77241      348   38621      2 1.0000  
    77242      348   38622      1 1.0000  
    77243      348   38622      1 1.0000  
    77244      348   38623      2 1.0000  
    77245      348   38623      2 1.0000  
    77246      348   38624      1 1.0000  
    77247      348   38624      1 1.0000  
    77248      348   38625      2 0.0000  
    77249      348   38625      2 0.0000  
    
    [65392 rows x 21 columns]
    0.031331280319323085
    

* Our double-different estimate is that PROGRESA increased enrolment by 3.13 percentage points. This estimate of the impact of PROGRESA is lower than the simple differences estimate from 2.3, and also lower than the simple difference estimate that controls for potentially confounding factors (2.4). Again, this all points to the fact that treated households were systematically different from control households, even in the absence of treatment. Most noticably, we previously estimated baseline differences in enrollment of 0.75%; of the various research designs we have tested, only the difference-in-difference estimator accounts for this.
* The counterfactual assumption is that in the absence of treatment, the difference in enrollment for poor households in the treatment group between 1997 and 1998 would be the same as the difference in enrollment for poor households in the control group between 1997 and 1998.  Another way of saying this is that we are assuming that there exist parallel trends over time in enrollment rates between treated and control villages.

### 2.5 Difference-in-Difference, version 2 (regression)

Now use a regression specification to estimate the average treatment effects of the program in a difference-in-differences framework. Include at least 5 control variables.

* What is your estimate of the impact of Progresa? Be very specific in interpreting your coefficients and standard errors, and make sure to specify exactly what units you are measuring and estimating.
* How do these estimates of the treatment effect compare to the estimates based on the simple difference?
* How do these estimates compare to the difference-in-difference estimates from 2.4 above? What accounts for these differences?
* What is the counterfactual assumption underlying this regression? 


```python
poor_df = progresa_df[progresa_df.poor == 1].copy()
poor_df.year = np.where(poor_df.year == 97,0,1)
ols3 = smf.ols(formula='sc ~ age + C(sex) + fam_n + hohedu + progresa*year', data=poor_df, missing='drop').fit()
# note that in the above command [progresa*year] is a shorthand for [progresa + year + progresa X year] 
print(ols3.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                     sc   R-squared:                       0.271
    Model:                            OLS   Adj. R-squared:                  0.271
    Method:                 Least Squares   F-statistic:                     3106.
    Date:                Mon, 10 Feb 2020   Prob (F-statistic):               0.00
    Time:                        17:08:58   Log-Likelihood:                -17031.
    No. Observations:               58352   AIC:                         3.408e+04
    Df Residuals:                   58344   BIC:                         3.415e+04
    Df Model:                           7                                         
    Covariance Type:            nonrobust                                         
    =================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------
    Intercept         1.4853      0.008    197.708      0.000       1.471       1.500
    C(sex)[T.1.0]     0.0341      0.003     12.692      0.000       0.029       0.039
    age              -0.0651      0.000   -143.278      0.000      -0.066      -0.064
    fam_n            -0.0011      0.001     -1.887      0.059      -0.002    4.26e-05
    hohedu            0.0085      0.001     15.863      0.000       0.007       0.010
    progresa          0.0037      0.004      0.977      0.329      -0.004       0.011
    year              0.0264      0.004      6.054      0.000       0.018       0.035
    progresa:year     0.0318      0.006      5.740      0.000       0.021       0.043
    ==============================================================================
    Omnibus:                     5673.772   Durbin-Watson:                   1.470
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             7512.805
    Skew:                          -0.877   Prob(JB):                         0.00
    Kurtosis:                       3.121   Cond. No.                         85.7
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

* Using a difference-in-difference regression, I estimate that the Progresa program increased enrollment of poor households by 3.18 percentage points (this is Progresa*year). Note that we further estimate that enrollment rates, in general, were 2.64 percentage points higher in 1998 (the coefficient on year) and that Progresa villages have on average 0.37 percentage points higher enrollment (the coefficient on Progresa). 
* Compared to the single differences, the estimate of the treatment effect is smaller.
* The estimated treatment effect is quite similar to that estimated in 2.4, but not identical. The differences result from the fact that in this regression we are also controlling for other observed characteristics of the household (such as age, hohedu, etc.) which are also correlated with enrollment. 
* The counterfactual assumption is that, after accounting for the factors that we control for in the regression, in the absence of treatment, the difference in enrollment in the treatment group between 1997 and 1998 would be the same as the difference for the control group.  Another way of saying this is that we are assuming that there exist parallel trends over time in enrollment rates between treated and control villages, conditional on the other variables in the regression

### 2.6 Spillover effects

Thus far, we have focused on the impact of PROGRESA on poor households. Repeat your analysis in 2.5, instead focusing on the impact of PROGRESA on non-poor households. 
* Do you observe any impacts of PROGRESA on the non-poor?
* Regardless of whether you find evidence of spillovers, describe one or two reasons why PROGRESA *might* have impacted non-poor households. Give concrete examples based on the context in which PROGRESA was implemented.


```python
nonpoor_df = progresa_df[progresa_df.poor == 0].copy()
nonpoor_df.year = np.where(nonpoor_df.year == 97, 0, 1)
ols4 = smf.ols(formula='sc ~ age + C(sex) + fam_n + hohedu + progresa*year',data=nonpoor_df).fit()
print(ols4.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                     sc   R-squared:                       0.278
    Model:                            OLS   Adj. R-squared:                  0.277
    Method:                 Least Squares   F-statistic:                     572.5
    Date:                Mon, 10 Feb 2020   Prob (F-statistic):               0.00
    Time:                        17:08:58   Log-Likelihood:                -3761.2
    No. Observations:               10423   AIC:                             7538.
    Df Residuals:                   10415   BIC:                             7596.
    Df Model:                           7                                         
    Covariance Type:            nonrobust                                         
    =================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------
    Intercept         1.5500      0.020     79.240      0.000       1.512       1.588
    C(sex)[T.1.0]     0.0292      0.007      4.295      0.000       0.016       0.043
    age              -0.0683      0.001    -60.008      0.000      -0.071      -0.066
    fam_n            -0.0059      0.001     -4.083      0.000      -0.009      -0.003
    hohedu            0.0103      0.001      9.576      0.000       0.008       0.012
    progresa          0.0292      0.009      3.129      0.002       0.011       0.048
    year              0.0374      0.011      3.450      0.001       0.016       0.059
    progresa:year    -0.0004      0.014     -0.026      0.979      -0.028       0.027
    ==============================================================================
    Omnibus:                      877.870   Durbin-Watson:                   1.478
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              851.992
    Skew:                          -0.642   Prob(JB):                    9.82e-186
    Kurtosis:                       2.441   Cond. No.                         89.4
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

* Using this simple difference-in-difference strategy, we do not observe any statistically different impacts of PROGRESA on the non-poor.
* PROGRESA could have benefited non-poor households for several reasons. For instance, if the PROGRESA program improved overall school quality (for all students) in treated villages, this would benefit poor and non-poor alike. Alternatively, if non-poor households were somehow able to cheat the assignment rule and receive the subsidy, they would benefit from imperfect treatment compliance.

### 2.7 Summary

* Based on all the analysis you have undertaken to date, do you believe that Progresa had a causal impact on the enrollment rates of poor households in Mexico? 
* Describe one other way that you might analyze these data to further investigate the causal impact of Progresa on enrollment, and clearly state the counterfactual assumption you would need to make for that approach to be valid.  *(Hint: Consider using the non-poor in your analysis)*

* Using several different empirical frameworks, we have explored the impact of the PROGRESA program on enrollment rates of poor households in Mexico. In general, it appears that PROGRESA did have positive impact on the school enrollment rates, though the exact estimate depends on the method used to measure impact. After controlling for other observable characteristics, the most reliable difference-in-difference estimator estimates that PROGRESA raised enrollment rates by roughly 3.18 percentage points. A separate estimator, that includes externalities on non-poor households, produces a more conservative, and statistically insignificant, estimate of 0.59 percentage points.

* Compare enrollment rates in 1998 between poor and non-poor across treatment and control villages


```python
df_98 = progresa_df[progresa_df.year == 98].copy()
ols5 = smf.ols(formula='sc ~ age + C(sex) + fam_n + hohedu + progresa*poor',data=df_98, missing='drop').fit()
print(ols5.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                     sc   R-squared:                       0.261
    Model:                            OLS   Adj. R-squared:                  0.261
    Method:                 Least Squares   F-statistic:                     1620.
    Date:                Sun, 03 Sep 2017   Prob (F-statistic):               0.00
    Time:                        21:20:15   Log-Likelihood:                -9514.2
    No. Observations:               32097   AIC:                         1.904e+04
    Df Residuals:                   32089   BIC:                         1.911e+04
    Df Model:                           7                                         
    Covariance Type:            nonrobust                                         
    =================================================================================
                        coef    std err          t      P>|t|      [95.0% Conf. Int.]
    ---------------------------------------------------------------------------------
    Intercept         1.5256      0.013    120.334      0.000         1.501     1.550
    C(sex)[T.1.0]     0.0312      0.004      8.573      0.000         0.024     0.038
    age              -0.0656      0.001   -102.316      0.000        -0.067    -0.064
    fam_n            -0.0019      0.001     -2.356      0.018        -0.003    -0.000
    hohedu            0.0103      0.001     14.801      0.000         0.009     0.012
    progresa          0.0295      0.010      3.025      0.002         0.010     0.049
    poor             -0.0059      0.008     -0.719      0.472        -0.022     0.010
    progresa:poor     0.0059      0.011      0.556      0.578        -0.015     0.027
    ==============================================================================
    Omnibus:                     3341.864   Durbin-Watson:                   1.705
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             4507.440
    Skew:                          -0.915   Prob(JB):                         0.00
    Kurtosis:                       3.139   Cond. No.                         137.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

In this estimate, we again use interactions, but this time we interact treatment status of the village with being poor, and therefore eligible for the program. In this estimate, the counterfactual assumption we are making is that the difference between poor and non-poor enrollment in treatment villages in the absence of treatment would have matched the observed difference between poor and non-poor enrollment in control villages. 
In this regression, we estimate a positive treatment effect. But it is insignificant.
