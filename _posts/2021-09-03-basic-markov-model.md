---
layout: post
title: A Markov Model for sleep stages using Oura Ring data
categories: [Sleep, Markov]
---

The [Oura ring](https://ouraring.com/) is a device worn on the finger that uses infrared measurements to detect sleep states, as well as other useful biometric data throughout the night (Resting heart rate, heart rate variability, respiratory rate and skin tempurature).

The data proves to be pretty useful for examining long-term trends and factors that influence sleep and overall health ([HRV example](https://ilmostromberg.com/1000-days-with-oura-ring/), [Sleep example](https://www.quantifiedbob.com/sleep-tracking-analysis-oura/)). 

Oura provides an developer API that can be used to download raw data. This is great because the summary data provided in app, while useful, discards some valauble time-based information that can be used to better identify trends and moderating factors of better sleep. 

In this post, I set up a basic markov model in STAN, using a [principled Bayesian workflow](https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html). We'll set up the basic model structure with some sensible priors, and then examine how our prior choices will impact inference. Then, we'll fit the model to some actual data and discuss how to improve it in a way that gives us better insight on the sleep data.

Below is an example of the data returned (after a bit of pre-processing). Each row corresponds to a 5 minute period of sleep. `hr_5min` refers to the average heart rate over each measurement period, `rmssd_5min` is the average HRV (using the root mean square of standard deviations measurement) and `sleep_state_ordered` refers to the sleep state within that measurement period, where `1` means awake, `2` means light sleep, `3` means REM sleep and `4` means deep sleep. This is a different ordering to how oura returns the sleep data - it won't impact this analysis but later on we might want to use an ordinal structure for our models. 



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>summary_date</th>
      <th>duration</th>
      <th>measurement_no</th>
      <th>hr_5min</th>
      <th>rmssd_5min</th>
      <th>sleep_state_ordered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-05-15</td>
      <td>34440</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-05-15</td>
      <td>34440</td>
      <td>2</td>
      <td>61</td>
      <td>61</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-05-15</td>
      <td>34440</td>
      <td>3</td>
      <td>58</td>
      <td>64</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-05-15</td>
      <td>34440</td>
      <td>4</td>
      <td>57</td>
      <td>62</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-05-15</td>
      <td>34440</td>
      <td>5</td>
      <td>55</td>
      <td>60</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>47836</th>
      <td>2021-09-02</td>
      <td>28140</td>
      <td>90</td>
      <td>62</td>
      <td>30</td>
      <td>3</td>
    </tr>
    <tr>
      <th>47837</th>
      <td>2021-09-02</td>
      <td>28140</td>
      <td>91</td>
      <td>63</td>
      <td>43</td>
      <td>1</td>
    </tr>
    <tr>
      <th>47838</th>
      <td>2021-09-02</td>
      <td>28140</td>
      <td>92</td>
      <td>57</td>
      <td>55</td>
      <td>2</td>
    </tr>
    <tr>
      <th>47839</th>
      <td>2021-09-02</td>
      <td>28140</td>
      <td>93</td>
      <td>55</td>
      <td>50</td>
      <td>2</td>
    </tr>
    <tr>
      <th>47840</th>
      <td>2021-09-02</td>
      <td>28140</td>
      <td>94</td>
      <td>54</td>
      <td>75</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>47841 rows × 6 columns</p>
</div>



The basic Markov Model suggests that the next sleep state observed will only be influenced by the current state (aka the memoryless property) $$Pr(Y_t = y | Y_{t-1}).$$ This might be an oversimplification initially, but we will expand on the model once we understand the sensitivity of the model to certain prior choices. 

To convert this into a statistical model, we must specify a likelihood function and some sensible prior choices. 
$$ y_t \sim Categorical(\theta_{y_{t-1}})$$
$$ \theta_k \sim Dirichlet(\alpha), k = 1, 2, 3, 4$$
$$ \alpha \sim Uniform(0, \infty)$$
The next observed state is a function of the transition matrix $\theta$ for the current state. Each row of the transition matrix $\theta$ is drawn from a Dirichlet distribution with hyperprior $\alpha$. We don't specifically set a prior for $\alpha$ yet - the default is for STAN to use a uniform prior over the allowed range of the parameter (the positive reals in this case). 
To test the implications of our selected priors, we can sample from the model without any data.


```python

model0 = """
data {
    int<lower=1> K; // number of states
    int<lower=1> N; // number of observations
    vector[N] y; // observations
}
parameters {
    vector<lower=0>[K] alpha;
    simplex[K] theta[K]; // transition parameters
}
model {
  for (k in 1:K)
    theta[k] ~ dirichlet(alpha);
}
"""

posterior = stan.build(model0
                       , {"K": 4
                          , "N": subsetFrame.shape[0]
                          , "y": subsetFrame['sleep_state_ordered'].to_list()
                         })

fit = posterior.sample(num_chains=4, num_samples=2000)
```

The parameter summaries are below. As expected with a uniform prior, the range of alpha is quite large. In turn, that allows each probability parameter to sample across the full range of its support $[0, 1]$. 


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/2f/s3znbbc13nn29v0hcm82ntgr0000gp/T/ipykernel_93601/1276650614.py in <module>
          4     ]
          5 }
    ----> 6 fit.to_frame().describe().round(decimals = 2).T
    

    NameError: name 'fit' is not defined


We can generate some examples of a night's sleep, based solely on what is predicted from our prior distributions (that is, not conditioned on the actual data). In the plot below there a 9 draws, where each colour represents a different sleep state. 

    
![png](images/Basic_Markov_Model_12_1.png)
    


Whilst some draws show switching between states, other draws are fixed within once state for an entire evening. We can clearly see the influence of our prior choices here, since these draws will correspond to large values of $\theta_k$ for a specific state $k$. Since each row of the transition matrix must sum to one, if one element is too close to 1, then the rest must be close to 0. 
Our belief for this model was that having a uniform prior over the probability parameter would be uninformative and therefore appropriate, however this does not take into account the relationship betweewn one parameter and the others within a row of the transition matrix. This is discussed in the [STAN documentation](https://mc-stan.org/docs/2_27/functions-reference/dirichlet-distribution.html). To quote directly 
'As the size of the simplex grows, the marginal draws become more and more concentrated below (not around) $1/K$. When one component of the simplex is large, the others must all be relatively small to compensate. For example, in a uniform distribution on  
10-simplexes, the probability that a component is greater than the mean of $1/10$ is only 39%. Most of the posterior marginal probability mass for each component is in the interval $(0, 0.1)$.'
My interpretation of what this all implies is that a sensible prior for $\alpha_k$ should constrain it to be small (close to 0). This will mean that element of the simplex for $\theta_k$ will be concentrated below $1/4$. The code for this is below - I'm using a $Cauchy (0, 1)$ distribution for each $\alpha_k$ so that the fat tails still allow for some chance of an element of each transition matrix row being larger than the others (for example, if I'm awake it's more likely that I'm awake in the next 5 minutes as well). 


```python
model1 = """
data {
    int<lower=1> K; // number of states
    int<lower=1> N; // number of observations
    vector[N] y; // observations
}
parameters {
    vector<lower=0>[K] alpha;
    simplex[K] theta[K]; // transition parameters
}
model {
  alpha ~ cauchy(0, 1);
  for (k in 1:K)
    theta[k] ~ dirichlet(alpha);
}
"""

posterior = stan.build(model1
                       , {"K": 4
                          , "N": subsetFrame.shape[0]
                          , "y": subsetFrame['sleep_state_ordered'].to_list()
                          , "x": subsetFrame['prev_sleep_state'].to_list()})

fit = posterior.sample(num_chains=4, num_samples=2000)
```

This behaviour is what we observe in the summary of each element of $\theta$. The mean of each is close to $1/4$ even though the range covers $[0,1]$. 


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>parameters</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>lp__</th>
      <td>8000.0</td>
      <td>-29.888483</td>
      <td>6.955246</td>
      <td>-6.801273e+01</td>
      <td>-33.883123</td>
      <td>-28.948132</td>
      <td>-24.967955</td>
      <td>-12.985357</td>
    </tr>
    <tr>
      <th>accept_stat__</th>
      <td>8000.0</td>
      <td>0.736764</td>
      <td>0.323544</td>
      <td>0.000000e+00</td>
      <td>0.547259</td>
      <td>0.912921</td>
      <td>0.982102</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>stepsize__</th>
      <td>8000.0</td>
      <td>0.029352</td>
      <td>0.006309</td>
      <td>2.125044e-02</td>
      <td>0.024466</td>
      <td>0.029422</td>
      <td>0.034308</td>
      <td>0.037312</td>
    </tr>
    <tr>
      <th>treedepth__</th>
      <td>8000.0</td>
      <td>5.696750</td>
      <td>1.043756</td>
      <td>0.000000e+00</td>
      <td>5.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>n_leapfrog__</th>
      <td>8000.0</td>
      <td>88.498625</td>
      <td>76.589875</td>
      <td>1.000000e+00</td>
      <td>42.000000</td>
      <td>63.000000</td>
      <td>127.000000</td>
      <td>1023.000000</td>
    </tr>
    <tr>
      <th>divergent__</th>
      <td>8000.0</td>
      <td>0.060125</td>
      <td>0.237733</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>energy__</th>
      <td>8000.0</td>
      <td>37.840897</td>
      <td>7.489333</td>
      <td>1.897966e+01</td>
      <td>32.551424</td>
      <td>36.973158</td>
      <td>42.301909</td>
      <td>77.590966</td>
    </tr>
    <tr>
      <th>alpha.1</th>
      <td>8000.0</td>
      <td>3.491891</td>
      <td>15.781768</td>
      <td>3.885349e-03</td>
      <td>0.382208</td>
      <td>0.918607</td>
      <td>2.207365</td>
      <td>549.144260</td>
    </tr>
    <tr>
      <th>alpha.2</th>
      <td>8000.0</td>
      <td>3.819091</td>
      <td>56.067457</td>
      <td>4.865678e-03</td>
      <td>0.446657</td>
      <td>0.974223</td>
      <td>2.291543</td>
      <td>4639.564793</td>
    </tr>
    <tr>
      <th>alpha.3</th>
      <td>8000.0</td>
      <td>2.582752</td>
      <td>8.040547</td>
      <td>5.578341e-03</td>
      <td>0.439522</td>
      <td>0.965192</td>
      <td>2.186049</td>
      <td>223.204059</td>
    </tr>
    <tr>
      <th>alpha.4</th>
      <td>8000.0</td>
      <td>4.072986</td>
      <td>22.670263</td>
      <td>1.409763e-02</td>
      <td>0.463154</td>
      <td>1.013528</td>
      <td>2.362955</td>
      <td>892.669299</td>
    </tr>
    <tr>
      <th>theta.1.1</th>
      <td>8000.0</td>
      <td>0.244165</td>
      <td>0.276894</td>
      <td>1.168195e-83</td>
      <td>0.017357</td>
      <td>0.124833</td>
      <td>0.398500</td>
      <td>0.999999</td>
    </tr>
    <tr>
      <th>theta.2.1</th>
      <td>8000.0</td>
      <td>0.244830</td>
      <td>0.280370</td>
      <td>3.325500e-102</td>
      <td>0.014171</td>
      <td>0.128837</td>
      <td>0.395157</td>
      <td>0.999867</td>
    </tr>
    <tr>
      <th>theta.3.1</th>
      <td>8000.0</td>
      <td>0.243920</td>
      <td>0.278915</td>
      <td>6.170594e-106</td>
      <td>0.016796</td>
      <td>0.123626</td>
      <td>0.396631</td>
      <td>0.999872</td>
    </tr>
    <tr>
      <th>theta.4.1</th>
      <td>8000.0</td>
      <td>0.244824</td>
      <td>0.278738</td>
      <td>6.820825e-148</td>
      <td>0.015103</td>
      <td>0.126361</td>
      <td>0.403606</td>
      <td>0.999457</td>
    </tr>
    <tr>
      <th>theta.1.2</th>
      <td>8000.0</td>
      <td>0.254811</td>
      <td>0.280824</td>
      <td>5.149602e-141</td>
      <td>0.020860</td>
      <td>0.140788</td>
      <td>0.429359</td>
      <td>0.999998</td>
    </tr>
    <tr>
      <th>theta.2.2</th>
      <td>8000.0</td>
      <td>0.246242</td>
      <td>0.277903</td>
      <td>9.499136e-86</td>
      <td>0.018716</td>
      <td>0.125742</td>
      <td>0.413655</td>
      <td>0.999944</td>
    </tr>
    <tr>
      <th>theta.3.2</th>
      <td>8000.0</td>
      <td>0.251679</td>
      <td>0.280582</td>
      <td>7.121696e-70</td>
      <td>0.019938</td>
      <td>0.131392</td>
      <td>0.419468</td>
      <td>0.999868</td>
    </tr>
    <tr>
      <th>theta.4.2</th>
      <td>8000.0</td>
      <td>0.249856</td>
      <td>0.278315</td>
      <td>2.512304e-54</td>
      <td>0.020400</td>
      <td>0.133504</td>
      <td>0.409666</td>
      <td>0.999990</td>
    </tr>
    <tr>
      <th>theta.1.3</th>
      <td>8000.0</td>
      <td>0.240945</td>
      <td>0.270699</td>
      <td>6.685044e-168</td>
      <td>0.020791</td>
      <td>0.129139</td>
      <td>0.391127</td>
      <td>0.999914</td>
    </tr>
    <tr>
      <th>theta.2.3</th>
      <td>8000.0</td>
      <td>0.246462</td>
      <td>0.277192</td>
      <td>6.036553e-64</td>
      <td>0.018165</td>
      <td>0.129794</td>
      <td>0.408330</td>
      <td>0.999992</td>
    </tr>
    <tr>
      <th>theta.3.3</th>
      <td>8000.0</td>
      <td>0.241025</td>
      <td>0.274413</td>
      <td>3.909004e-55</td>
      <td>0.017484</td>
      <td>0.123752</td>
      <td>0.398925</td>
      <td>0.999998</td>
    </tr>
    <tr>
      <th>theta.4.3</th>
      <td>8000.0</td>
      <td>0.243949</td>
      <td>0.273759</td>
      <td>5.532954e-74</td>
      <td>0.018012</td>
      <td>0.130409</td>
      <td>0.400598</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>theta.1.4</th>
      <td>8000.0</td>
      <td>0.260079</td>
      <td>0.285789</td>
      <td>1.034835e-14</td>
      <td>0.024202</td>
      <td>0.140709</td>
      <td>0.425138</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>theta.2.4</th>
      <td>8000.0</td>
      <td>0.262467</td>
      <td>0.292042</td>
      <td>9.720592e-17</td>
      <td>0.021666</td>
      <td>0.139550</td>
      <td>0.436401</td>
      <td>0.999998</td>
    </tr>
    <tr>
      <th>theta.3.4</th>
      <td>8000.0</td>
      <td>0.263376</td>
      <td>0.287533</td>
      <td>4.996004e-16</td>
      <td>0.025648</td>
      <td>0.146372</td>
      <td>0.434642</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>theta.4.4</th>
      <td>8000.0</td>
      <td>0.261371</td>
      <td>0.288829</td>
      <td>2.220446e-16</td>
      <td>0.023108</td>
      <td>0.136716</td>
      <td>0.431376</td>
      <td>0.999998</td>
    </tr>
  </tbody>
</table>
</div>



Again we can simulate some predicted nights of sleep. Although draws are still mostly stuck in a single state, other draws look closer to a realistic evening, where multiple states are found across the night. 

  
![png](images/Basic_Markov_Model_19_1.png)
    


We can now fit the model to some observed data. This is acheived by including a sampling statement in STAN `y[i] ~ categorical(theta[y[i - 1]]);`.


```python
{

data {
    int<lower=1> K; // number of states
    int<lower=1> N; // number of observations
    int<lower=1, upper =K> y[N]; // observations
}
parameters {
    vector<lower=0>[K] alpha;
    simplex[K] theta[K]; // transition parameters
}
model {
  alpha ~ cauchy(0, 1);
  
  for (k in 1:K)
    theta[k] ~ dirichlet(alpha);
  
  for (i in 2:N)
      y[i] ~ categorical(theta[y[i - 1]]);
}
"""

posterior = stan.build(model3
                       , {"K": 4
                          , "N": subsetFrame.shape[0]
                          , "y": subsetFrame['sleep_state_ordered'].to_list()
                         })

fit = posterior.sample(num_chains=4, num_samples=2000)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>parameters</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>lp__</th>
      <td>8000.0</td>
      <td>-1484.079799</td>
      <td>2.886782</td>
      <td>-1498.597631</td>
      <td>-1485.844757</td>
      <td>-1483.742300</td>
      <td>-1482.005373</td>
      <td>-1477.040311</td>
    </tr>
    <tr>
      <th>accept_stat__</th>
      <td>8000.0</td>
      <td>0.891377</td>
      <td>0.109484</td>
      <td>0.000554</td>
      <td>0.838559</td>
      <td>0.922609</td>
      <td>0.976276</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>stepsize__</th>
      <td>8000.0</td>
      <td>0.539966</td>
      <td>0.014592</td>
      <td>0.525408</td>
      <td>0.525619</td>
      <td>0.538451</td>
      <td>0.552798</td>
      <td>0.557553</td>
    </tr>
    <tr>
      <th>treedepth__</th>
      <td>8000.0</td>
      <td>2.994750</td>
      <td>0.072271</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>n_leapfrog__</th>
      <td>8000.0</td>
      <td>6.989500</td>
      <td>0.204683</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>7.000000</td>
      <td>7.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>divergent__</th>
      <td>8000.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>energy__</th>
      <td>8000.0</td>
      <td>1492.075123</td>
      <td>4.013902</td>
      <td>1480.889527</td>
      <td>1489.234647</td>
      <td>1491.700915</td>
      <td>1494.617506</td>
      <td>1512.131938</td>
    </tr>
    <tr>
      <th>alpha.1</th>
      <td>8000.0</td>
      <td>0.824666</td>
      <td>0.362728</td>
      <td>0.093982</td>
      <td>0.562906</td>
      <td>0.767939</td>
      <td>1.032264</td>
      <td>2.593199</td>
    </tr>
    <tr>
      <th>alpha.2</th>
      <td>8000.0</td>
      <td>1.434563</td>
      <td>0.665816</td>
      <td>0.101816</td>
      <td>0.960160</td>
      <td>1.314853</td>
      <td>1.798026</td>
      <td>5.448049</td>
    </tr>
    <tr>
      <th>alpha.3</th>
      <td>8000.0</td>
      <td>0.563543</td>
      <td>0.256304</td>
      <td>0.059522</td>
      <td>0.374950</td>
      <td>0.523494</td>
      <td>0.705842</td>
      <td>1.932927</td>
    </tr>
    <tr>
      <th>alpha.4</th>
      <td>8000.0</td>
      <td>0.618073</td>
      <td>0.265435</td>
      <td>0.037608</td>
      <td>0.428685</td>
      <td>0.579457</td>
      <td>0.767676</td>
      <td>1.978744</td>
    </tr>
    <tr>
      <th>theta.1.1</th>
      <td>8000.0</td>
      <td>0.583162</td>
      <td>0.028148</td>
      <td>0.469964</td>
      <td>0.563442</td>
      <td>0.583213</td>
      <td>0.602663</td>
      <td>0.685750</td>
    </tr>
    <tr>
      <th>theta.2.1</th>
      <td>8000.0</td>
      <td>0.101662</td>
      <td>0.009754</td>
      <td>0.066306</td>
      <td>0.094891</td>
      <td>0.101420</td>
      <td>0.108083</td>
      <td>0.147855</td>
    </tr>
    <tr>
      <th>theta.3.1</th>
      <td>8000.0</td>
      <td>0.075207</td>
      <td>0.015562</td>
      <td>0.028341</td>
      <td>0.064361</td>
      <td>0.074320</td>
      <td>0.084899</td>
      <td>0.142925</td>
    </tr>
    <tr>
      <th>theta.4.1</th>
      <td>8000.0</td>
      <td>0.070617</td>
      <td>0.017224</td>
      <td>0.022642</td>
      <td>0.058518</td>
      <td>0.069435</td>
      <td>0.081274</td>
      <td>0.148280</td>
    </tr>
    <tr>
      <th>theta.1.2</th>
      <td>8000.0</td>
      <td>0.365140</td>
      <td>0.027460</td>
      <td>0.273528</td>
      <td>0.346406</td>
      <td>0.364747</td>
      <td>0.383943</td>
      <td>0.473221</td>
    </tr>
    <tr>
      <th>theta.2.2</th>
      <td>8000.0</td>
      <td>0.751109</td>
      <td>0.014156</td>
      <td>0.696731</td>
      <td>0.741530</td>
      <td>0.751263</td>
      <td>0.760662</td>
      <td>0.797098</td>
    </tr>
    <tr>
      <th>theta.3.2</th>
      <td>8000.0</td>
      <td>0.174981</td>
      <td>0.022770</td>
      <td>0.096584</td>
      <td>0.159194</td>
      <td>0.174544</td>
      <td>0.189830</td>
      <td>0.276367</td>
    </tr>
    <tr>
      <th>theta.4.2</th>
      <td>8000.0</td>
      <td>0.313494</td>
      <td>0.031038</td>
      <td>0.205925</td>
      <td>0.292178</td>
      <td>0.312696</td>
      <td>0.334336</td>
      <td>0.432065</td>
    </tr>
    <tr>
      <th>theta.1.3</th>
      <td>8000.0</td>
      <td>0.030586</td>
      <td>0.009647</td>
      <td>0.006058</td>
      <td>0.023498</td>
      <td>0.029584</td>
      <td>0.036708</td>
      <td>0.079175</td>
    </tr>
    <tr>
      <th>theta.2.3</th>
      <td>8000.0</td>
      <td>0.068216</td>
      <td>0.008397</td>
      <td>0.041669</td>
      <td>0.062426</td>
      <td>0.067967</td>
      <td>0.073591</td>
      <td>0.101873</td>
    </tr>
    <tr>
      <th>theta.3.3</th>
      <td>8000.0</td>
      <td>0.725885</td>
      <td>0.026808</td>
      <td>0.596106</td>
      <td>0.708077</td>
      <td>0.726587</td>
      <td>0.743937</td>
      <td>0.832003</td>
    </tr>
    <tr>
      <th>theta.4.3</th>
      <td>8000.0</td>
      <td>0.006931</td>
      <td>0.005499</td>
      <td>0.000007</td>
      <td>0.002856</td>
      <td>0.005590</td>
      <td>0.009508</td>
      <td>0.048942</td>
    </tr>
    <tr>
      <th>theta.1.4</th>
      <td>8000.0</td>
      <td>0.021112</td>
      <td>0.008278</td>
      <td>0.002917</td>
      <td>0.015119</td>
      <td>0.020008</td>
      <td>0.026015</td>
      <td>0.069517</td>
    </tr>
    <tr>
      <th>theta.2.4</th>
      <td>8000.0</td>
      <td>0.079014</td>
      <td>0.008723</td>
      <td>0.045541</td>
      <td>0.072905</td>
      <td>0.078730</td>
      <td>0.084763</td>
      <td>0.115762</td>
    </tr>
    <tr>
      <th>theta.3.4</th>
      <td>8000.0</td>
      <td>0.023927</td>
      <td>0.009260</td>
      <td>0.004268</td>
      <td>0.017200</td>
      <td>0.022731</td>
      <td>0.029395</td>
      <td>0.079729</td>
    </tr>
    <tr>
      <th>theta.4.4</th>
      <td>8000.0</td>
      <td>0.608958</td>
      <td>0.032556</td>
      <td>0.490197</td>
      <td>0.587432</td>
      <td>0.609132</td>
      <td>0.631288</td>
      <td>0.715748</td>
    </tr>
  </tbody>
</table>
</div>


The posterior support for the $\alpha_k$ parameters is significantly tighter compared to the support of the prior distribution for these parameters. In turn, this means that the support of the posterior distribution for each element of the transition matrix is much smaller than the prior distribution implies.
We can interpret the transition parameters directly in terms of their impact on a typical night of sleep. The parameter $\theta_{1, 1}$ has a 50% chance of being between 0.58 and 0.69, which means that we're 50% confident that being awake within a 5 minute period will mean there's a 0.58-0.69 chance of being awake in the next 5 minutes. 

Compare this to $\theta_{3, 3}$, which has a 50% chance of being between 0.71 and 0.74. This means that the chances of staying in the REM sleep state is generally higher than the comparable chances of staying in the awake state - which is a good thing as our definition of good sleep means less time awake and more time in REM and deep sleep. 

We're more likely to see a stage of deep sleep directly after light sleep, compared to REM sleep or being awake ($\theta_{2, 4}$ is 50% between 0.073 and 0.085, compared to 0.015 to 0.026 for $\theta_{1, 4}$ and 0.017 to 0.029 for $\theta_{2, 4}$. If we want to maximise deep sleep within a night of sleep, we need to make sure we can be in light sleep beforehand.

One thing worth noting here is that although each night of sleep is a different chain, we model as if it is a single chain. This shouldn't impact the estimates much, as the structure of the model is simple, but it'll need to be improved for more complex versions of the model.


We can again sample some predicted nights of sleep from the model, and investigate the impact of conditioning on the data for our model. Each draw now shows a reasonable amount of mixing between states, which matches the observed behaviour we see in the data. 

The predictive checks also show a few areas in which the model could be improved. Some of the sleep stages (particularly REM and deep), last for much longer than is usually observed (the entire sleep cycle typically lasts 90 minutes, so each stage should be shorter than this). As well, deep sleep is more common earlier in the evening, whereas REM sleep is more common in the early morning after a few hours of sleep. This phenomena is not yet included in our model.

    
![png](images/Basic_Markov_Model_26_1.png)
    


A nice feature of a using a more appropriate model for the process we're interested in, is that it can still be used to produce simple summary stats (this is particularly easy in a Bayesian model, where a summary statistic is just a function of the posterior samples). 

Here we can produce the average amount of each sleep stage within a night of sleep. Below, we plot the entire posterior distribution of the amount of time spent in each state. The posterior distributions look reasonable - they imply the some nights will experience only a small amount of each stage, but that in other nights two hour is probable. The tails cover three hours of REM and deep sleep, which has never occured in my experience (although it has for other people). As we improve the model the coverage of the posterior for these summary values should improve as well.




| Sleep State | Average Time |
|-------------|--------------|
| REM         | 1.275542     |
| awake       | 1.643990     |
| deep        | 1.032833     |
| light       | 4.397354     |



    
![png](images/Basic_Markov_Model_29_1.png)
    


In this post, we set up a basic markov model for sleep, investigated the impact of our prior choices and then fit the model to some data and interpreted the parameters. The markov model is nice for inference because the parameters are directly interpretable. 

In the next post, we will investigate how to include timing effects (time of night, and length of each sleep stage) into the model. 
