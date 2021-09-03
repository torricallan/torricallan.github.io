---
layout: post
title: A Markov Model for sleep stages using Oura Ring data
categories: [Sleep, Markov]
---

# A Markov Model for sleep stages using Oura Ring data

The [Oura ring](https://ouraring.com/) is a device worn on the finger that uses infrared measurements to detect sleep states, as well as other useful biometric data throughout the night(Resting heart rate, heart rate variability, respiratory rate and skin tempurature).

The data proves to be pretty useful for examining long-term trends and factors that influence sleep and overall health ([HRV example](https://ilmostromberg.com/1000-days-with-oura-ring/), [Sleep example](https://www.quantifiedbob.com/sleep-tracking-analysis-oura/). 

Oura provides an developer API that can be used to download raw data. This is great because the summary data provided in app, while useful, discards some valauble time-based information that can be used to better identify trends and moderating factors of better sleep. 

In this post, I set up a basic markov model in STAN, using a [principled Bayesian workflow](https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html). We'll set up the basic model structure with some sensible priors, and then examine how our prior choices will impact inference. Then, we'll fit the model to some actual data and discuss how to improve it in a way that gives us better insight on the sleep data.


```python
"metadata": {
    "tags": [
        "remove_cell"
    ]
}


import pandas as pd
import requests
import stan
import numpy as np
import nest_asyncio
import seaborn as sns
import matplotlib.pyplot as plt
import arviz as az

nest_asyncio.apply()

# Data query

URL = "https://api.ouraring.com/v1/sleep?start=2020-05-15&access_token=5FSOALDAZH7PQTXT5KGNNMZZ365BZJRG"

res = requests.get(URL).json()

df = pd.json_normalize(res['sleep'])

df = df[['summary_date', 'duration', 'hr_5min', 'rmssd_5min', 'hypnogram_5min']]

df = df.assign(sleep_state = df['hypnogram_5min'].str.split(''))
df['sleep_state'] = df['sleep_state'].apply(lambda x: x[:-1])
df['sleep_state'] = df['sleep_state'].apply(lambda x: x[1:])

df = df.assign(hr_count = df['hr_5min'].str.len())
df = df.assign(rmssd_count = df['rmssd_5min'].str.len())
df = df.assign(sleep_count = df['hr_5min'].str.len())

df.loc[df.hr_count != df.sleep_count]['sleep_state'].apply(lambda x: [4] + x)

sleep_df = df[['summary_date', 'duration', 'sleep_state']].explode(['sleep_state'])
sleep_df['measurement_no'] = sleep_df.groupby(['summary_date', 'duration']).cumcount() + 1

hr_df = df[['summary_date', 'duration', 'hr_5min', 'rmssd_5min']].explode(['hr_5min', 'rmssd_5min'])
hr_df['measurement_no'] = hr_df.groupby(['summary_date', 'duration']).cumcount() + 1

forModelling = sleep_df.merge(hr_df, on = ['summary_date', 'duration', 'measurement_no'], how = 'right')

def relabel_states(x):
    switcher = {
        '1':4
        , '2':2
        , '3':3
        , '4':1
    }
    return switcher.get(x)

forModelling['sleep_state_ordered'] = forModelling['sleep_state'].apply(relabel_states)

forModelling['prev_sleep_state'] = forModelling.groupby(['summary_date', 'duration'])['sleep_state_ordered'].shift(1)

forModelling['sleep_state_ordered'] = forModelling['sleep_state_ordered'].fillna(forModelling['prev_sleep_state']).astype(int)
```

Below is an example of the data returned (after a bit of pre-processing). Each row corresponds to a 5 minute period of sleep. `hr_5min` refers to the average heart rate over each measurement period, `rmssd_5min` is the average HRV (using the root mean square of standard deviations measurement) and `sleep_state_ordered` refers to the sleep state within that measurement period, where `1` means awake, `2` means light sleep, `3` means REM sleep and `4` means deep sleep. This is a different ordering to how oura returns the sleep data - it won't impact this analysis but later on we might want to use an ordinal structure for our models. 


```python
{
    "tags": [
        "remove_input"
    ]
}

forModelling[['summary_date', 'duration', 'measurement_no', 'hr_5min', 'rmssd_5min', 'sleep_state_ordered']]
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




```python
subsetFrame = forModelling[forModelling['summary_date'] > '2021-08-15']
subsetFrame = subsetFrame[subsetFrame['measurement_no'] > 1]
```

The basic Markov Model suggests that the next sleep state observed will only be influenced by the current state (aka the memoryless property) $$Pr(Y_t = y | Y_{t-1}).$$ This might be an oversimplification initially, but we will expand on the model once we understand the sensitivity of the model to certain prior choices. 

To convert this into a statistical model, we must specify a likelihood function and some sensible prior choices. 
$$ y_t \sim Categorical(\theta_{y_{t-1}})$$
$$ \theta_k \sim Dirichlet(\alpha), k = 1, 2, 3, 4$$
$$ \alpha \sim Uniform(0, \infty)$$
The next observed state is a function of the transition matrix $\theta$ for the current state. Each row of the transition matrix $\theta$ is drawn from a Dirichlet distribution with hyperprior $\alpha$. We don't specifically set a prior for $\alpha$ yet - the default is for STAN to use a uniform prior over the allowed range of the parameter (the positive reals in this case). 
To test the implications of our selected priors, we can sample from the model without any data.


```python
{
    "tags": [
        "remove_output"
    ]
}
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

    [36mBuilding:[0m 0.1s
    [1A[0J[36mBuilding:[0m 0.2s
    [1A[0J[36mBuilding:[0m 0.4s
    [1A[0J[36mBuilding:[0m 0.5s
    [1A[0J[36mBuilding:[0m 0.6s
    [1A[0J[36mBuilding:[0m 0.7s
    [1A[0J[36mBuilding:[0m 0.8s
    [1A[0J[36mBuilding:[0m 0.9s
    [1A[0J[36mBuilding:[0m 1.0s
    [1A[0J[36mBuilding:[0m 1.1s
    [1A[0J[36mBuilding:[0m 1.2s
    [1A[0J[36mBuilding:[0m 1.3s
    [1A[0J[36mBuilding:[0m 1.4s
    [1A[0J[36mBuilding:[0m 1.5s
    [1A[0J[36mBuilding:[0m 1.6s
    [1A[0J[36mBuilding:[0m 1.7s
    [1A[0J[36mBuilding:[0m 1.8s
    [1A[0J[36mBuilding:[0m 1.9s
    [1A[0J[36mBuilding:[0m 2.0s
    [1A[0J[36mBuilding:[0m 2.1s
    [1A[0J[36mBuilding:[0m 2.2s
    [1A[0J[36mBuilding:[0m 2.3s
    [1A[0J[36mBuilding:[0m 2.4s
    [1A[0J[36mBuilding:[0m 2.5s
    [1A[0J[36mBuilding:[0m 2.6s
    [1A[0J[36mBuilding:[0m 2.8s
    [1A[0J[36mBuilding:[0m 2.9s
    [1A[0J[36mBuilding:[0m 3.0s
    [1A[0J[36mBuilding:[0m 3.1s
    [1A[0J[36mBuilding:[0m 3.2s
    [1A[0J[36mBuilding:[0m 3.3s
    [1A[0J[36mBuilding:[0m 3.4s
    [1A[0J[36mBuilding:[0m 3.5s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:8:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/chainable_object.hpp:6:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/typedefs.hpp:7:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/Eigen_NumTraits.hpp:5:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/core.hpp:4:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/core/init_threadpool_tbb.hpp:104:14: warning: variable 'tbb_max_threads' is used uninitialized whenever 'if' condition is false [-Wsometimes-uninitialized]
      } else if (n_threads == -1) {
                 ^~~~~~~~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/core/init_threadpool_tbb.hpp:112:53: note: uninitialized use occurs here
          tbb::global_control::max_allowed_parallelism, tbb_max_threads);
                                                        ^~~~~~~~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/core/init_threadpool_tbb.hpp:104:10: note: remove the 'if' if its condition is always true
      } else if (n_threads == -1) {
             ^~~~~~~~~~~~~~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/core/init_threadpool_tbb.hpp:99:22: note: initialize the variable 'tbb_max_threads' to silence this warning
      int tbb_max_threads;
                         ^
                          = 0
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:8:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core.hpp:28:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/operator_addition.hpp:6:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_matching_dims.hpp:33:8: warning: unused variable 'error' [-Wunused-variable]
      bool error = false;
           ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_matching_dims.hpp:57:23: warning: comparison of integers of different signs: 'int' and 'std::__1::vector<int, std::__1::allocator<int> >::size_type' (aka 'unsigned long') [-Wsign-compare]
        for (int i = 0; i < y1_d.size(); i++) {
                        ~ ^ ~~~~~~~~~~~


    [1A[0J[36mBuilding:[0m 3.6s
    [1A[0J[36mBuilding:[0m 3.7s
    [1A[0J[36mBuilding:[0m 3.8s
    [1A[0J[36mBuilding:[0m 3.9s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:8:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core.hpp:53:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/profiling.hpp:9:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err.hpp:20:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_less.hpp:63:16: warning: unused variable 'n' [-Wunused-variable]
      Eigen::Index n = 0;
                   ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_less.hpp:98:16: warning: unused variable 'n' [-Wunused-variable]
      Eigen::Index n = 0;
                   ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_less.hpp:133:16: warning: unused variable 'n' [-Wunused-variable]
      Eigen::Index n = 0;
                   ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:8:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core.hpp:53:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/profiling.hpp:9:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err.hpp:50:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/hmm_check.hpp:33:7: warning: unused variable 'n_transitions' [-Wunused-variable]
      int n_transitions = log_omegas.cols() - 1;
          ^


    [1A[0J[36mBuilding:[0m 4.0s
    [1A[0J[36mBuilding:[0m 4.1s
    [1A[0J[36mBuilding:[0m 4.2s
    [1A[0J[36mBuilding:[0m 4.3s
    [1A[0J[36mBuilding:[0m 4.4s
    [1A[0J[36mBuilding:[0m 4.5s
    [1A[0J[36mBuilding:[0m 4.6s
    [1A[0J[36mBuilding:[0m 4.7s
    [1A[0J[36mBuilding:[0m 4.8s
    [1A[0J[36mBuilding:[0m 4.9s
    [1A[0J[36mBuilding:[0m 5.0s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:7:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/fun.hpp:118:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/fun/gp_matern52_cov.hpp:304:10: warning: unused variable 'neg_root_5' [-Wunused-variable]
      double neg_root_5 = -root_5;
             ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:7:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/fun.hpp:183:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/fun/log_mix.hpp:86:13: warning: unused variable 'N' [-Wunused-variable]
      const int N = stan::math::size(theta);
                ^


    [1A[0J[36mBuilding:[0m 5.2s
    [1A[0J[36mBuilding:[0m 5.3s
    [1A[0J[36mBuilding:[0m 5.4s
    [1A[0J[36mBuilding:[0m 5.5s
    [1A[0J[36mBuilding:[0m 5.6s
    [1A[0J[36mBuilding:[0m 5.7s
    [1A[0J[36mBuilding:[0m 5.8s
    [1A[0J[36mBuilding:[0m 5.9s
    [1A[0J[36mBuilding:[0m 6.0s
    [1A[0J[36mBuilding:[0m 6.1s
    [1A[0J[36mBuilding:[0m 6.2s
    [1A[0J[36mBuilding:[0m 6.3s
    [1A[0J[36mBuilding:[0m 6.4s
    [1A[0J[36mBuilding:[0m 6.5s
    [1A[0J[36mBuilding:[0m 6.6s
    [1A[0J[36mBuilding:[0m 6.7s
    [1A[0J[36mBuilding:[0m 6.8s
    [1A[0J[36mBuilding:[0m 6.9s
    [1A[0J[36mBuilding:[0m 7.0s
    [1A[0J[36mBuilding:[0m 7.1s
    [1A[0J[36mBuilding:[0m 7.2s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:28:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/beta.hpp:70:32: warning: lambda capture 'b' is not used [-Wunused-lambda-capture]
                               [a, b, digamma_ab](auto& vi) mutable {
                                 ~~^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/beta.hpp:96:39: warning: lambda capture 'a' is not used [-Wunused-lambda-capture]
      return make_callback_var(beta_val, [a, b, digamma_ab](auto& vi) mutable {
                                          ^~


    [1A[0J[36mBuilding:[0m 7.3s
    [1A[0J[36mBuilding:[0m 7.5s
    [1A[0J[36mBuilding:[0m 7.6s
    [1A[0J[36mBuilding:[0m 7.7s
    [1A[0J[36mBuilding:[0m 7.8s
    [1A[0J[36mBuilding:[0m 7.9s
    [1A[0J[36mBuilding:[0m 8.0s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:123:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/matrix_power.hpp:52:24: warning: comparison of integers of different signs: 'size_t' (aka 'unsigned long') and 'const int' [-Wsign-compare]
      for (size_t i = 2; i <= n; ++i) {
                         ~ ^  ~
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:134:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/ordered_constrain.hpp:40:30: warning: comparison of integers of different signs: 'Eigen::Index' (aka 'long') and 'size_t' (aka 'unsigned long') [-Wsign-compare]
      for (Eigen::Index n = 1; n < N; ++n) {
                               ~ ^ ~
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:137:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/positive_ordered_constrain.hpp:40:21: warning: comparison of integers of different signs: 'int' and 'size_t' (aka 'unsigned long') [-Wsign-compare]
      for (int n = 1; n < N; ++n) {
                      ~ ^ ~
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:153:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/simplex_constrain.hpp:40:30: warning: comparison of integers of different signs: 'Eigen::Index' (aka 'long') and 'size_t' (aka 'unsigned long') [-Wsign-compare]
      for (Eigen::Index k = 0; k < N; ++k) {
                               ~ ^ ~


    [1A[0J[36mBuilding:[0m 8.1s
    [1A[0J[36mBuilding:[0m 8.2s


    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/simplex_constrain.hpp:94:30: warning: comparison of integers of different signs: 'Eigen::Index' (aka 'long') and 'size_t' (aka 'unsigned long') [-Wsign-compare]
      for (Eigen::Index k = 0; k < N; ++k) {
                               ~ ^ ~


    [1A[0J[36mBuilding:[0m 8.3s
    [1A[0J[36mBuilding:[0m 8.4s
    [1A[0J[36mBuilding:[0m 8.5s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:11:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor.hpp:27:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor/operands_and_partials.hpp:29:1: warning: 'ops_partials_edge' defined as a class template here but previously declared as a struct template; this is valid, but may result in linker errors under the Microsoft C++ ABI [-Wmismatched-tags]
    class ops_partials_edge<double, var> {
    ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/functor/operands_and_partials.hpp:21:1: note: did you mean class here?
    struct ops_partials_edge;
    ^~~~~~
    class
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:11:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor.hpp:27:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor/operands_and_partials.hpp:177:1: warning: 'ops_partials_edge' defined as a class template here but previously declared as a struct template; this is valid, but may result in linker errors under the Microsoft C++ ABI [-Wmismatched-tags]
    class ops_partials_edge<double, std::vector<var>> {
    ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/functor/operands_and_partials.hpp:21:1: note: did you mean class here?
    struct ops_partials_edge;
    ^~~~~~
    class
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:11:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor.hpp:27:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor/operands_and_partials.hpp:199:1: warning: 'ops_partials_edge' defined as a class template here but previously declared as a struct template; this is valid, but may result in linker errors under the Microsoft C++ ABI [-Wmismatched-tags]
    class ops_partials_edge<double, Op, require_eigen_st<is_var, Op>> {
    ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/functor/operands_and_partials.hpp:21:1: note: did you mean class here?
    struct ops_partials_edge;
    ^~~~~~
    class
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:11:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor.hpp:27:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor/operands_and_partials.hpp:219:1: warning: 'ops_partials_edge' defined as a class template here but previously declared as a struct template; this is valid, but may result in linker errors under the Microsoft C++ ABI [-Wmismatched-tags]
    class ops_partials_edge<double, var_value<Op>, require_eigen_t<Op>> {
    ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/functor/operands_and_partials.hpp:21:1: note: did you mean class here?
    struct ops_partials_edge;
    ^~~~~~
    class
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:11:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor.hpp:27:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor/operands_and_partials.hpp:245:1: warning: 'ops_partials_edge' defined as a class template here but previously declared as a struct template; this is valid, but may result in linker errors under the Microsoft C++ ABI [-Wmismatched-tags]
    class ops_partials_edge<double, std::vector<Eigen::Matrix<var, R, C>>> {
    ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/functor/operands_and_partials.hpp:21:1: note: did you mean class here?
    struct ops_partials_edge;
    ^~~~~~
    class
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:11:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor.hpp:27:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor/operands_and_partials.hpp:274:1: warning: 'ops_partials_edge' defined as a class template here but previously declared as a struct template; this is valid, but may result in linker errors under the Microsoft C++ ABI [-Wmismatched-tags]
    class ops_partials_edge<double, std::vector<std::vector<var>>> {
    ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/functor/operands_and_partials.hpp:21:1: note: did you mean class here?
    struct ops_partials_edge;
    ^~~~~~
    class
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:11:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor.hpp:27:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor/operands_and_partials.hpp:300:1: warning: 'ops_partials_edge' defined as a class template here but previously declared as a struct template; this is valid, but may result in linker errors under the Microsoft C++ ABI [-Wmismatched-tags]
    class ops_partials_edge<double, std::vector<var_value<Op>>,
    ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/functor/operands_and_partials.hpp:21:1: note: did you mean class here?
    struct ops_partials_edge;
    ^~~~~~
    class
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:11:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor.hpp:29:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor/finite_diff_hessian_auto.hpp:61:24: warning: comparison of integers of different signs: 'size_t' (aka 'unsigned long') and 'int' [-Wsign-compare]
      for (size_t i = 0; i < d; ++i) {
                         ~ ^ ~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor/finite_diff_hessian_auto.hpp:69:24: warning: comparison of integers of different signs: 'size_t' (aka 'unsigned long') and 'int' [-Wsign-compare]
      for (size_t i = 0; i < d; ++i) {
                         ~ ^ ~


    [1A[0J[36mBuilding:[0m 8.6s
    [1A[0J[36mBuilding:[0m 8.7s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:87:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/double_exponential_cdf.hpp:77:10: warning: unused variable 'N' [-Wunused-variable]
      size_t N = max_size(y, mu, sigma);
             ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:128:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/gaussian_dlm_obs_rng.hpp:41:21: warning: comparison of integers of different signs: 'int' and 'size_t' (aka 'unsigned long') [-Wsign-compare]
      for (int i = 0; i < M; i++) {
                      ~ ^ ~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/gaussian_dlm_obs_rng.hpp:98:7: warning: unused variable 'n' [-Wunused-variable]
      int n = G.rows();  // number of states
          ^


    [1A[0J[36mBuilding:[0m 8.8s
    [1A[0J[36mBuilding:[0m 8.9s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:139:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/hmm_marginal.hpp:26:13: warning: unused variable 'n_states' [-Wunused-variable]
      const int n_states = omegas.rows();
                ^


    [1A[0J[36mBuilding:[0m 9.0s
    [1A[0J[36mBuilding:[0m 9.1s
    [1A[0J[36mBuilding:[0m 9.2s
    [1A[0J[36mBuilding:[0m 9.3s
    [1A[0J[36mBuilding:[0m 9.4s
    [1A[0J[36mBuilding:[0m 9.5s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:330:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/std_normal_rng.hpp:23:22: warning: unused variable 'function' [-Wunused-variable]
      static const char* function = "std_normal_rng";
                         ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:350:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/von_mises_cdf.hpp:72:10: warning: unused variable 'ck' [-Wunused-variable]
      double ck = 50;
             ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:8:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/io/deserializer.hpp:337:28: warning: comparison of integers of different signs: 'size_t' (aka 'unsigned long') and 'Eigen::Index' (aka 'long') [-Wsign-compare]
          for (size_t i = 0; i < m; ++i) {
                             ~ ^ ~


    [1A[0J[36mBuilding:[0m 9.7s
    [1A[0J[36mBuilding:[0m 9.8s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:17:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing.hpp:6:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/assign.hpp:268:21: warning: comparison of integers of different signs: 'int' and 'std::__1::vector<int, std::__1::allocator<int> >::size_type' (aka 'unsigned long') [-Wsign-compare]
      for (int i = 0; i < idx.ns_.size(); ++i) {
                      ~ ^ ~~~~~~~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/assign.hpp:531:21: warning: comparison of integers of different signs: 'int' and 'std::__1::vector<int, std::__1::allocator<int> >::size_type' (aka 'unsigned long') [-Wsign-compare]
      for (int i = 0; i < col_idx.ns_.size(); ++i) {
                      ~ ^ ~~~~~~~~~~~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/assign.hpp:630:21: warning: comparison of integers of different signs: 'int' and 'std::__1::vector<int, std::__1::allocator<int> >::size_type' (aka 'unsigned long') [-Wsign-compare]
      for (int j = 0; j < col_idx.ns_.size(); ++j) {
                      ~ ^ ~~~~~~~~~~~~~~~~~~
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:17:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing.hpp:8:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue.hpp:159:21: warning: comparison of integers of different signs: 'int' and 'std::__1::vector<int, std::__1::allocator<int> >::size_type' (aka 'unsigned long') [-Wsign-compare]
      for (int i = 0; i < idx.ns_.size(); ++i) {
                      ~ ^ ~~~~~~~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue.hpp:266:21: warning: comparison of integers of different signs: 'int' and 'std::__1::vector<int, std::__1::allocator<int> >::size_type' (aka 'unsigned long') [-Wsign-compare]
      for (int i = 0; i < idx.ns_.size(); ++i) {
                      ~ ^ ~~~~~~~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue.hpp:441:21: warning: comparison of integers of different signs: 'int' and 'std::__1::vector<int, std::__1::allocator<int> >::size_type' (aka 'unsigned long') [-Wsign-compare]
      for (int i = 0; i < col_idx.ns_.size(); ++i) {
                      ~ ^ ~~~~~~~~~~~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue.hpp:471:21: warning: comparison of integers of different signs: 'int' and 'std::__1::vector<int, std::__1::allocator<int> >::size_type' (aka 'unsigned long') [-Wsign-compare]
      for (int i = 0; i < row_idx.ns_.size(); ++i) {
                      ~ ^ ~~~~~~~~~~~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue.hpp:556:13: warning: unused variable 'cols' [-Wunused-variable]
      const int cols = rvalue_index_size(col_idx, x_ref.cols());
                ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue.hpp:558:21: warning: comparison of integers of different signs: 'int' and 'std::__1::vector<int, std::__1::allocator<int> >::size_type' (aka 'unsigned long') [-Wsign-compare]
      for (int j = 0; j < col_idx.ns_.size(); ++j) {
                      ~ ^ ~~~~~~~~~~~~~~~~~~
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:17:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing.hpp:9:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue_varmat.hpp:66:21: warning: comparison of integers of different signs: 'int' and 'const unsigned long' [-Wsign-compare]
      for (int i = 0; i < ret_size; ++i) {
                      ~ ^ ~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue_varmat.hpp:102:21: warning: comparison of integers of different signs: 'int' and 'const unsigned long' [-Wsign-compare]
      for (int i = 0; i < ret_rows; ++i) {
                      ~ ^ ~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue_varmat.hpp:142:21: warning: comparison of integers of different signs: 'int' and 'const unsigned long' [-Wsign-compare]
      for (int j = 0; j < ret_size; ++j) {
                      ~ ^ ~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue_varmat.hpp:183:21: warning: comparison of integers of different signs: 'int' and 'const unsigned long' [-Wsign-compare]
      for (int i = 0; i < ret_size; ++i) {
                      ~ ^ ~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue_varmat.hpp:222:22: warning: unused variable 'x_cols' [-Wunused-variable]
      const Eigen::Index x_cols = x.cols();
                         ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue_varmat.hpp:227:21: warning: comparison of integers of different signs: 'int' and 'const unsigned long' [-Wsign-compare]
      for (int i = 0; i < ret_rows; ++i) {
                      ~ ^ ~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue_varmat.hpp:231:21: warning: comparison of integers of different signs: 'int' and 'const unsigned long' [-Wsign-compare]
      for (int j = 0; j < ret_cols; ++j) {
                      ~ ^ ~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue_varmat.hpp:235:23: warning: comparison of integers of different signs: 'int' and 'const unsigned long' [-Wsign-compare]
        for (int i = 0; i < ret_rows; ++i) {
                        ~ ^ ~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue_varmat.hpp:278:21: warning: comparison of integers of different signs: 'int' and 'const unsigned long' [-Wsign-compare]
      for (int j = 0; j < ret_cols; ++j) {
                      ~ ^ ~~~~~~~~


    [1A[0J[36mBuilding:[0m 9.9s
    [1A[0J[36mBuilding:[0m 10.0s
    [1A[0J[36mBuilding:[0m 10.1s
    [1A[0J[36mBuilding:[0m 10.2s
    [1A[0J[36mBuilding:[0m 10.3s
    [1A[0J[36mBuilding:[0m 10.4s
    [1A[0J[36mBuilding:[0m 10.5s
    [1A[0J[36mBuilding:[0m 10.6s
    [1A[0J[36mBuilding:[0m 10.7s
    [1A[0J[36mBuilding:[0m 10.8s
    [1A[0J[36mBuilding:[0m 10.9s
    [1A[0J[36mBuilding:[0m 11.0s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:8:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core.hpp:29:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/operator_divide_equal.hpp:5:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/operator_division.hpp:15:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/operator_subtraction.hpp:84:21: warning: lambda capture 'b' is not used [-Wunused-lambda-capture]
          [avi = a.vi_, b](const auto& vi) mutable { avi->adj_ += vi.adj_; });
                        ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/operator_minus_equal.hpp:24:16: note: in instantiation of function template specialization 'stan::math::operator-<double, nullptr>' requested here
      vi_ = (*this - b).vi_;
                   ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/core/complex_base.hpp:136:9: note: in instantiation of member function 'stan::math::var_value<double, void>::operator-=' requested here
        re_ -= x;
            ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/core/operator_subtraction.hpp:24:5: note: in instantiation of function template specialization 'stan::math::complex_base<stan::math::var_value<double, void> >::operator-=<int>' requested here
      y -= rhs;
        ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/core/operator_subtraction.hpp:55:20: note: in instantiation of function template specialization 'stan::math::internal::complex_subtract<std::__1::complex<stan::math::var>, int>' requested here
      return internal::complex_subtract(x, y);
                       ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/fun/acosh.hpp:105:31: note: in instantiation of function template specialization 'stan::math::operator-<stan::math::var_value<double, void>, int>' requested here
      auto y = log(z + sqrt(z * z - 1));
                                  ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/acosh.hpp:94:32: note: in instantiation of function template specialization 'stan::math::internal::complex_acosh<stan::math::var_value<double, void> >' requested here
      return stan::math::internal::complex_acosh(z);
                                   ^


    [1A[0J[36mBuilding:[0m 11.1s
    [1A[0J[36mBuilding:[0m 11.2s
    [1A[0J[36mBuilding:[0m 11.3s
    [1A[0J[36mBuilding:[0m 11.4s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:8:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core.hpp:28:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/operator_addition.hpp:80:21: warning: lambda capture 'b' is not used [-Wunused-lambda-capture]
          [avi = a.vi_, b](const auto& vi) mutable { avi->adj_ += vi.adj_; });
                        ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/grad_inc_beta.hpp:45:43: note: in instantiation of function template specialization 'stan::math::operator+<int, nullptr>' requested here
        grad_2F1(dF1, dF2, a + b, var(1.0), a + 1, z);
                                              ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:8:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core.hpp:28:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/operator_addition.hpp:80:21: warning: lambda capture 'b' is not used [-Wunused-lambda-capture]
          [avi = a.vi_, b](const auto& vi) mutable { avi->adj_ += vi.adj_; });
                        ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/fun/trigamma.hpp:63:31: note: in instantiation of function template specialization 'stan::math::operator+<double, nullptr>' requested here
        value = -trigamma_impl(-x + 1.0) + square(pi() / sin(-pi() * x));
                                  ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/trigamma.hpp:23:44: note: in instantiation of function template specialization 'stan::math::trigamma_impl<stan::math::var_value<double, void> >' requested here
    inline var trigamma(const var& u) { return trigamma_impl(u); }
                                               ^


    [1A[0J[36mBuilding:[0m 11.5s
    [1A[0J[36mBuilding:[0m 11.7s
    [1A[0J[36mBuilding:[0m 11.8s
    [1A[0J[36mBuilding:[0m 11.9s
    [1A[0J[36mBuilding:[0m 12.0s
    [1A[0J[36mBuilding:[0m 12.1s
    [1A[0J[36mBuilding:[0m 12.2s
    [1A[0J[36mBuilding:[0m 12.3s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:8:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core.hpp:53:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/profiling.hpp:9:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_2F1_converges.hpp:5:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_not_nan.hpp:5:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/elementwise_check.hpp:153:24: warning: comparison of integers of different signs: 'size_t' (aka 'unsigned long') and 'Eigen::EigenBase::Index' (aka 'long') [-Wsign-compare]
      for (size_t i = 0; i < x.size(); i++) {
                         ~ ^ ~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_positive.hpp:29:3: note: in instantiation of function template specialization 'stan::math::elementwise_check<(lambda at /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_positive.hpp:29:21), Eigen::Matrix<double, -1, 1, 0, -1, 1>, nullptr, nullptr>' requested here
      elementwise_check([](double x) { return x > 0; }, function, name, y,
      ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/dirichlet_lpdf.hpp:78:5: note: in instantiation of function template specialization 'stan::math::check_positive<Eigen::Matrix<double, -1, 1, 0, -1, 1> >' requested here
        check_positive(function, "prior sample sizes", alpha_vec[t]);
        ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:163:13: note: in instantiation of function template specialization 'stan::math::dirichlet_lpdf<false, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, nullptr>' requested here
                dirichlet_lpdf<propto__>(rvalue(theta, "theta", index_uni(k)),
                ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:453:14: note: in instantiation of function template specialization 'model_vloc2crl_namespace::model_vloc2crl::log_prob_impl<false, false, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, nullptr, nullptr>' requested here
          return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
                 ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_base_crtp.hpp:90:50: note: in instantiation of function template specialization 'model_vloc2crl_namespace::model_vloc2crl::log_prob<false, false, double>' requested here
        return static_cast<const M*>(this)->template log_prob<false, false, double>(
                                                     ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:44:3: note: in instantiation of member function 'stan::model::model_base_crtp<model_vloc2crl_namespace::model_vloc2crl>::log_prob' requested here
      ~model_vloc2crl() { }
      ^


    [1A[0J[36mBuilding:[0m 12.4s
    [1A[0J[36mBuilding:[0m 12.5s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:96:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/lb_constrain.hpp:214:35: warning: lambda capture 'lp' is not used [-Wunused-lambda-capture]
          reverse_pass_callback([ret, lp, arena_lb = var(lb)]() mutable {
                                      ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/io/deserializer.hpp:386:26: note: in instantiation of function template specialization 'stan::math::lb_constrain<Eigen::Map<Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, 0, Eigen::Stride<0, 0> >, int, nullptr, nullptr, nullptr>' requested here
          return stan::math::lb_constrain(this->read<Ret>(sizes...), lb, lp);
                             ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:149:29: note: in instantiation of function template specialization 'stan::io::deserializer<stan::math::var_value<double, void> >::read_constrain_lb<Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, false, int, stan::math::var_value<double, void>, int>' requested here
          alpha = in__.template read_constrain_lb<Eigen::Matrix<local_scalar_t__, -1, 1>, jacobian__>(
                                ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:453:14: note: in instantiation of function template specialization 'model_vloc2crl_namespace::model_vloc2crl::log_prob_impl<false, false, Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, nullptr, nullptr>' requested here
          return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
                 ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_base_crtp.hpp:95:50: note: in instantiation of function template specialization 'model_vloc2crl_namespace::model_vloc2crl::log_prob<false, false, stan::math::var_value<double, void> >' requested here
        return static_cast<const M*>(this)->template log_prob<false, false>(theta,
                                                     ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:44:3: note: in instantiation of member function 'stan::model::model_base_crtp<model_vloc2crl_namespace::model_vloc2crl>::log_prob' requested here
      ~model_vloc2crl() { }
      ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:153:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/simplex_constrain.hpp:94:30: warning: comparison of integers of different signs: 'Eigen::Index' (aka 'long') and 'size_t' (aka 'unsigned long') [-Wsign-compare]
      for (Eigen::Index k = 0; k < N; ++k) {
                               ~ ^ ~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/io/deserializer.hpp:569:14: note: in instantiation of function template specialization 'stan::math::simplex_constrain<Eigen::Map<Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, 0, Eigen::Stride<0, 0> >, nullptr>' requested here
          return simplex_constrain(this->read<Ret>(size - 1), lp);
                 ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/io/deserializer.hpp:602:17: note: in instantiation of function template specialization 'stan::io::deserializer<stan::math::var_value<double, void> >::read_constrain_simplex<Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, false, stan::math::var_value<double, void>, nullptr>' requested here
              this->read_constrain_simplex<value_type_t<Ret>, Jacobian>(lp,
                    ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:156:29: note: in instantiation of function template specialization 'stan::io::deserializer<stan::math::var_value<double, void> >::read_constrain_simplex<std::__1::vector<Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, std::__1::allocator<Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1> > >, false, stan::math::var_value<double, void>, int, nullptr>' requested here
          theta = in__.template read_constrain_simplex<std::vector<Eigen::Matrix<local_scalar_t__, -1, 1>>, jacobian__>(
                                ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:453:14: note: in instantiation of function template specialization 'model_vloc2crl_namespace::model_vloc2crl::log_prob_impl<false, false, Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, nullptr, nullptr>' requested here
          return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
                 ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_base_crtp.hpp:95:50: note: in instantiation of function template specialization 'model_vloc2crl_namespace::model_vloc2crl::log_prob<false, false, stan::math::var_value<double, void> >' requested here
        return static_cast<const M*>(this)->template log_prob<false, false>(theta,
                                                     ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:44:3: note: in instantiation of member function 'stan::model::model_base_crtp<model_vloc2crl_namespace::model_vloc2crl>::log_prob' requested here
      ~model_vloc2crl() { }
      ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:153:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/simplex_constrain.hpp:40:30: warning: comparison of integers of different signs: 'Eigen::Index' (aka 'long') and 'size_t' (aka 'unsigned long') [-Wsign-compare]
      for (Eigen::Index k = 0; k < N; ++k) {
                               ~ ^ ~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/io/deserializer.hpp:571:14: note: in instantiation of function template specialization 'stan::math::simplex_constrain<Eigen::Map<Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, 0, Eigen::Stride<0, 0> >, nullptr>' requested here
          return simplex_constrain(this->read<Ret>(size - 1));
                 ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/io/deserializer.hpp:602:17: note: in instantiation of function template specialization 'stan::io::deserializer<stan::math::var_value<double, void> >::read_constrain_simplex<Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, false, stan::math::var_value<double, void>, nullptr>' requested here
              this->read_constrain_simplex<value_type_t<Ret>, Jacobian>(lp,
                    ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:156:29: note: in instantiation of function template specialization 'stan::io::deserializer<stan::math::var_value<double, void> >::read_constrain_simplex<std::__1::vector<Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, std::__1::allocator<Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1> > >, false, stan::math::var_value<double, void>, int, nullptr>' requested here
          theta = in__.template read_constrain_simplex<std::vector<Eigen::Matrix<local_scalar_t__, -1, 1>>, jacobian__>(
                                ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:453:14: note: in instantiation of function template specialization 'model_vloc2crl_namespace::model_vloc2crl::log_prob_impl<false, false, Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, nullptr, nullptr>' requested here
          return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
                 ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_base_crtp.hpp:95:50: note: in instantiation of function template specialization 'model_vloc2crl_namespace::model_vloc2crl::log_prob<false, false, stan::math::var_value<double, void> >' requested here
        return static_cast<const M*>(this)->template log_prob<false, false>(theta,
                                                     ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:44:3: note: in instantiation of member function 'stan::model::model_base_crtp<model_vloc2crl_namespace::model_vloc2crl>::log_prob' requested here
      ~model_vloc2crl() { }
      ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:8:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core.hpp:53:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/profiling.hpp:9:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_2F1_converges.hpp:5:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_not_nan.hpp:5:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/elementwise_check.hpp:153:24: warning: comparison of integers of different signs: 'size_t' (aka 'unsigned long') and 'Eigen::EigenBase::Index' (aka 'long') [-Wsign-compare]
      for (size_t i = 0; i < x.size(); i++) {
                         ~ ^ ~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_positive.hpp:29:3: note: in instantiation of function template specialization 'stan::math::elementwise_check<(lambda at /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_positive.hpp:29:21), Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, nullptr, nullptr>' requested here
      elementwise_check([](double x) { return x > 0; }, function, name, y,
      ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/dirichlet_lpdf.hpp:78:5: note: in instantiation of function template specialization 'stan::math::check_positive<Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1> >' requested here
        check_positive(function, "prior sample sizes", alpha_vec[t]);
        ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:163:13: note: in instantiation of function template specialization 'stan::math::dirichlet_lpdf<false, Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, nullptr>' requested here
                dirichlet_lpdf<propto__>(rvalue(theta, "theta", index_uni(k)),
                ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:453:14: note: in instantiation of function template specialization 'model_vloc2crl_namespace::model_vloc2crl::log_prob_impl<false, false, Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, nullptr, nullptr>' requested here
          return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
                 ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_base_crtp.hpp:95:50: note: in instantiation of function template specialization 'model_vloc2crl_namespace::model_vloc2crl::log_prob<false, false, stan::math::var_value<double, void> >' requested here
        return static_cast<const M*>(this)->template log_prob<false, false>(theta,
                                                     ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:44:3: note: in instantiation of member function 'stan::model::model_base_crtp<model_vloc2crl_namespace::model_vloc2crl>::log_prob' requested here
      ~model_vloc2crl() { }
      ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:8:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core.hpp:29:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/operator_divide_equal.hpp:5:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/operator_division.hpp:15:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/operator_subtraction.hpp:104:21: warning: lambda capture 'a' is not used [-Wunused-lambda-capture]
          [bvi = b.vi_, a](const auto& vi) mutable { bvi->adj_ -= vi.adj_; });
                        ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_simplex.hpp:43:18: note: in instantiation of function template specialization 'stan::math::operator-<double, nullptr>' requested here
      if (!(fabs(1.0 - theta_ref.sum()) <= CONSTRAINT_TOLERANCE)) {
                     ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/dirichlet_lpdf.hpp:79:5: note: in instantiation of function template specialization 'stan::math::check_simplex<Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, nullptr>' requested here
        check_simplex(function, "probabilities", theta_vec[t]);
        ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:163:13: note: in instantiation of function template specialization 'stan::math::dirichlet_lpdf<false, Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, nullptr>' requested here
                dirichlet_lpdf<propto__>(rvalue(theta, "theta", index_uni(k)),
                ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:453:14: note: in instantiation of function template specialization 'model_vloc2crl_namespace::model_vloc2crl::log_prob_impl<false, false, Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, nullptr, nullptr>' requested here
          return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
                 ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_base_crtp.hpp:95:50: note: in instantiation of function template specialization 'model_vloc2crl_namespace::model_vloc2crl::log_prob<false, false, stan::math::var_value<double, void> >' requested here
        return static_cast<const M*>(this)->template log_prob<false, false>(theta,
                                                     ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:44:3: note: in instantiation of member function 'stan::model::model_base_crtp<model_vloc2crl_namespace::model_vloc2crl>::log_prob' requested here
      ~model_vloc2crl() { }
      ^


    [1A[0J[36mBuilding:[0m 12.6s
    [1A[0J[36mBuilding:[0m 12.7s
    [1A[0J[36mBuilding:[0m 12.8s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:8:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core.hpp:58:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/set_zero_all_adjoints.hpp:14:13: warning: 'static' function 'set_zero_all_adjoints' declared in header file should be declared 'static inline' [-Wunneeded-internal-declaration]
    static void set_zero_all_adjoints() {
                ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:7:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/fun.hpp:110:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/fun/generalized_inverse.hpp:34:9: warning: unused type alias 'value_t' [-Wunused-local-typedef]
      using value_t = value_type_t<EigMat>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:7:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/fun.hpp:321:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/fun/tail.hpp:42:9: warning: unused type alias 'idx_t' [-Wunused-local-typedef]
      using idx_t = index_type_t<std::vector<T>>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:32:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/cholesky_decompose.hpp:83:11: warning: unused type alias 'Block_' [-Wunused-local-typedef]
        using Block_ = Eigen::Block<Eigen::MatrixXd>;
              ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:34:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/cholesky_factor_constrain.hpp:32:9: warning: unused type alias 'T_scalar' [-Wunused-local-typedef]
      using T_scalar = value_type_t<T>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:74:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/generalized_inverse.hpp:64:9: warning: unused type alias 'value_t' [-Wunused-local-typedef]
      using value_t = value_type_t<VarMat>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:121:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/lub_constrain.hpp:485:9: warning: unused type alias 'plain_x_array' [-Wunused-local-typedef]
      using plain_x_array = plain_type_t<decltype(arena_x_val.array())>;
            ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/lub_constrain.hpp:565:9: warning: unused type alias 'plain_x_array' [-Wunused-local-typedef]
      using plain_x_array = plain_type_t<decltype(arena_x_val.array())>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/bernoulli_logit_glm_log.hpp:5:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/bernoulli_logit_glm_lpmf.hpp:59:9: warning: unused type alias 'T_y_val' [-Wunused-local-typedef]
      using T_y_val =
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:31:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/beta_log.hpp:5:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/beta_lpdf.hpp:50:9: warning: unused type alias 'T_partials_matrix' [-Wunused-local-typedef]
      using T_partials_matrix = Eigen::Matrix<T_partials_return, Eigen::Dynamic, 1>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:37:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/beta_proportion_log.hpp:5:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/beta_proportion_lpdf.hpp:56:9: warning: unused type alias 'T_partials_array' [-Wunused-local-typedef]
      using T_partials_array = Eigen::Array<T_partials_return, Eigen::Dynamic, 1>;
            ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/beta_proportion_lpdf.hpp:55:9: warning: unused type alias 'T_partials_return_kappa' [-Wunused-local-typedef]
      using T_partials_return_kappa = return_type_t<T_prec>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:63:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/cauchy_log.hpp:5:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/cauchy_lpdf.hpp:46:9: warning: unused type alias 'T_partials_array' [-Wunused-local-typedef]
      using T_partials_array = Eigen::Array<T_partials_return, Eigen::Dynamic, 1>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:74:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/dirichlet_log.hpp:5:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/dirichlet_lpdf.hpp:62:9: warning: unused type alias 'T_partials_vec' [-Wunused-local-typedef]
      using T_partials_vec = typename Eigen::Matrix<T_partials_return, -1, 1>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:91:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/double_exponential_log.hpp:5:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/double_exponential_lpdf.hpp:45:9: warning: unused type alias 'T_partials_array' [-Wunused-local-typedef]
      using T_partials_array = Eigen::Array<T_partials_return, Eigen::Dynamic, 1>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:111:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/frechet_cdf.hpp:32:9: warning: unused type alias 'T_partials_array' [-Wunused-local-typedef]
      using T_partials_array = Eigen::Array<T_partials_return, Eigen::Dynamic, 1>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:196:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/multi_normal_cholesky_log.hpp:6:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/multi_normal_cholesky_lpdf.hpp:56:9: warning: unused type alias 'vector_partials_t' [-Wunused-local-typedef]
      using vector_partials_t = Eigen::Matrix<T_partials_return, Eigen::Dynamic, 1>;
            ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/multi_normal_cholesky_lpdf.hpp:57:9: warning: unused type alias 'row_vector_partials_t' [-Wunused-local-typedef]
      using row_vector_partials_t
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:248:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/ordered_logistic_log.hpp:5:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/ordered_logistic_lpmf.hpp:79:9: warning: unused type alias 'T_partials_array' [-Wunused-local-typedef]
      using T_partials_array = Eigen::Array<T_partials_return, -1, 1>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:284:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/poisson_log_glm_log.hpp:5:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/poisson_log_glm_lpmf.hpp:62:9: warning: unused type alias 'T_alpha_val' [-Wunused-local-typedef]
      using T_alpha_val = typename std::conditional_t<
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:310:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/skew_double_exponential_lpdf.hpp:47:9: warning: unused type alias 'T_partials_array' [-Wunused-local-typedef]
      using T_partials_array = Eigen::Array<T_partials_return, Eigen::Dynamic, 1>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:17:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing.hpp:7:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/assign_varmat.hpp:329:9: warning: unused type alias 'pair_type' [-Wunused-local-typedef]
      using pair_type = std::pair<int, arena_vec>;
            ^


    [1A[0J[36mBuilding:[0m 12.9s
    [1A[0J[36mBuilding:[0m 13.0s
    [1A[0J[36mBuilding:[0m 13.1s
    [1A[0J[36mBuilding:[0m 13.2s
    [1A[0J[36mBuilding:[0m 13.3s
    [1A[0J[36mBuilding:[0m 13.4s
    [1A[0J[36mBuilding:[0m 13.6s
    [1A[0J[36mBuilding:[0m 13.7s
    [1A[0J[36mBuilding:[0m 13.8s
    [1A[0J[36mBuilding:[0m 13.9s
    [1A[0J[36mBuilding:[0m 14.0s
    [1A[0J[36mBuilding:[0m 14.1s
    [1A[0J[36mBuilding:[0m 14.2s
    [1A[0J[36mBuilding:[0m 14.3s
    [1A[0J[36mBuilding:[0m 14.4s
    [1A[0J[36mBuilding:[0m 14.5s
    [1A[0J[36mBuilding:[0m 14.6s
    [1A[0J[36mBuilding:[0m 14.7s
    [1A[0J[36mBuilding:[0m 14.8s
    [1A[0J[36mBuilding:[0m 14.9s
    [1A[0J[36mBuilding:[0m 15.0s
    [1A[0J[36mBuilding:[0m 15.1s
    [1A[0J[36mBuilding:[0m 15.2s
    [1A[0J[36mBuilding:[0m 15.3s
    [1A[0J[36mBuilding:[0m 15.4s
    [1A[0J[36mBuilding:[0m 15.5s
    [1A[0J[36mBuilding:[0m 15.6s
    [1A[0J[36mBuilding:[0m 15.7s
    [1A[0J[36mBuilding:[0m 15.8s
    [1A[0J[36mBuilding:[0m 16.0s
    [1A[0J[36mBuilding:[0m 16.1s
    [1A[0J[36mBuilding:[0m 16.2s
    [1A[0J[36mBuilding:[0m 16.3s
    [1A[0J[36mBuilding:[0m 16.4s
    [1A[0J[36mBuilding:[0m 16.5s
    [1A[0J[36mBuilding:[0m 16.6s
    [1A[0J[36mBuilding:[0m 16.7s
    [1A[0J[36mBuilding:[0m 16.8s
    [1A[0J[36mBuilding:[0m 16.9s
    [1A[0J[36mBuilding:[0m 17.0s
    [1A[0J[36mBuilding:[0m 17.1s
    [1A[0J[36mBuilding:[0m 17.2s
    [1A[0J[36mBuilding:[0m 17.3s
    [1A[0J[36mBuilding:[0m 17.4s
    [1A[0J[36mBuilding:[0m 17.5s
    [1A[0J[36mBuilding:[0m 17.7s
    [1A[0J[36mBuilding:[0m 17.8s
    [1A[0J[36mBuilding:[0m 17.9s
    [1A[0J[36mBuilding:[0m 18.0s
    [1A[0J[36mBuilding:[0m 18.1s
    [1A[0J[36mBuilding:[0m 18.2s
    [1A[0J[36mBuilding:[0m 18.3s
    [1A[0J[36mBuilding:[0m 18.4s
    [1A[0J[36mBuilding:[0m 18.5s
    [1A[0J[36mBuilding:[0m 18.6s
    [1A[0J[36mBuilding:[0m 18.7s
    [1A[0J[36mBuilding:[0m 18.8s
    [1A[0J[36mBuilding:[0m 18.9s
    [1A[0J[36mBuilding:[0m 19.0s
    [1A[0J[36mBuilding:[0m 19.1s
    [1A[0J[36mBuilding:[0m 19.2s
    [1A[0J[36mBuilding:[0m 19.3s
    [1A[0J[36mBuilding:[0m 19.4s
    [1A[0J[36mBuilding:[0m 19.5s
    [1A[0J[36mBuilding:[0m 19.7s
    [1A[0J[36mBuilding:[0m 19.8s
    [1A[0J[36mBuilding:[0m 19.9s
    [1A[0J[36mBuilding:[0m 20.0s
    [1A[0J[36mBuilding:[0m 20.1s
    [1A[0J[36mBuilding:[0m 20.2s
    [1A[0J[36mBuilding:[0m 20.3s


    81 warnings generated.


    [1A[0J

    ld: warning: direct access in function 'long double boost::math::detail::igamma_temme_large<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >(long double, long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> const&, boost::integral_constant<int, 64> const*)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'boost::math::tools::promote_args<long double, float, float, float, float, float>::type boost::math::log1pmx<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >(long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> const&)::function' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::igamma_temme_large<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >(long double, long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> const&, boost::integral_constant<int, 64> const*)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'boost::math::tools::promote_args<long double, float, float, float, float, float>::type boost::math::log1pmx<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >(long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> const&)::function' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::igamma_temme_large<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >(long double, long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> const&, boost::integral_constant<int, 64> const*)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'boost::math::tools::promote_args<long double, float, float, float, float, float>::type boost::math::log1pmx<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >(long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> const&)::function' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'void boost::throw_exception<boost::math::evaluation_error>(boost::math::evaluation_error const&)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'typeinfo for boost::wrapexcept<boost::math::evaluation_error>' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'boost::wrapexcept<boost::math::evaluation_error>::rethrow() const' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'typeinfo for boost::wrapexcept<boost::math::evaluation_error>' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_j0<long double>(long double)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_j0<long double>(long double)::Q2' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_j0<long double>(long double)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_j0<long double>(long double)::P2' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_j0<long double>(long double)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_j0<long double>(long double)::Q1' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_j0<long double>(long double)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_j0<long double>(long double)::P1' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_j1<long double>(long double)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_j1<long double>(long double)::Q2' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_j1<long double>(long double)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_j1<long double>(long double)::P2' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_j1<long double>(long double)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_j1<long double>(long double)::Q1' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_j1<long double>(long double)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_j1<long double>(long double)::P1' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_j1<long double>(long double)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_j1<long double>(long double)::QS' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_j1<long double>(long double)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_j1<long double>(long double)::PS' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_j1<long double>(long double)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_j1<long double>(long double)::QC' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_j1<long double>(long double)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_j1<long double>(long double)::PC' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_k0_imp<long double>(long double const&, boost::integral_constant<int, 64> const&)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_k0_imp<long double>(long double const&, boost::integral_constant<int, 64> const&)::Q' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_k0_imp<long double>(long double const&, boost::integral_constant<int, 64> const&)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_k0_imp<long double>(long double const&, boost::integral_constant<int, 64> const&)::P' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_k0_imp<long double>(long double const&, boost::integral_constant<int, 64> const&)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_k0_imp<long double>(long double const&, boost::integral_constant<int, 64> const&)::Q' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_k0_imp<long double>(long double const&, boost::integral_constant<int, 64> const&)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_k0_imp<long double>(long double const&, boost::integral_constant<int, 64> const&)::P' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.31' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::digamma_initializer<double, boost::math::policies::policy<boost::math::policies::pole_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::overflow_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.31' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::digamma_initializer<double, boost::math::policies::policy<boost::math::policies::pole_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::overflow_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.32' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::lanczos::lanczos_initializer<boost::math::lanczos::lanczos17m64, long double>::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.32' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::lanczos::lanczos17m64::lanczos_sum_expG_scaled<long double>(long double const&)::denom' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.32' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::lanczos::lanczos17m64::lanczos_sum_expG_scaled<long double>(long double const&)::num' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.32' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::lanczos::lanczos17m64::lanczos_sum<long double>(long double const&)::denom' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.32' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::lanczos::lanczos17m64::lanczos_sum<long double>(long double const&)::num' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.32' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::lanczos::lanczos_initializer<boost::math::lanczos::lanczos17m64, long double>::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.33' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::igamma_initializer<double, boost::math::policies::policy<boost::math::policies::pole_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::overflow_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.33' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::igamma_initializer<double, boost::math::policies::policy<boost::math::policies::pole_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::overflow_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.34' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::lgamma_initializer<double, boost::math::policies::policy<boost::math::policies::pole_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::overflow_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.34' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::lgamma_initializer<double, boost::math::policies::policy<boost::math::policies::pole_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::overflow_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.35' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::erf_initializer<double, boost::math::policies::policy<boost::math::policies::pole_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::overflow_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy>, boost::integral_constant<int, 53> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.35' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::erf_initializer<double, boost::math::policies::policy<boost::math::policies::pole_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::overflow_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy>, boost::integral_constant<int, 53> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.36' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::expm1_initializer<double, boost::math::policies::policy<boost::math::policies::pole_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::overflow_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy>, boost::integral_constant<int, 53> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.36' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::expm1_initializer<double, boost::math::policies::policy<boost::math::policies::pole_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::overflow_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy>, boost::integral_constant<int, 53> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.37' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::igamma_initializer<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.37' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::igamma_initializer<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.38' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::lgamma_initializer<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.38' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::lgamma_initializer<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.39' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::erf_initializer<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy>, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.39' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::erf_initializer<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy>, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.40' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::expm1_initializer<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy>, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.40' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::expm1_initializer<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy>, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.41' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_i0_initializer<long double, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.41' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_i0_initializer<long double, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.42' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_i1_initializer<long double, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.42' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_i1_initializer<long double, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.43' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::owens_t_initializer<double, boost::math::policies::policy<boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy>, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.43' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::owens_t_initializer<double, boost::math::policies::policy<boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy>, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.44' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_j0_initializer<long double>::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.44' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_j0_initializer<long double>::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.45' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_j1_initializer<long double>::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.45' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_j1_initializer<long double>::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.46' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_y0_initializer<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.46' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_y0_initializer<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.47' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_y1_initializer<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.47' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_y1<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >(long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> const&)::Q1' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.47' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_y1<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >(long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> const&)::P1' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.47' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_y1_initializer<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.48' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_k0_initializer<long double, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.48' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_k0_initializer<long double, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.49' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_k1_initializer<long double, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.49' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_k1_initializer<long double, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/vloc2crl/model_vloc2crl.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    [32mBuilding:[0m 20.4s, done.
    [36mSampling:[0m   0%
    [1A[0J[36mSampling:[0m   0% (1/12000)
    [1A[0J[36mSampling:[0m   0% (2/12000)
    [1A[0J[36mSampling:[0m   1% (102/12000)
    [1A[0J[36mSampling:[0m   1% (103/12000)
    [1A[0J[36mSampling:[0m   2% (202/12000)
    [1A[0J[36mSampling:[0m   3% (301/12000)
    [1A[0J[36mSampling:[0m   3% (401/12000)
    [1A[0J[36mSampling:[0m   4% (500/12000)
    [1A[0J[36mSampling:[0m   5% (600/12000)
    [1A[0J[36mSampling:[0m   6% (700/12000)
    [1A[0J[36mSampling:[0m   7% (800/12000)
    [1A[0J[36mSampling:[0m   8% (900/12000)
    [1A[0J[36mSampling:[0m   8% (1000/12000)
    [1A[0J[36mSampling:[0m   9% (1100/12000)
    [1A[0J[36mSampling:[0m  10% (1200/12000)
    [1A[0J[36mSampling:[0m  11% (1300/12000)
    [1A[0J[36mSampling:[0m  12% (1400/12000)
    [1A[0J[36mSampling:[0m  12% (1500/12000)
    [1A[0J[36mSampling:[0m  13% (1600/12000)
    [1A[0J[36mSampling:[0m  14% (1700/12000)
    [1A[0J[36mSampling:[0m  15% (1800/12000)
    [1A[0J[36mSampling:[0m  16% (1900/12000)
    [1A[0J[36mSampling:[0m  17% (2000/12000)
    [1A[0J[36mSampling:[0m  18% (2100/12000)
    [1A[0J[36mSampling:[0m  18% (2200/12000)
    [1A[0J[36mSampling:[0m  19% (2300/12000)
    [1A[0J[36mSampling:[0m  20% (2400/12000)
    [1A[0J[36mSampling:[0m  21% (2500/12000)
    [1A[0J[36mSampling:[0m  22% (2600/12000)
    [1A[0J[36mSampling:[0m  22% (2700/12000)
    [1A[0J[36mSampling:[0m  23% (2800/12000)
    [1A[0J[36mSampling:[0m  24% (2900/12000)
    [1A[0J[36mSampling:[0m  25% (3000/12000)
    [1A[0J[36mSampling:[0m  26% (3100/12000)
    [1A[0J[36mSampling:[0m  27% (3200/12000)
    [1A[0J[36mSampling:[0m  44% (5300/12000)
    [1A[0J[36mSampling:[0m  62% (7500/12000)
    [1A[0J[36mSampling:[0m  81% (9700/12000)
    [1A[0J[36mSampling:[0m 100% (12000/12000)
    [1A[0J[32mSampling:[0m 100% (12000/12000), done.
    [36mMessages received during sampling:[0m
      Gradient evaluation took 3.9e-05 seconds
      1000 transitions using 10 leapfrog steps per transition would take 0.39 seconds.
      Adjust your expectations accordingly!
      Gradient evaluation took 3.6e-05 seconds
      1000 transitions using 10 leapfrog steps per transition would take 0.36 seconds.
      Adjust your expectations accordingly!
      Gradient evaluation took 3.2e-05 seconds
      1000 transitions using 10 leapfrog steps per transition would take 0.32 seconds.
      Adjust your expectations accordingly!
      Gradient evaluation took 3.2e-05 seconds
      1000 transitions using 10 leapfrog steps per transition would take 0.32 seconds.
      Adjust your expectations accordingly!
      Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
      Exception: dirichlet_lpdf: prior sample sizes[3] is 0, but must be positive! (in '/var/folders/2f/s3znbbc13nn29v0hcm82ntgr0000gp/T/httpstan_n1ws_8vu/model_vloc2crl.stan', line 13, column 4 to column 32)
      If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
      but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.


The parameter summaries are below. As expected with a uniform prior, the range of alpha is quite large. In turn, that allows each probability parameter to sample across the full range of its support $[0, 1]$. 


```python
{
    "tags": [
        "remove_input"
    ]
}

fit.to_frame().describe().round(decimals = 2).T
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/2f/s3znbbc13nn29v0hcm82ntgr0000gp/T/ipykernel_93601/1276650614.py in <module>
          4     ]
          5 }
    ----> 6 fit.to_frame().describe().round(decimals = 2).T
    

    NameError: name 'fit' is not defined



```python
{
    "tags": [
        "remove_cell"
    ]
}

def select_probs(prev_state, theta):
    if prev_state == 1:
        probs = [theta[0], theta[1], theta[2], theta[3]]
    elif prev_state == 2:
        probs = [theta[4], theta[5], theta[6], theta[7]]
    elif prev_state == 3:
        probs = [theta[8], theta[9], theta[10], theta[11]]
    elif prev_state == 4:
        probs = [theta[12], theta[13], theta[14], theta[15]]

    return probs

def draw_series(n_sim, theta):
    y = [1]
    
    for i in range(1, n_sim):
        probs = select_probs(y[i-1], theta)
        rng = np.random.default_rng()
        draw = rng.multinomial(1, probs)
        result = np.where(draw == np.amax(draw))
        y.append(int(result[0]) + 1)
        
    return y


def label_states(state_number):
    if state_number == 1:
        state = "awake"
    elif state_number == 2:
        state = "light"
    elif state_number == 3:
        state = "REM"
    elif state_number == 4:
        state = "deep"
    return state
```


```python
{
    "tags": [
        "remove_input"
    ]
}
parameters = fit.to_frame()
parameters = parameters[['theta.1.1', 'theta.1.2', 'theta.1.3', 'theta.1.4'
                             , 'theta.2.1', 'theta.2.2', 'theta.2.3', 'theta.2.4'
                             , 'theta.3.1', 'theta.3.2', 'theta.3.3', 'theta.3.4'
                             , 'theta.4.1', 'theta.4.2', 'theta.4.3', 'theta.4.4']]

grouped = parameters.groupby('draws')

prior_predictions = []

for name, group in grouped:
    prior_predictions.append(draw_series(n_sim = 100
                , theta = [float(group['theta.1.1']), float(group['theta.1.2']), float(group['theta.1.3']), float(group['theta.1.4'])
                , float(group['theta.2.1']), float(group['theta.2.2']), float(group['theta.2.3']), float(group['theta.2.4'])
                , float(group['theta.3.1']), float(group['theta.3.2']), float(group['theta.3.3']), float(group['theta.3.4'])
                , float(group['theta.4.1']), float(group['theta.4.2']), float(group['theta.4.3']), float(group['theta.4.4'])]))

prior_predictions = pd.DataFrame(prior_predictions)
prior_predictions = prior_predictions.reset_index()

prior_predictions_long = prior_predictions.melt(id_vars = "index")
prior_predictions_long['time'] = prior_predictions_long['variable']*5
prior_predictions_long['y_value'] = np.where((prior_predictions_long['value'] == 1) | (prior_predictions_long['value'] == 2), 0.1, -0.1)
prior_predictions_long['group_y_value'] = prior_predictions_long['y_value'] + prior_predictions_long['index']

prior_predictions_long['sleep_state'] = prior_predictions_long['value'].apply(label_states)
```

We can generate some examples of a night's sleep, based solely on what is predicted from our prior distributions (that is, not conditioned on the actual data). In the plot below there a 9 draws, where each colour represents a different sleep state. 


```python
{
    "tags": [
        "remove_input"
    ]
}
g = sns.relplot(data = prior_predictions_long[prior_predictions_long['index'] < 10], x = 'time', y = 'group_y_value', hue = 'sleep_state')
g.set(ylabel = "Draw", title = "Prior Predictive Checks")
```




    <seaborn.axisgrid.FacetGrid at 0x7fd5f7420e20>




    
![png](Basic_Markov_Model_files/Basic_Markov_Model_12_1.png)
    


Whilst some draws show switching between states, other draws are fixed within once state for an entire evening. We can clearly see the influence of our prior choices here, since these draws will correspond to large values of $\theta_k$ for a specific state $k$. Since each row of the transition matrix must sum to one, if one element is too close to 1, then the rest must be close to 0. 
Our belief for this model was that having a uniform prior over the probability parameter would be uninformative and therefore appropriate, however this does not take into account the relationship betweewn one parameter and the others within a row of the transition matrix. This is discussed in the [STAN documentation](https://mc-stan.org/docs/2_27/functions-reference/dirichlet-distribution.html). To quote directly 
'As the size of the simplex grows, the marginal draws become more and more concentrated below (not around) $1/K$. When one component of the simplex is large, the others must all be relatively small to compensate. For example, in a uniform distribution on  
10-simplexes, the probability that a component is greater than the mean of $1/10$ is only 39%. Most of the posterior marginal probability mass for each component is in the interval $(0, 0.1)$.'
My interpretation of what this all implies is that a sensible prior for $\alpha_k$ should constrain it to be small (close to 0). This will mean that element of the simplex for $\theta_k$ will be concentrated below $1/4$. The code for this is below - I'm using a $Cauchy (0, 1)$ distribution for each $\alpha_k$ so that the fat tails still allow for some chance of an element of each transition matrix row being larger than the others (for example, if I'm awake it's more likely that I'm awake in the next 5 minutes as well). 


```python
{
    "tags": [
        "remove_output"
    ]
}
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

    [36mBuilding:[0m 0.1s
    [1A[0J[36mBuilding:[0m 0.2s
    [1A[0J[36mBuilding:[0m 0.3s
    [1A[0J[36mBuilding:[0m 0.4s
    [1A[0J[36mBuilding:[0m 0.5s
    [1A[0J[36mBuilding:[0m 0.6s
    [1A[0J[36mBuilding:[0m 0.7s
    [1A[0J[36mBuilding:[0m 0.8s
    [1A[0J[36mBuilding:[0m 0.9s
    [1A[0J[36mBuilding:[0m 1.1s
    [1A[0J[36mBuilding:[0m 1.2s
    [1A[0J[36mBuilding:[0m 1.3s
    [1A[0J[36mBuilding:[0m 1.4s
    [1A[0J[36mBuilding:[0m 1.5s
    [1A[0J[36mBuilding:[0m 1.6s
    [1A[0J[36mBuilding:[0m 1.7s
    [1A[0J[36mBuilding:[0m 1.8s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:8:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/chainable_object.hpp:6:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/typedefs.hpp:7:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/Eigen_NumTraits.hpp:5:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/core.hpp:4:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/core/init_threadpool_tbb.hpp:104:14: warning: variable 'tbb_max_threads' is used uninitialized whenever 'if' condition is false [-Wsometimes-uninitialized]
      } else if (n_threads == -1) {
                 ^~~~~~~~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/core/init_threadpool_tbb.hpp:112:53: note: uninitialized use occurs here
          tbb::global_control::max_allowed_parallelism, tbb_max_threads);
                                                        ^~~~~~~~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/core/init_threadpool_tbb.hpp:104:10: note: remove the 'if' if its condition is always true
      } else if (n_threads == -1) {
             ^~~~~~~~~~~~~~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/core/init_threadpool_tbb.hpp:99:22: note: initialize the variable 'tbb_max_threads' to silence this warning
      int tbb_max_threads;
                         ^
                          = 0
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:8:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core.hpp:28:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/operator_addition.hpp:6:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_matching_dims.hpp:33:8: warning: unused variable 'error' [-Wunused-variable]
      bool error = false;
           ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_matching_dims.hpp:57:23: warning: comparison of integers of different signs: 'int' and 'std::__1::vector<int, std::__1::allocator<int> >::size_type' (aka 'unsigned long') [-Wsign-compare]
        for (int i = 0; i < y1_d.size(); i++) {
                        ~ ^ ~~~~~~~~~~~


    [1A[0J[36mBuilding:[0m 1.9s
    [1A[0J[36mBuilding:[0m 2.0s
    [1A[0J[36mBuilding:[0m 2.1s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:8:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core.hpp:53:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/profiling.hpp:9:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err.hpp:20:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_less.hpp:63:16: warning: unused variable 'n' [-Wunused-variable]
      Eigen::Index n = 0;
                   ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_less.hpp:98:16: warning: unused variable 'n' [-Wunused-variable]
      Eigen::Index n = 0;
                   ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_less.hpp:133:16: warning: unused variable 'n' [-Wunused-variable]
      Eigen::Index n = 0;
                   ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:8:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core.hpp:53:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/profiling.hpp:9:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err.hpp:50:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/hmm_check.hpp:33:7: warning: unused variable 'n_transitions' [-Wunused-variable]
      int n_transitions = log_omegas.cols() - 1;
          ^


    [1A[0J[36mBuilding:[0m 2.2s
    [1A[0J[36mBuilding:[0m 2.3s
    [1A[0J[36mBuilding:[0m 2.4s
    [1A[0J[36mBuilding:[0m 2.5s
    [1A[0J[36mBuilding:[0m 2.6s
    [1A[0J[36mBuilding:[0m 2.7s
    [1A[0J[36mBuilding:[0m 2.8s
    [1A[0J[36mBuilding:[0m 2.9s
    [1A[0J[36mBuilding:[0m 3.1s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:7:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/fun.hpp:118:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/fun/gp_matern52_cov.hpp:304:10: warning: unused variable 'neg_root_5' [-Wunused-variable]
      double neg_root_5 = -root_5;
             ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:7:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/fun.hpp:183:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/fun/log_mix.hpp:86:13: warning: unused variable 'N' [-Wunused-variable]
      const int N = stan::math::size(theta);
                ^


    [1A[0J[36mBuilding:[0m 3.2s
    [1A[0J[36mBuilding:[0m 3.3s
    [1A[0J[36mBuilding:[0m 3.4s
    [1A[0J[36mBuilding:[0m 3.5s
    [1A[0J[36mBuilding:[0m 3.6s
    [1A[0J[36mBuilding:[0m 3.7s
    [1A[0J[36mBuilding:[0m 3.8s
    [1A[0J[36mBuilding:[0m 3.9s
    [1A[0J[36mBuilding:[0m 4.0s
    [1A[0J[36mBuilding:[0m 4.1s
    [1A[0J[36mBuilding:[0m 4.2s
    [1A[0J[36mBuilding:[0m 4.3s
    [1A[0J[36mBuilding:[0m 4.4s
    [1A[0J[36mBuilding:[0m 4.5s
    [1A[0J[36mBuilding:[0m 4.6s
    [1A[0J[36mBuilding:[0m 4.7s
    [1A[0J[36mBuilding:[0m 4.9s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:28:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/beta.hpp:70:32: warning: lambda capture 'b' is not used [-Wunused-lambda-capture]
                               [a, b, digamma_ab](auto& vi) mutable {
                                 ~~^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/beta.hpp:96:39: warning: lambda capture 'a' is not used [-Wunused-lambda-capture]
      return make_callback_var(beta_val, [a, b, digamma_ab](auto& vi) mutable {
                                          ^~


    [1A[0J[36mBuilding:[0m 5.0s
    [1A[0J[36mBuilding:[0m 5.1s
    [1A[0J[36mBuilding:[0m 5.2s
    [1A[0J[36mBuilding:[0m 5.3s
    [1A[0J[36mBuilding:[0m 5.4s
    [1A[0J[36mBuilding:[0m 5.5s
    [1A[0J[36mBuilding:[0m 5.6s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:123:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/matrix_power.hpp:52:24: warning: comparison of integers of different signs: 'size_t' (aka 'unsigned long') and 'const int' [-Wsign-compare]
      for (size_t i = 2; i <= n; ++i) {
                         ~ ^  ~
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:134:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/ordered_constrain.hpp:40:30: warning: comparison of integers of different signs: 'Eigen::Index' (aka 'long') and 'size_t' (aka 'unsigned long') [-Wsign-compare]
      for (Eigen::Index n = 1; n < N; ++n) {
                               ~ ^ ~
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:137:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/positive_ordered_constrain.hpp:40:21: warning: comparison of integers of different signs: 'int' and 'size_t' (aka 'unsigned long') [-Wsign-compare]
      for (int n = 1; n < N; ++n) {
                      ~ ^ ~
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:153:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/simplex_constrain.hpp:40:30: warning: comparison of integers of different signs: 'Eigen::Index' (aka 'long') and 'size_t' (aka 'unsigned long') [-Wsign-compare]
      for (Eigen::Index k = 0; k < N; ++k) {
                               ~ ^ ~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/simplex_constrain.hpp:94:30: warning: comparison of integers of different signs: 'Eigen::Index' (aka 'long') and 'size_t' (aka 'unsigned long') [-Wsign-compare]
      for (Eigen::Index k = 0; k < N; ++k) {
                               ~ ^ ~


    [1A[0J[36mBuilding:[0m 5.7s
    [1A[0J[36mBuilding:[0m 5.8s
    [1A[0J[36mBuilding:[0m 5.9s
    [1A[0J[36mBuilding:[0m 6.0s
    [1A[0J[36mBuilding:[0m 6.1s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:11:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor.hpp:27:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor/operands_and_partials.hpp:29:1: warning: 'ops_partials_edge' defined as a class template here but previously declared as a struct template; this is valid, but may result in linker errors under the Microsoft C++ ABI [-Wmismatched-tags]
    class ops_partials_edge<double, var> {
    ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/functor/operands_and_partials.hpp:21:1: note: did you mean class here?
    struct ops_partials_edge;
    ^~~~~~
    class
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:11:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor.hpp:27:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor/operands_and_partials.hpp:177:1: warning: 'ops_partials_edge' defined as a class template here but previously declared as a struct template; this is valid, but may result in linker errors under the Microsoft C++ ABI [-Wmismatched-tags]
    class ops_partials_edge<double, std::vector<var>> {
    ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/functor/operands_and_partials.hpp:21:1: note: did you mean class here?
    struct ops_partials_edge;
    ^~~~~~
    class
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:11:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor.hpp:27:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor/operands_and_partials.hpp:199:1: warning: 'ops_partials_edge' defined as a class template here but previously declared as a struct template; this is valid, but may result in linker errors under the Microsoft C++ ABI [-Wmismatched-tags]
    class ops_partials_edge<double, Op, require_eigen_st<is_var, Op>> {
    ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/functor/operands_and_partials.hpp:21:1: note: did you mean class here?
    struct ops_partials_edge;
    ^~~~~~
    class
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:11:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor.hpp:27:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor/operands_and_partials.hpp:219:1: warning: 'ops_partials_edge' defined as a class template here but previously declared as a struct template; this is valid, but may result in linker errors under the Microsoft C++ ABI [-Wmismatched-tags]
    class ops_partials_edge<double, var_value<Op>, require_eigen_t<Op>> {
    ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/functor/operands_and_partials.hpp:21:1: note: did you mean class here?
    struct ops_partials_edge;
    ^~~~~~
    class
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:11:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor.hpp:27:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor/operands_and_partials.hpp:245:1: warning: 'ops_partials_edge' defined as a class template here but previously declared as a struct template; this is valid, but may result in linker errors under the Microsoft C++ ABI [-Wmismatched-tags]
    class ops_partials_edge<double, std::vector<Eigen::Matrix<var, R, C>>> {
    ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/functor/operands_and_partials.hpp:21:1: note: did you mean class here?
    struct ops_partials_edge;
    ^~~~~~
    class
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:11:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor.hpp:27:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor/operands_and_partials.hpp:274:1: warning: 'ops_partials_edge' defined as a class template here but previously declared as a struct template; this is valid, but may result in linker errors under the Microsoft C++ ABI [-Wmismatched-tags]
    class ops_partials_edge<double, std::vector<std::vector<var>>> {
    ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/functor/operands_and_partials.hpp:21:1: note: did you mean class here?
    struct ops_partials_edge;
    ^~~~~~
    class
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:11:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor.hpp:27:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor/operands_and_partials.hpp:300:1: warning: 'ops_partials_edge' defined as a class template here but previously declared as a struct template; this is valid, but may result in linker errors under the Microsoft C++ ABI [-Wmismatched-tags]
    class ops_partials_edge<double, std::vector<var_value<Op>>,
    ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/functor/operands_and_partials.hpp:21:1: note: did you mean class here?
    struct ops_partials_edge;
    ^~~~~~
    class
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:11:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor.hpp:29:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor/finite_diff_hessian_auto.hpp:61:24: warning: comparison of integers of different signs: 'size_t' (aka 'unsigned long') and 'int' [-Wsign-compare]
      for (size_t i = 0; i < d; ++i) {
                         ~ ^ ~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/functor/finite_diff_hessian_auto.hpp:69:24: warning: comparison of integers of different signs: 'size_t' (aka 'unsigned long') and 'int' [-Wsign-compare]
      for (size_t i = 0; i < d; ++i) {
                         ~ ^ ~
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:87:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/double_exponential_cdf.hpp:77:10: warning: unused variable 'N' [-Wunused-variable]
      size_t N = max_size(y, mu, sigma);
             ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:128:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/gaussian_dlm_obs_rng.hpp:41:21: warning: comparison of integers of different signs: 'int' and 'size_t' (aka 'unsigned long') [-Wsign-compare]
      for (int i = 0; i < M; i++) {
                      ~ ^ ~


    [1A[0J[36mBuilding:[0m 6.2s
    [1A[0J[36mBuilding:[0m 6.3s


    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/gaussian_dlm_obs_rng.hpp:98:7: warning: unused variable 'n' [-Wunused-variable]
      int n = G.rows();  // number of states
          ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:139:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/hmm_marginal.hpp:26:13: warning: unused variable 'n_states' [-Wunused-variable]
      const int n_states = omegas.rows();
                ^


    [1A[0J[36mBuilding:[0m 6.4s
    [1A[0J[36mBuilding:[0m 6.5s
    [1A[0J[36mBuilding:[0m 6.6s
    [1A[0J[36mBuilding:[0m 6.7s
    [1A[0J[36mBuilding:[0m 6.9s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:330:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/std_normal_rng.hpp:23:22: warning: unused variable 'function' [-Wunused-variable]
      static const char* function = "std_normal_rng";
                         ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:350:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/von_mises_cdf.hpp:72:10: warning: unused variable 'ck' [-Wunused-variable]
      double ck = 50;
             ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:8:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/io/deserializer.hpp:337:28: warning: comparison of integers of different signs: 'size_t' (aka 'unsigned long') and 'Eigen::Index' (aka 'long') [-Wsign-compare]
          for (size_t i = 0; i < m; ++i) {
                             ~ ^ ~


    [1A[0J[36mBuilding:[0m 7.0s
    [1A[0J[36mBuilding:[0m 7.1s
    [1A[0J[36mBuilding:[0m 7.2s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:17:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing.hpp:6:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/assign.hpp:268:21: warning: comparison of integers of different signs: 'int' and 'std::__1::vector<int, std::__1::allocator<int> >::size_type' (aka 'unsigned long') [-Wsign-compare]
      for (int i = 0; i < idx.ns_.size(); ++i) {
                      ~ ^ ~~~~~~~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/assign.hpp:531:21: warning: comparison of integers of different signs: 'int' and 'std::__1::vector<int, std::__1::allocator<int> >::size_type' (aka 'unsigned long') [-Wsign-compare]
      for (int i = 0; i < col_idx.ns_.size(); ++i) {
                      ~ ^ ~~~~~~~~~~~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/assign.hpp:630:21: warning: comparison of integers of different signs: 'int' and 'std::__1::vector<int, std::__1::allocator<int> >::size_type' (aka 'unsigned long') [-Wsign-compare]
      for (int j = 0; j < col_idx.ns_.size(); ++j) {
                      ~ ^ ~~~~~~~~~~~~~~~~~~
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:17:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing.hpp:8:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue.hpp:159:21: warning: comparison of integers of different signs: 'int' and 'std::__1::vector<int, std::__1::allocator<int> >::size_type' (aka 'unsigned long') [-Wsign-compare]
      for (int i = 0; i < idx.ns_.size(); ++i) {
                      ~ ^ ~~~~~~~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue.hpp:266:21: warning: comparison of integers of different signs: 'int' and 'std::__1::vector<int, std::__1::allocator<int> >::size_type' (aka 'unsigned long') [-Wsign-compare]
      for (int i = 0; i < idx.ns_.size(); ++i) {
                      ~ ^ ~~~~~~~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue.hpp:441:21: warning: comparison of integers of different signs: 'int' and 'std::__1::vector<int, std::__1::allocator<int> >::size_type' (aka 'unsigned long') [-Wsign-compare]
      for (int i = 0; i < col_idx.ns_.size(); ++i) {
                      ~ ^ ~~~~~~~~~~~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue.hpp:471:21: warning: comparison of integers of different signs: 'int' and 'std::__1::vector<int, std::__1::allocator<int> >::size_type' (aka 'unsigned long') [-Wsign-compare]
      for (int i = 0; i < row_idx.ns_.size(); ++i) {
                      ~ ^ ~~~~~~~~~~~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue.hpp:556:13: warning: unused variable 'cols' [-Wunused-variable]
      const int cols = rvalue_index_size(col_idx, x_ref.cols());
                ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue.hpp:558:21: warning: comparison of integers of different signs: 'int' and 'std::__1::vector<int, std::__1::allocator<int> >::size_type' (aka 'unsigned long') [-Wsign-compare]
      for (int j = 0; j < col_idx.ns_.size(); ++j) {
                      ~ ^ ~~~~~~~~~~~~~~~~~~
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:17:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing.hpp:9:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue_varmat.hpp:66:21: warning: comparison of integers of different signs: 'int' and 'const unsigned long' [-Wsign-compare]
      for (int i = 0; i < ret_size; ++i) {
                      ~ ^ ~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue_varmat.hpp:102:21: warning: comparison of integers of different signs: 'int' and 'const unsigned long' [-Wsign-compare]
      for (int i = 0; i < ret_rows; ++i) {
                      ~ ^ ~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue_varmat.hpp:142:21: warning: comparison of integers of different signs: 'int' and 'const unsigned long' [-Wsign-compare]
      for (int j = 0; j < ret_size; ++j) {
                      ~ ^ ~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue_varmat.hpp:183:21: warning: comparison of integers of different signs: 'int' and 'const unsigned long' [-Wsign-compare]
      for (int i = 0; i < ret_size; ++i) {
                      ~ ^ ~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue_varmat.hpp:222:22: warning: unused variable 'x_cols' [-Wunused-variable]
      const Eigen::Index x_cols = x.cols();
                         ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue_varmat.hpp:227:21: warning: comparison of integers of different signs: 'int' and 'const unsigned long' [-Wsign-compare]
      for (int i = 0; i < ret_rows; ++i) {
                      ~ ^ ~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue_varmat.hpp:231:21: warning: comparison of integers of different signs: 'int' and 'const unsigned long' [-Wsign-compare]
      for (int j = 0; j < ret_cols; ++j) {
                      ~ ^ ~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue_varmat.hpp:235:23: warning: comparison of integers of different signs: 'int' and 'const unsigned long' [-Wsign-compare]
        for (int i = 0; i < ret_rows; ++i) {
                        ~ ^ ~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/rvalue_varmat.hpp:278:21: warning: comparison of integers of different signs: 'int' and 'const unsigned long' [-Wsign-compare]
      for (int j = 0; j < ret_cols; ++j) {
                      ~ ^ ~~~~~~~~


    [1A[0J[36mBuilding:[0m 7.3s
    [1A[0J[36mBuilding:[0m 7.4s
    [1A[0J[36mBuilding:[0m 7.5s
    [1A[0J[36mBuilding:[0m 7.6s
    [1A[0J[36mBuilding:[0m 7.7s
    [1A[0J[36mBuilding:[0m 7.8s
    [1A[0J[36mBuilding:[0m 7.9s
    [1A[0J[36mBuilding:[0m 8.0s
    [1A[0J[36mBuilding:[0m 8.1s
    [1A[0J[36mBuilding:[0m 8.2s
    [1A[0J[36mBuilding:[0m 8.3s
    [1A[0J[36mBuilding:[0m 8.4s
    [1A[0J[36mBuilding:[0m 8.5s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:8:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core.hpp:29:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/operator_divide_equal.hpp:5:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/operator_division.hpp:15:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/operator_subtraction.hpp:84:21: warning: lambda capture 'b' is not used [-Wunused-lambda-capture]
          [avi = a.vi_, b](const auto& vi) mutable { avi->adj_ += vi.adj_; });
                        ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/operator_minus_equal.hpp:24:16: note: in instantiation of function template specialization 'stan::math::operator-<double, nullptr>' requested here
      vi_ = (*this - b).vi_;
                   ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/core/complex_base.hpp:136:9: note: in instantiation of member function 'stan::math::var_value<double, void>::operator-=' requested here
        re_ -= x;
            ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/core/operator_subtraction.hpp:24:5: note: in instantiation of function template specialization 'stan::math::complex_base<stan::math::var_value<double, void> >::operator-=<int>' requested here
      y -= rhs;
        ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/core/operator_subtraction.hpp:55:20: note: in instantiation of function template specialization 'stan::math::internal::complex_subtract<std::__1::complex<stan::math::var>, int>' requested here
      return internal::complex_subtract(x, y);
                       ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/fun/acosh.hpp:105:31: note: in instantiation of function template specialization 'stan::math::operator-<stan::math::var_value<double, void>, int>' requested here
      auto y = log(z + sqrt(z * z - 1));
                                  ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/acosh.hpp:94:32: note: in instantiation of function template specialization 'stan::math::internal::complex_acosh<stan::math::var_value<double, void> >' requested here
      return stan::math::internal::complex_acosh(z);
                                   ^


    [1A[0J[36mBuilding:[0m 8.6s
    [1A[0J[36mBuilding:[0m 8.7s
    [1A[0J[36mBuilding:[0m 8.8s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:8:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core.hpp:28:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/operator_addition.hpp:80:21: warning: lambda capture 'b' is not used [-Wunused-lambda-capture]
          [avi = a.vi_, b](const auto& vi) mutable { avi->adj_ += vi.adj_; });
                        ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/grad_inc_beta.hpp:45:43: note: in instantiation of function template specialization 'stan::math::operator+<int, nullptr>' requested here
        grad_2F1(dF1, dF2, a + b, var(1.0), a + 1, z);
                                              ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:8:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core.hpp:28:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/operator_addition.hpp:80:21: warning: lambda capture 'b' is not used [-Wunused-lambda-capture]
          [avi = a.vi_, b](const auto& vi) mutable { avi->adj_ += vi.adj_; });
                        ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/fun/trigamma.hpp:63:31: note: in instantiation of function template specialization 'stan::math::operator+<double, nullptr>' requested here
        value = -trigamma_impl(-x + 1.0) + square(pi() / sin(-pi() * x));
                                  ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/trigamma.hpp:23:44: note: in instantiation of function template specialization 'stan::math::trigamma_impl<stan::math::var_value<double, void> >' requested here
    inline var trigamma(const var& u) { return trigamma_impl(u); }
                                               ^


    [1A[0J[36mBuilding:[0m 8.9s
    [1A[0J[36mBuilding:[0m 9.0s
    [1A[0J[36mBuilding:[0m 9.2s
    [1A[0J[36mBuilding:[0m 9.3s
    [1A[0J[36mBuilding:[0m 9.4s
    [1A[0J[36mBuilding:[0m 9.5s
    [1A[0J[36mBuilding:[0m 9.6s
    [1A[0J[36mBuilding:[0m 9.7s
    [1A[0J[36mBuilding:[0m 9.8s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:8:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core.hpp:53:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/profiling.hpp:9:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_2F1_converges.hpp:5:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_not_nan.hpp:5:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/elementwise_check.hpp:153:24: warning: comparison of integers of different signs: 'size_t' (aka 'unsigned long') and 'Eigen::EigenBase::Index' (aka 'long') [-Wsign-compare]
      for (size_t i = 0; i < x.size(); i++) {
                         ~ ^ ~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_not_nan.hpp:28:3: note: in instantiation of function template specialization 'stan::math::elementwise_check<(lambda at /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_not_nan.hpp:28:21), Eigen::ArrayWrapper<const Eigen::Matrix<double, -1, 1, 0, -1, 1> >, nullptr, nullptr>' requested here
      elementwise_check([](double x) { return !std::isnan(x); }, function, name, y,
      ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/cauchy_lpdf.hpp:72:3: note: in instantiation of function template specialization 'stan::math::check_not_nan<Eigen::ArrayWrapper<const Eigen::Matrix<double, -1, 1, 0, -1, 1> > >' requested here
      check_not_nan(function, "Random variable", y_val);
      ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:161:24: note: in instantiation of function template specialization 'stan::math::cauchy_lpdf<false, Eigen::Matrix<double, -1, 1, 0, -1, 1>, int, int, nullptr>' requested here
            lp_accum__.add(cauchy_lpdf<propto__>(alpha, 0, 1));
                           ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:456:14: note: in instantiation of function template specialization 'model_hrgsmnes_namespace::model_hrgsmnes::log_prob_impl<false, false, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, nullptr, nullptr>' requested here
          return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
                 ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_base_crtp.hpp:90:50: note: in instantiation of function template specialization 'model_hrgsmnes_namespace::model_hrgsmnes::log_prob<false, false, double>' requested here
        return static_cast<const M*>(this)->template log_prob<false, false, double>(
                                                     ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:45:3: note: in instantiation of member function 'stan::model::model_base_crtp<model_hrgsmnes_namespace::model_hrgsmnes>::log_prob' requested here
      ~model_hrgsmnes() { }
      ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:8:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core.hpp:53:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/profiling.hpp:9:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_2F1_converges.hpp:5:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_not_nan.hpp:5:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/elementwise_check.hpp:153:24: warning: comparison of integers of different signs: 'size_t' (aka 'unsigned long') and 'Eigen::EigenBase::Index' (aka 'long') [-Wsign-compare]
      for (size_t i = 0; i < x.size(); i++) {
                         ~ ^ ~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_positive.hpp:29:3: note: in instantiation of function template specialization 'stan::math::elementwise_check<(lambda at /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_positive.hpp:29:21), Eigen::Matrix<double, -1, 1, 0, -1, 1>, nullptr, nullptr>' requested here
      elementwise_check([](double x) { return x > 0; }, function, name, y,
      ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/dirichlet_lpdf.hpp:78:5: note: in instantiation of function template specialization 'stan::math::check_positive<Eigen::Matrix<double, -1, 1, 0, -1, 1> >' requested here
        check_positive(function, "prior sample sizes", alpha_vec[t]);
        ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:166:13: note: in instantiation of function template specialization 'stan::math::dirichlet_lpdf<false, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, nullptr>' requested here
                dirichlet_lpdf<propto__>(rvalue(theta, "theta", index_uni(k)),
                ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:456:14: note: in instantiation of function template specialization 'model_hrgsmnes_namespace::model_hrgsmnes::log_prob_impl<false, false, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, nullptr, nullptr>' requested here
          return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
                 ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_base_crtp.hpp:90:50: note: in instantiation of function template specialization 'model_hrgsmnes_namespace::model_hrgsmnes::log_prob<false, false, double>' requested here
        return static_cast<const M*>(this)->template log_prob<false, false, double>(
                                                     ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:45:3: note: in instantiation of member function 'stan::model::model_base_crtp<model_hrgsmnes_namespace::model_hrgsmnes>::log_prob' requested here
      ~model_hrgsmnes() { }
      ^


    [1A[0J[36mBuilding:[0m 9.9s
    [1A[0J[36mBuilding:[0m 10.0s
    [1A[0J[36mBuilding:[0m 10.1s
    [1A[0J[36mBuilding:[0m 10.2s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:96:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/lb_constrain.hpp:214:35: warning: lambda capture 'lp' is not used [-Wunused-lambda-capture]
          reverse_pass_callback([ret, lp, arena_lb = var(lb)]() mutable {
                                      ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/io/deserializer.hpp:386:26: note: in instantiation of function template specialization 'stan::math::lb_constrain<Eigen::Map<Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, 0, Eigen::Stride<0, 0> >, int, nullptr, nullptr, nullptr>' requested here
          return stan::math::lb_constrain(this->read<Ret>(sizes...), lb, lp);
                             ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:150:29: note: in instantiation of function template specialization 'stan::io::deserializer<stan::math::var_value<double, void> >::read_constrain_lb<Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, false, int, stan::math::var_value<double, void>, int>' requested here
          alpha = in__.template read_constrain_lb<Eigen::Matrix<local_scalar_t__, -1, 1>, jacobian__>(
                                ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:456:14: note: in instantiation of function template specialization 'model_hrgsmnes_namespace::model_hrgsmnes::log_prob_impl<false, false, Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, nullptr, nullptr>' requested here
          return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
                 ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_base_crtp.hpp:95:50: note: in instantiation of function template specialization 'model_hrgsmnes_namespace::model_hrgsmnes::log_prob<false, false, stan::math::var_value<double, void> >' requested here
        return static_cast<const M*>(this)->template log_prob<false, false>(theta,
                                                     ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:45:3: note: in instantiation of member function 'stan::model::model_base_crtp<model_hrgsmnes_namespace::model_hrgsmnes>::log_prob' requested here
      ~model_hrgsmnes() { }
      ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:153:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/simplex_constrain.hpp:94:30: warning: comparison of integers of different signs: 'Eigen::Index' (aka 'long') and 'size_t' (aka 'unsigned long') [-Wsign-compare]
      for (Eigen::Index k = 0; k < N; ++k) {
                               ~ ^ ~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/io/deserializer.hpp:569:14: note: in instantiation of function template specialization 'stan::math::simplex_constrain<Eigen::Map<Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, 0, Eigen::Stride<0, 0> >, nullptr>' requested here
          return simplex_constrain(this->read<Ret>(size - 1), lp);
                 ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/io/deserializer.hpp:602:17: note: in instantiation of function template specialization 'stan::io::deserializer<stan::math::var_value<double, void> >::read_constrain_simplex<Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, false, stan::math::var_value<double, void>, nullptr>' requested here
              this->read_constrain_simplex<value_type_t<Ret>, Jacobian>(lp,
                    ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:157:29: note: in instantiation of function template specialization 'stan::io::deserializer<stan::math::var_value<double, void> >::read_constrain_simplex<std::__1::vector<Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, std::__1::allocator<Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1> > >, false, stan::math::var_value<double, void>, int, nullptr>' requested here
          theta = in__.template read_constrain_simplex<std::vector<Eigen::Matrix<local_scalar_t__, -1, 1>>, jacobian__>(
                                ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:456:14: note: in instantiation of function template specialization 'model_hrgsmnes_namespace::model_hrgsmnes::log_prob_impl<false, false, Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, nullptr, nullptr>' requested here
          return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
                 ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_base_crtp.hpp:95:50: note: in instantiation of function template specialization 'model_hrgsmnes_namespace::model_hrgsmnes::log_prob<false, false, stan::math::var_value<double, void> >' requested here
        return static_cast<const M*>(this)->template log_prob<false, false>(theta,
                                                     ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:45:3: note: in instantiation of member function 'stan::model::model_base_crtp<model_hrgsmnes_namespace::model_hrgsmnes>::log_prob' requested here
      ~model_hrgsmnes() { }
      ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:153:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/simplex_constrain.hpp:40:30: warning: comparison of integers of different signs: 'Eigen::Index' (aka 'long') and 'size_t' (aka 'unsigned long') [-Wsign-compare]
      for (Eigen::Index k = 0; k < N; ++k) {
                               ~ ^ ~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/io/deserializer.hpp:571:14: note: in instantiation of function template specialization 'stan::math::simplex_constrain<Eigen::Map<Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, 0, Eigen::Stride<0, 0> >, nullptr>' requested here
          return simplex_constrain(this->read<Ret>(size - 1));
                 ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/io/deserializer.hpp:602:17: note: in instantiation of function template specialization 'stan::io::deserializer<stan::math::var_value<double, void> >::read_constrain_simplex<Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, false, stan::math::var_value<double, void>, nullptr>' requested here
              this->read_constrain_simplex<value_type_t<Ret>, Jacobian>(lp,
                    ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:157:29: note: in instantiation of function template specialization 'stan::io::deserializer<stan::math::var_value<double, void> >::read_constrain_simplex<std::__1::vector<Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, std::__1::allocator<Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1> > >, false, stan::math::var_value<double, void>, int, nullptr>' requested here
          theta = in__.template read_constrain_simplex<std::vector<Eigen::Matrix<local_scalar_t__, -1, 1>>, jacobian__>(
                                ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:456:14: note: in instantiation of function template specialization 'model_hrgsmnes_namespace::model_hrgsmnes::log_prob_impl<false, false, Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, nullptr, nullptr>' requested here
          return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
                 ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_base_crtp.hpp:95:50: note: in instantiation of function template specialization 'model_hrgsmnes_namespace::model_hrgsmnes::log_prob<false, false, stan::math::var_value<double, void> >' requested here
        return static_cast<const M*>(this)->template log_prob<false, false>(theta,
                                                     ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:45:3: note: in instantiation of member function 'stan::model::model_base_crtp<model_hrgsmnes_namespace::model_hrgsmnes>::log_prob' requested here
      ~model_hrgsmnes() { }
      ^


    [1A[0J[36mBuilding:[0m 10.3s
    [1A[0J[36mBuilding:[0m 10.4s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:8:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core.hpp:53:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/profiling.hpp:9:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_2F1_converges.hpp:5:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_not_nan.hpp:5:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/elementwise_check.hpp:153:24: warning: comparison of integers of different signs: 'size_t' (aka 'unsigned long') and 'Eigen::EigenBase::Index' (aka 'long') [-Wsign-compare]
      for (size_t i = 0; i < x.size(); i++) {
                         ~ ^ ~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_not_nan.hpp:28:3: note: in instantiation of function template specialization 'stan::math::elementwise_check<(lambda at /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_not_nan.hpp:28:21), Eigen::Array<double, -1, 1, 0, -1, 1>, nullptr, nullptr>' requested here
      elementwise_check([](double x) { return !std::isnan(x); }, function, name, y,
      ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/cauchy_lpdf.hpp:72:3: note: in instantiation of function template specialization 'stan::math::check_not_nan<Eigen::Array<double, -1, 1, 0, -1, 1> >' requested here
      check_not_nan(function, "Random variable", y_val);
      ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:161:24: note: in instantiation of function template specialization 'stan::math::cauchy_lpdf<false, Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, int, int, nullptr>' requested here
            lp_accum__.add(cauchy_lpdf<propto__>(alpha, 0, 1));
                           ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:456:14: note: in instantiation of function template specialization 'model_hrgsmnes_namespace::model_hrgsmnes::log_prob_impl<false, false, Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, nullptr, nullptr>' requested here
          return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
                 ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_base_crtp.hpp:95:50: note: in instantiation of function template specialization 'model_hrgsmnes_namespace::model_hrgsmnes::log_prob<false, false, stan::math::var_value<double, void> >' requested here
        return static_cast<const M*>(this)->template log_prob<false, false>(theta,
                                                     ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:45:3: note: in instantiation of member function 'stan::model::model_base_crtp<model_hrgsmnes_namespace::model_hrgsmnes>::log_prob' requested here
      ~model_hrgsmnes() { }
      ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:8:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core.hpp:53:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/profiling.hpp:9:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_2F1_converges.hpp:5:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_not_nan.hpp:5:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/elementwise_check.hpp:153:24: warning: comparison of integers of different signs: 'size_t' (aka 'unsigned long') and 'Eigen::EigenBase::Index' (aka 'long') [-Wsign-compare]
      for (size_t i = 0; i < x.size(); i++) {
                         ~ ^ ~~~~~~~~
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_positive.hpp:29:3: note: in instantiation of function template specialization 'stan::math::elementwise_check<(lambda at /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_positive.hpp:29:21), Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, nullptr, nullptr>' requested here
      elementwise_check([](double x) { return x > 0; }, function, name, y,
      ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/dirichlet_lpdf.hpp:78:5: note: in instantiation of function template specialization 'stan::math::check_positive<Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1> >' requested here
        check_positive(function, "prior sample sizes", alpha_vec[t]);
        ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:166:13: note: in instantiation of function template specialization 'stan::math::dirichlet_lpdf<false, Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, nullptr>' requested here
                dirichlet_lpdf<propto__>(rvalue(theta, "theta", index_uni(k)),
                ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:456:14: note: in instantiation of function template specialization 'model_hrgsmnes_namespace::model_hrgsmnes::log_prob_impl<false, false, Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, nullptr, nullptr>' requested here
          return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
                 ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_base_crtp.hpp:95:50: note: in instantiation of function template specialization 'model_hrgsmnes_namespace::model_hrgsmnes::log_prob<false, false, stan::math::var_value<double, void> >' requested here
        return static_cast<const M*>(this)->template log_prob<false, false>(theta,
                                                     ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:45:3: note: in instantiation of member function 'stan::model::model_base_crtp<model_hrgsmnes_namespace::model_hrgsmnes>::log_prob' requested here
      ~model_hrgsmnes() { }
      ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:8:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core.hpp:29:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/operator_divide_equal.hpp:5:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/operator_division.hpp:15:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/operator_subtraction.hpp:104:21: warning: lambda capture 'a' is not used [-Wunused-lambda-capture]
          [bvi = b.vi_, a](const auto& vi) mutable { bvi->adj_ -= vi.adj_; });
                        ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/err/check_simplex.hpp:43:18: note: in instantiation of function template specialization 'stan::math::operator-<double, nullptr>' requested here
      if (!(fabs(1.0 - theta_ref.sum()) <= CONSTRAINT_TOLERANCE)) {
                     ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/dirichlet_lpdf.hpp:79:5: note: in instantiation of function template specialization 'stan::math::check_simplex<Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, nullptr>' requested here
        check_simplex(function, "probabilities", theta_vec[t]);
        ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:166:13: note: in instantiation of function template specialization 'stan::math::dirichlet_lpdf<false, Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, nullptr>' requested here
                dirichlet_lpdf<propto__>(rvalue(theta, "theta", index_uni(k)),
                ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:456:14: note: in instantiation of function template specialization 'model_hrgsmnes_namespace::model_hrgsmnes::log_prob_impl<false, false, Eigen::Matrix<stan::math::var_value<double, void>, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, nullptr, nullptr>' requested here
          return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
                 ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_base_crtp.hpp:95:50: note: in instantiation of function template specialization 'model_hrgsmnes_namespace::model_hrgsmnes::log_prob<false, false, stan::math::var_value<double, void> >' requested here
        return static_cast<const M*>(this)->template log_prob<false, false>(theta,
                                                     ^
    /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:45:3: note: in instantiation of member function 'stan::model::model_base_crtp<model_hrgsmnes_namespace::model_hrgsmnes>::log_prob' requested here
      ~model_hrgsmnes() { }
      ^


    [1A[0J[36mBuilding:[0m 10.5s
    [1A[0J[36mBuilding:[0m 10.6s
    [1A[0J[36mBuilding:[0m 10.7s


    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:8:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core.hpp:58:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/core/set_zero_all_adjoints.hpp:14:13: warning: 'static' function 'set_zero_all_adjoints' declared in header file should be declared 'static inline' [-Wunneeded-internal-declaration]
    static void set_zero_all_adjoints() {
                ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:7:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/fun.hpp:110:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/fun/generalized_inverse.hpp:34:9: warning: unused type alias 'value_t' [-Wunused-local-typedef]
      using value_t = value_type_t<EigMat>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:7:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/fun.hpp:321:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/fun/tail.hpp:42:9: warning: unused type alias 'idx_t' [-Wunused-local-typedef]
      using idx_t = index_type_t<std::vector<T>>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:32:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/cholesky_decompose.hpp:83:11: warning: unused type alias 'Block_' [-Wunused-local-typedef]
        using Block_ = Eigen::Block<Eigen::MatrixXd>;
              ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:34:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/cholesky_factor_constrain.hpp:32:9: warning: unused type alias 'T_scalar' [-Wunused-local-typedef]
      using T_scalar = value_type_t<T>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:74:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/generalized_inverse.hpp:64:9: warning: unused type alias 'value_t' [-Wunused-local-typedef]
      using value_t = value_type_t<VarMat>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun.hpp:121:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/lub_constrain.hpp:485:9: warning: unused type alias 'plain_x_array' [-Wunused-local-typedef]
      using plain_x_array = plain_type_t<decltype(arena_x_val.array())>;
            ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev/fun/lub_constrain.hpp:565:9: warning: unused type alias 'plain_x_array' [-Wunused-local-typedef]
      using plain_x_array = plain_type_t<decltype(arena_x_val.array())>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:10:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/bernoulli_logit_glm_log.hpp:5:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/bernoulli_logit_glm_lpmf.hpp:59:9: warning: unused type alias 'T_y_val' [-Wunused-local-typedef]
      using T_y_val =
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:31:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/beta_log.hpp:5:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/beta_lpdf.hpp:50:9: warning: unused type alias 'T_partials_matrix' [-Wunused-local-typedef]
      using T_partials_matrix = Eigen::Matrix<T_partials_return, Eigen::Dynamic, 1>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:37:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/beta_proportion_log.hpp:5:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/beta_proportion_lpdf.hpp:56:9: warning: unused type alias 'T_partials_array' [-Wunused-local-typedef]
      using T_partials_array = Eigen::Array<T_partials_return, Eigen::Dynamic, 1>;
            ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/beta_proportion_lpdf.hpp:55:9: warning: unused type alias 'T_partials_return_kappa' [-Wunused-local-typedef]
      using T_partials_return_kappa = return_type_t<T_prec>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:63:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/cauchy_log.hpp:5:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/cauchy_lpdf.hpp:46:9: warning: unused type alias 'T_partials_array' [-Wunused-local-typedef]
      using T_partials_array = Eigen::Array<T_partials_return, Eigen::Dynamic, 1>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:74:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/dirichlet_log.hpp:5:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/dirichlet_lpdf.hpp:62:9: warning: unused type alias 'T_partials_vec' [-Wunused-local-typedef]
      using T_partials_vec = typename Eigen::Matrix<T_partials_return, -1, 1>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:91:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/double_exponential_log.hpp:5:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/double_exponential_lpdf.hpp:45:9: warning: unused type alias 'T_partials_array' [-Wunused-local-typedef]
      using T_partials_array = Eigen::Array<T_partials_return, Eigen::Dynamic, 1>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:111:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/frechet_cdf.hpp:32:9: warning: unused type alias 'T_partials_array' [-Wunused-local-typedef]
      using T_partials_array = Eigen::Array<T_partials_return, Eigen::Dynamic, 1>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:196:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/multi_normal_cholesky_log.hpp:6:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/multi_normal_cholesky_lpdf.hpp:56:9: warning: unused type alias 'vector_partials_t' [-Wunused-local-typedef]
      using vector_partials_t = Eigen::Matrix<T_partials_return, Eigen::Dynamic, 1>;
            ^
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/multi_normal_cholesky_lpdf.hpp:57:9: warning: unused type alias 'row_vector_partials_t' [-Wunused-local-typedef]
      using row_vector_partials_t
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:248:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/ordered_logistic_log.hpp:5:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/ordered_logistic_lpmf.hpp:79:9: warning: unused type alias 'T_partials_array' [-Wunused-local-typedef]
      using T_partials_array = Eigen::Array<T_partials_return, -1, 1>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:284:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/poisson_log_glm_log.hpp:5:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/poisson_log_glm_lpmf.hpp:62:9: warning: unused type alias 'T_alpha_val' [-Wunused-local-typedef]
      using T_alpha_val = typename std::conditional_t<
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:4:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math.hpp:19:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/rev.hpp:13:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim.hpp:16:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob.hpp:310:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/math/prim/prob/skew_double_exponential_lpdf.hpp:47:9: warning: unused type alias 'T_partials_array' [-Wunused-local-typedef]
      using T_partials_array = Eigen::Array<T_partials_return, Eigen::Dynamic, 1>;
            ^
    In file included from /Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.cpp:2:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/model_header.hpp:17:
    In file included from /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing.hpp:7:
    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/include/stan/model/indexing/assign_varmat.hpp:329:9: warning: unused type alias 'pair_type' [-Wunused-local-typedef]
      using pair_type = std::pair<int, arena_vec>;
            ^


    [1A[0J[36mBuilding:[0m 10.9s
    [1A[0J[36mBuilding:[0m 11.0s
    [1A[0J[36mBuilding:[0m 11.1s
    [1A[0J[36mBuilding:[0m 11.2s
    [1A[0J[36mBuilding:[0m 11.3s
    [1A[0J[36mBuilding:[0m 11.4s
    [1A[0J[36mBuilding:[0m 11.5s
    [1A[0J[36mBuilding:[0m 11.6s
    [1A[0J[36mBuilding:[0m 11.7s
    [1A[0J[36mBuilding:[0m 11.8s
    [1A[0J[36mBuilding:[0m 11.9s
    [1A[0J[36mBuilding:[0m 12.0s
    [1A[0J[36mBuilding:[0m 12.1s
    [1A[0J[36mBuilding:[0m 12.2s
    [1A[0J[36mBuilding:[0m 12.3s
    [1A[0J[36mBuilding:[0m 12.4s
    [1A[0J[36mBuilding:[0m 12.5s
    [1A[0J[36mBuilding:[0m 12.6s
    [1A[0J[36mBuilding:[0m 12.7s
    [1A[0J[36mBuilding:[0m 12.8s
    [1A[0J[36mBuilding:[0m 12.9s
    [1A[0J[36mBuilding:[0m 13.1s
    [1A[0J[36mBuilding:[0m 13.2s
    [1A[0J[36mBuilding:[0m 13.3s
    [1A[0J[36mBuilding:[0m 13.4s
    [1A[0J[36mBuilding:[0m 13.5s
    [1A[0J[36mBuilding:[0m 13.6s
    [1A[0J[36mBuilding:[0m 13.7s
    [1A[0J[36mBuilding:[0m 13.8s
    [1A[0J[36mBuilding:[0m 13.9s
    [1A[0J[36mBuilding:[0m 14.0s
    [1A[0J[36mBuilding:[0m 14.1s
    [1A[0J[36mBuilding:[0m 14.2s
    [1A[0J[36mBuilding:[0m 14.3s
    [1A[0J[36mBuilding:[0m 14.4s
    [1A[0J[36mBuilding:[0m 14.5s
    [1A[0J[36mBuilding:[0m 14.6s
    [1A[0J[36mBuilding:[0m 14.7s
    [1A[0J[36mBuilding:[0m 14.8s
    [1A[0J[36mBuilding:[0m 15.0s
    [1A[0J[36mBuilding:[0m 15.1s
    [1A[0J[36mBuilding:[0m 15.2s
    [1A[0J[36mBuilding:[0m 15.3s
    [1A[0J[36mBuilding:[0m 15.4s
    [1A[0J[36mBuilding:[0m 15.5s
    [1A[0J[36mBuilding:[0m 15.6s
    [1A[0J[36mBuilding:[0m 15.7s
    [1A[0J[36mBuilding:[0m 15.8s
    [1A[0J[36mBuilding:[0m 15.9s
    [1A[0J[36mBuilding:[0m 16.0s
    [1A[0J[36mBuilding:[0m 16.1s
    [1A[0J[36mBuilding:[0m 16.2s
    [1A[0J[36mBuilding:[0m 16.3s
    [1A[0J[36mBuilding:[0m 16.4s
    [1A[0J[36mBuilding:[0m 16.5s
    [1A[0J[36mBuilding:[0m 16.7s
    [1A[0J[36mBuilding:[0m 16.8s
    [1A[0J[36mBuilding:[0m 16.9s
    [1A[0J[36mBuilding:[0m 17.0s
    [1A[0J[36mBuilding:[0m 17.1s
    [1A[0J[36mBuilding:[0m 17.2s
    [1A[0J[36mBuilding:[0m 17.3s
    [1A[0J[36mBuilding:[0m 17.4s
    [1A[0J[36mBuilding:[0m 17.5s
    [1A[0J[36mBuilding:[0m 17.6s
    [1A[0J[36mBuilding:[0m 17.7s
    [1A[0J[36mBuilding:[0m 17.8s
    [1A[0J[36mBuilding:[0m 17.9s
    [1A[0J[36mBuilding:[0m 18.0s
    [1A[0J[36mBuilding:[0m 18.1s
    [1A[0J[36mBuilding:[0m 18.2s
    [1A[0J[36mBuilding:[0m 18.3s
    [1A[0J[36mBuilding:[0m 18.4s
    [1A[0J[36mBuilding:[0m 18.6s
    [1A[0J[36mBuilding:[0m 18.7s
    [1A[0J[36mBuilding:[0m 18.8s
    [1A[0J[36mBuilding:[0m 18.9s
    [1A[0J[36mBuilding:[0m 19.0s
    [1A[0J[36mBuilding:[0m 19.1s
    [1A[0J[36mBuilding:[0m 19.2s
    [1A[0J[36mBuilding:[0m 19.3s
    [1A[0J[36mBuilding:[0m 19.4s
    [1A[0J[36mBuilding:[0m 19.5s
    [1A[0J[36mBuilding:[0m 19.6s
    [1A[0J

    83 warnings generated.
    ld: warning: direct access in function 'long double boost::math::detail::igamma_temme_large<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >(long double, long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> const&, boost::integral_constant<int, 64> const*)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'boost::math::tools::promote_args<long double, float, float, float, float, float>::type boost::math::log1pmx<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >(long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> const&)::function' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::igamma_temme_large<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >(long double, long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> const&, boost::integral_constant<int, 64> const*)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'boost::math::tools::promote_args<long double, float, float, float, float, float>::type boost::math::log1pmx<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >(long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> const&)::function' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::igamma_temme_large<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >(long double, long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> const&, boost::integral_constant<int, 64> const*)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'boost::math::tools::promote_args<long double, float, float, float, float, float>::type boost::math::log1pmx<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >(long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> const&)::function' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'void boost::throw_exception<boost::math::evaluation_error>(boost::math::evaluation_error const&)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'typeinfo for boost::wrapexcept<boost::math::evaluation_error>' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'boost::wrapexcept<boost::math::evaluation_error>::rethrow() const' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'typeinfo for boost::wrapexcept<boost::math::evaluation_error>' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_j0<long double>(long double)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_j0<long double>(long double)::Q2' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_j0<long double>(long double)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_j0<long double>(long double)::P2' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_j0<long double>(long double)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_j0<long double>(long double)::Q1' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_j0<long double>(long double)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_j0<long double>(long double)::P1' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_j1<long double>(long double)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_j1<long double>(long double)::Q2' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_j1<long double>(long double)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_j1<long double>(long double)::P2' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_j1<long double>(long double)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_j1<long double>(long double)::Q1' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_j1<long double>(long double)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_j1<long double>(long double)::P1' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_j1<long double>(long double)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_j1<long double>(long double)::QS' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_j1<long double>(long double)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_j1<long double>(long double)::PS' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_j1<long double>(long double)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_j1<long double>(long double)::QC' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_j1<long double>(long double)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_j1<long double>(long double)::PC' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_k0_imp<long double>(long double const&, boost::integral_constant<int, 64> const&)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_k0_imp<long double>(long double const&, boost::integral_constant<int, 64> const&)::Q' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_k0_imp<long double>(long double const&, boost::integral_constant<int, 64> const&)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_k0_imp<long double>(long double const&, boost::integral_constant<int, 64> const&)::P' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_k0_imp<long double>(long double const&, boost::integral_constant<int, 64> const&)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_k0_imp<long double>(long double const&, boost::integral_constant<int, 64> const&)::Q' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function 'long double boost::math::detail::bessel_k0_imp<long double>(long double const&, boost::integral_constant<int, 64> const&)' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_k0_imp<long double>(long double const&, boost::integral_constant<int, 64> const&)::P' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.31' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::digamma_initializer<double, boost::math::policies::policy<boost::math::policies::pole_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::overflow_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.31' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::digamma_initializer<double, boost::math::policies::policy<boost::math::policies::pole_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::overflow_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.32' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::lanczos::lanczos_initializer<boost::math::lanczos::lanczos17m64, long double>::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.32' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::lanczos::lanczos17m64::lanczos_sum_expG_scaled<long double>(long double const&)::denom' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.32' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::lanczos::lanczos17m64::lanczos_sum_expG_scaled<long double>(long double const&)::num' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.32' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::lanczos::lanczos17m64::lanczos_sum<long double>(long double const&)::denom' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.32' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::lanczos::lanczos17m64::lanczos_sum<long double>(long double const&)::num' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.32' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::lanczos::lanczos_initializer<boost::math::lanczos::lanczos17m64, long double>::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.33' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::igamma_initializer<double, boost::math::policies::policy<boost::math::policies::pole_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::overflow_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.33' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::igamma_initializer<double, boost::math::policies::policy<boost::math::policies::pole_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::overflow_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.34' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::lgamma_initializer<double, boost::math::policies::policy<boost::math::policies::pole_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::overflow_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.34' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::lgamma_initializer<double, boost::math::policies::policy<boost::math::policies::pole_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::overflow_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.35' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::erf_initializer<double, boost::math::policies::policy<boost::math::policies::pole_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::overflow_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy>, boost::integral_constant<int, 53> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.35' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::erf_initializer<double, boost::math::policies::policy<boost::math::policies::pole_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::overflow_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy>, boost::integral_constant<int, 53> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.36' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::expm1_initializer<double, boost::math::policies::policy<boost::math::policies::pole_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::overflow_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy>, boost::integral_constant<int, 53> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.36' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::expm1_initializer<double, boost::math::policies::policy<boost::math::policies::pole_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::overflow_error<(boost::math::policies::error_policy_type)1>, boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy>, boost::integral_constant<int, 53> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.37' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::igamma_initializer<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.37' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::igamma_initializer<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.38' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::lgamma_initializer<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.38' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::lgamma_initializer<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.39' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::erf_initializer<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy>, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.39' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::erf_initializer<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy>, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.40' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::expm1_initializer<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy>, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.40' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::expm1_initializer<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy>, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.41' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_i0_initializer<long double, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.41' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_i0_initializer<long double, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.42' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_i1_initializer<long double, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.42' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_i1_initializer<long double, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.43' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::owens_t_initializer<double, boost::math::policies::policy<boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy>, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.43' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::owens_t_initializer<double, boost::math::policies::policy<boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy>, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.44' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_j0_initializer<long double>::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.44' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_j0_initializer<long double>::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.45' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_j1_initializer<long double>::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.45' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_j1_initializer<long double>::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.46' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_y0_initializer<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.46' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_y0_initializer<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.47' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_y1_initializer<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.47' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_y1<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >(long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> const&)::Q1' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.47' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'long double boost::math::detail::bessel_y1<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >(long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> const&)::P1' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.47' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_y1_initializer<long double, boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::promote_double<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.48' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_k0_initializer<long double, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.48' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_k0_initializer<long double, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.49' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_k1_initializer<long double, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in function '___cxx_global_var_init.49' from file '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/httpstan/stan_services.o' to global weak symbol 'guard variable for boost::math::detail::bessel_k1_initializer<long double, boost::integral_constant<int, 64> >::initializer' from file 'build/temp.macosx-10.9-x86_64-3.9/Users/torricallan1/Library/Caches/httpstan/4.5.0/models/hrgsmnes/model_hrgsmnes.o' means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    [32mBuilding:[0m 19.6s, done.
    [36mMessages from [0m[36;1mstanc[0m[36m:[0m
    Warning: The parameter alpha has 2 priors.
    [36mSampling:[0m   0%
    [1A[0J[36mSampling:[0m   4% (500/12000)
    [1A[0J[36mSampling:[0m   8% (1000/12000)
    [1A[0J[36mSampling:[0m  33% (4000/12000)
    [1A[0J[36mSampling:[0m  58% (7000/12000)
    [1A[0J[36mSampling:[0m  79% (9500/12000)
    [1A[0J[36mSampling:[0m 100% (12000/12000)
    [1A[0J[32mSampling:[0m 100% (12000/12000), done.
    [36mMessages received during sampling:[0m
      Gradient evaluation took 2.6e-05 seconds
      1000 transitions using 10 leapfrog steps per transition would take 0.26 seconds.
      Adjust your expectations accordingly!
      Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
      Exception: dirichlet_lpdf: prior sample sizes[1] is 0, but must be positive! (in '/var/folders/2f/s3znbbc13nn29v0hcm82ntgr0000gp/T/httpstan_v12sjura/model_hrgsmnes.stan', line 14, column 4 to column 32)
      If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
      but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
      Gradient evaluation took 2.3e-05 seconds
      1000 transitions using 10 leapfrog steps per transition would take 0.23 seconds.
      Adjust your expectations accordingly!
      Gradient evaluation took 2.1e-05 seconds
      1000 transitions using 10 leapfrog steps per transition would take 0.21 seconds.
      Adjust your expectations accordingly!
      Gradient evaluation took 2.1e-05 seconds
      1000 transitions using 10 leapfrog steps per transition would take 0.21 seconds.
      Adjust your expectations accordingly!
      Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
      Exception: dirichlet_lpdf: prior sample sizes[3] is 0, but must be positive! (in '/var/folders/2f/s3znbbc13nn29v0hcm82ntgr0000gp/T/httpstan_v12sjura/model_hrgsmnes.stan', line 14, column 4 to column 32)
      If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
      but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.


This behaviour is what we observe in the summary of each element of $\theta$. The mean of each is close to $1/4$ even though the range covers $[0,1]$. 


```python
{
    "tags": [
        "remove_input"
    ]
}
fit.to_frame().describe().T
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




```python
{
    "tags": [
        "remove_cell"
    ]
}
parameters = fit.to_frame()
parameters = parameters[['theta.1.1', 'theta.1.2', 'theta.1.3', 'theta.1.4'
                             , 'theta.2.1', 'theta.2.2', 'theta.2.3', 'theta.2.4'
                             , 'theta.3.1', 'theta.3.2', 'theta.3.3', 'theta.3.4'
                             , 'theta.4.1', 'theta.4.2', 'theta.4.3', 'theta.4.4']]

grouped = parameters.groupby('draws')

prior_predictions = []

for name, group in grouped:
    prior_predictions.append(draw_series(n_sim = 100
                , theta = [float(group['theta.1.1']), float(group['theta.1.2']), float(group['theta.1.3']), float(group['theta.1.4'])
                , float(group['theta.2.1']), float(group['theta.2.2']), float(group['theta.2.3']), float(group['theta.2.4'])
                , float(group['theta.3.1']), float(group['theta.3.2']), float(group['theta.3.3']), float(group['theta.3.4'])
                , float(group['theta.4.1']), float(group['theta.4.2']), float(group['theta.4.3']), float(group['theta.4.4'])]))


prior_predictions = pd.DataFrame(prior_predictions)
prior_predictions = prior_predictions.reset_index()

prior_predictions_long = prior_predictions.melt(id_vars = "index")
prior_predictions_long['time'] = prior_predictions_long['variable']*5
prior_predictions_long['y_value'] = np.where((prior_predictions_long['value'] == 1) | (prior_predictions_long['value'] == 2), 0.1, -0.1)
prior_predictions_long['group_y_value'] = prior_predictions_long['y_value'] + prior_predictions_long['index']
prior_predictions_long['sleep_state'] = prior_predictions_long['value'].apply(label_states)
```

Again we can simulate some predicted nights of sleep. Although draws are still mostly stuck in a single state, other draws look closer to a realistic evening, where multiple states are found across the night. 


```python
{
    "tags": [
        "remove_input"
    ]
}
g = sns.relplot(data = prior_predictions_long[prior_predictions_long['index'] < 10], x = 'time', y = 'group_y_value', hue = 'sleep_state')
g.set(ylabel = "Draw", title = "Prior Predictive Checks")
```




    <seaborn.axisgrid.FacetGrid at 0x7fd5fca49d60>




    
![png](Basic_Markov_Model_files/Basic_Markov_Model_19_1.png)
    


We can now fit the model to some observed data. This is acheived by including a sampling statement in STAN `y[i] ~ categorical(theta[y[i - 1]]);`.


```python
{
    "tags": [
        "remove_output"
    ]
}
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

    [32mBuilding:[0m found in cache, done.
    [36mSampling:[0m   0%
    [1A[0J[36mSampling:[0m   0% (1/12000)
    [1A[0J[36mSampling:[0m   0% (2/12000)
    [1A[0J[36mSampling:[0m   0% (3/12000)
    [1A[0J[36mSampling:[0m   0% (4/12000)
    [1A[0J[36mSampling:[0m   2% (203/12000)
    [1A[0J[36mSampling:[0m   3% (402/12000)
    [1A[0J[36mSampling:[0m   6% (701/12000)
    [1A[0J[36mSampling:[0m   8% (1000/12000)
    [1A[0J[36mSampling:[0m  11% (1300/12000)
    [1A[0J[36mSampling:[0m  13% (1600/12000)
    [1A[0J[36mSampling:[0m  16% (1900/12000)
    [1A[0J[36mSampling:[0m  18% (2200/12000)
    [1A[0J[36mSampling:[0m  22% (2600/12000)
    [1A[0J[36mSampling:[0m  24% (2900/12000)
    [1A[0J[36mSampling:[0m  44% (5300/12000)
    [1A[0J[36mSampling:[0m  64% (7700/12000)
    [1A[0J[36mSampling:[0m  82% (9800/12000)
    [1A[0J[36mSampling:[0m 100% (12000/12000)
    [1A[0J[32mSampling:[0m 100% (12000/12000), done.
    [36mMessages received during sampling:[0m
      Gradient evaluation took 0.000318 seconds
      1000 transitions using 10 leapfrog steps per transition would take 3.18 seconds.
      Adjust your expectations accordingly!
      Gradient evaluation took 0.000315 seconds
      1000 transitions using 10 leapfrog steps per transition would take 3.15 seconds.
      Adjust your expectations accordingly!
      Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
      Exception: dirichlet_lpdf: prior sample sizes[2] is 0, but must be positive! (in '/var/folders/2f/s3znbbc13nn29v0hcm82ntgr0000gp/T/httpstan_28gfj9c8/model_g7r3nudp.stan', line 15, column 4 to column 32)
      If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
      but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.
      Gradient evaluation took 0.000268 seconds
      1000 transitions using 10 leapfrog steps per transition would take 2.68 seconds.
      Adjust your expectations accordingly!
      Gradient evaluation took 0.000244 seconds
      1000 transitions using 10 leapfrog steps per transition would take 2.44 seconds.
      Adjust your expectations accordingly!
      Informational Message: The current Metropolis proposal is about to be rejected because of the following issue:
      Exception: dirichlet_lpdf: prior sample sizes[3] is 0, but must be positive! (in '/var/folders/2f/s3znbbc13nn29v0hcm82ntgr0000gp/T/httpstan_28gfj9c8/model_g7r3nudp.stan', line 15, column 4 to column 32)
      If this warning occurs sporadically, such as for highly constrained variable types like covariance matrices, then the sampler is fine,
      but if this warning occurs often then your model may be either severely ill-conditioned or misspecified.



```python
{
    "tags": [
        "remove_input",
    ]
}
fit.to_frame().describe().T
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


```python
{
    "tags": [
        "remove_cells"
    ]
}
parameters = fit.to_frame()
parameters = parameters[['theta.1.1', 'theta.1.2', 'theta.1.3', 'theta.1.4'
                             , 'theta.2.1', 'theta.2.2', 'theta.2.3', 'theta.2.4'
                             , 'theta.3.1', 'theta.3.2', 'theta.3.3', 'theta.3.4'
                             , 'theta.4.1', 'theta.4.2', 'theta.4.3', 'theta.4.4']]

grouped = parameters.groupby('draws')

posterior_predictions = []

for name, group in grouped:
    posterior_predictions.append(draw_series(n_sim = 100
                , theta = [float(group['theta.1.1']), float(group['theta.1.2']), float(group['theta.1.3']), float(group['theta.1.4'])
                , float(group['theta.2.1']), float(group['theta.2.2']), float(group['theta.2.3']), float(group['theta.2.4'])
                , float(group['theta.3.1']), float(group['theta.3.2']), float(group['theta.3.3']), float(group['theta.3.4'])
                , float(group['theta.4.1']), float(group['theta.4.2']), float(group['theta.4.3']), float(group['theta.4.4'])]))

posterior_predictions = pd.DataFrame(posterior_predictions)
posterior_predictions = posterior_predictions.reset_index()

posterior_predictions_long = posterior_predictions.melt(id_vars = "index")
posterior_predictions_long['time'] = posterior_predictions_long['variable']*5
posterior_predictions_long['y_value'] = np.where((posterior_predictions_long['value'] == 1) | (posterior_predictions_long['value'] == 2), 0.1, -0.1)
posterior_predictions_long['group_y_value'] = posterior_predictions_long['y_value'] + posterior_predictions_long['index']

posterior_predictions_long['sleep_state'] = posterior_predictions_long['value'].apply(label_states)
```

We can again sample some predicted nights of sleep from the model, and investigate the impact of conditioning on the data for our model. Each draw now shows a reasonable amount of mixing between states, which matches the observed behaviour we see in the data. 

The predictive checks also show a few areas in which the model could be improved. Some of the sleep stages (particularly REM and deep), last for much longer than is usually observed (the entire sleep cycle typically lasts 90 minutes, so each stage should be shorter than this). As well, deep sleep is more common earlier in the evening, whereas REM sleep is more common in the early morning after a few hours of sleep. This phenomena is not yet included in our model.


```python
{
    "tags": [
        "remove_input"
    ]
}
g = sns.relplot(data = posterior_predictions_long[posterior_predictions_long['index'] < 10], x = 'time', y = 'group_y_value', hue = 'sleep_state')
g.set(ylabel = "Draw", title = "Posterior Predictive Checks")
```




    <seaborn.axisgrid.FacetGrid at 0x13ca71a30>




    
![png](Basic_Markov_Model_files/Basic_Markov_Model_26_1.png)
    


A nice feature of a using a more appropriate model for the process we're interested in, is that it can still be used to produce simple summary stats (this is particularly easy in a Bayesian model, where a summary statistic is just a function of the posterior samples). 

Here we can produce the average amount of each sleep stage within a night of sleep. Below, we plot the entire posterior distribution of the amount of time spent in each state. The posterior distributions look reasonable - they imply the some nights will experience only a small amount of each stage, but that in other nights two hour is probable. The tails cover three hours of REM and deep sleep, which has never occured in my experience (although it has for other people). As we improve the model the coverage of the posterior for these summary values should improve as well.


```python
{
    "tags": [
        "remove_input"
    ]
}
posterior_predictions_sleep_time = posterior_predictions_long.groupby(['index', 'sleep_state'])['variable'].count()*5
posterior_predictions_sleep_time = posterior_predictions_sleep_time.reset_index()
posterior_predictions_sleep_time.groupby('sleep_state')['variable'].mean() * 1/60.0
```




    sleep_state
    REM      1.275542
    awake    1.643990
    deep     1.032833
    light    4.397354
    Name: variable, dtype: float64




```python
{
    "tags": [
        "remove_input"
    ]
}
g = sns.FacetGrid(posterior_predictions_sleep_time, row="sleep_state", hue="sleep_state", aspect=15, height=1.5)

# Draw the densities in a few steps
g.map(sns.kdeplot, "variable",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)

g.map(sns.kdeplot, "variable", clip_on=False, color="w", lw=2, bw_adjust=.5)
```




    <seaborn.axisgrid.FacetGrid at 0x142975a90>




    
![png](Basic_Markov_Model_files/Basic_Markov_Model_29_1.png)
    


In this post, we set up a basic markov model for sleep, investigated the impact of our prior choices and then fit the model to some data and interpreted the parameters. The markov model is nice for inference because the parameters are directly interpretable. 

In the next post, we will investigate how to include timing effects (time of night, and length of each sleep stage) into the model. 
