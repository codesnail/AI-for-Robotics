
# Probabilistic State Estimation with Bayesian Filters

In this article, I will demonstrate Bayesian filters in action. Bayesian filtering is a basic technique of Probabilistic State Estimation. The problem can be understood with an example. Consider a car driving on the road. It has a map of its surroundings, as well as a GPS. Assume that we are starting from our home. We have the exact coordinates of this initial location. As we drive off, we have two ways to know about our current location. Depending on which road we took, direction, turns, average speed of the car etc., we can calculate our position based on the map. This is a knowledge based estimate. Depending on the quality of input attributes and the quality of our knowledge model, we may get a more or less accurate estimate of our exact location, but you can imagine that there is a margin of error. We can also use the other source of information we have, the GPS, which gives us the coordinates of our location. The GPS also has a margin of error. The idea is to combine our knowledge based estimate and the GPS reading, to get a combined estimate of our current location that is more accurate than the individual estimates. Bayesian Filtering is one approach to get this estimate.

The general problem setting is as follows. We have an environment with a number of possible states, or even a continuous state space. We have a prior belief about the state of the environment, and about the dynamics (or motion model, or knowledge) of the environment, i.e., how the environment can move from one state to another. This may involve actions and their influence on the state. In addition, we have a way to sense the environment. All of these things (prior knowledge, influence of actions, and sensor readings) are non-deterministic. Implied here is the fact that we don't have absolute knowledge of the environment, or at least the attribute of the environment that we are interested in. So we have to infer or estimate it in the presence of this uncertainty.

There are many practical applications of this problem e.g. in Robotics, Finance, Healthcare, etc. In Robotics, take the example of a self-driving car. It gets information about its environment (e.g. other cars, lane lines and traffic signs) from its sensors, e.g. a camera or LIDAR. In order to drive safely, among other things, it needs to know the current position of other cars, as well as the expected future position of those cars. For example, if another car is changing lanes, where will it enter its own lane? Depending on that, it has to make a decision of whether to slow-down, keep its speed or take some other action. It makes an estimate of the future position of the car using a "motion model" of the other car, then updates its estimate based on its sensor reading. It does this in a continuous loop. This state estimate is then used to take an approrpiate action (e.g. apply brakes, honk, steer etc).

In healthcare, this can be applied to monitoring patient condition and administering treatments. An ICU patient's current condition may dictate probabilities of various outcomes based on prior knowledge, e.g. how likely is the patient to have a septic shock. Call it the Patient's Health Model. In addition, there are various sensor readings e.g. blood pressure, heart rate, glucose level etc., which are monitored regularly. An intelligent patient monitoring agent takes both of these into account and estimates the probability of septic shock, in order to administer a treatment or alert a human expert.

In finance, a stock trading agent may need to estimate the state of a stock, e.g. is the price about to move up or down? It may have prior knowledge based on a model based on fundamentals of the company, and it can track various technical indicators. Based on the market conditions, it updates its current estimate of the state of stock based on both to make a decision on whether to buy or sell.

To solidify the concepts, I will formulate the problem with a very simple example of weather prediction. This problem is taken from the book *Probabilistic Robotics* by Sebastian Thrun et. al. 

Consider that we have 3 possible weathers: sunny, cloudy, rainy. Today is the first day, and we know (for a fact) that it is cloudy.


```python
world = ['sunny', 'cloudy', 'rainy'] # possible world (or environment) states 
p = np.array([0., 1., 0.]) # initial state is cloudy
```

I modeled the initial state as a probability distribution, eventhough I stated above that we know for sure that it's cloudy. The reason for this will become apparent when we see how the state estimates work in Bayesian Filtering. For now, just understand that we modeled this certainty by having a probability of 1 for cloudy, and 0 for others.

Let's assume that we can't directly see or know about the weather (maybe after observing the weather on day 1, we went into an underground shelter without any link to the outside world). But we have a meteorologist with us and he knows a thing or two about weather transitions. In other words, if it was cloudy yesterday, what should I expect today? Or if it is rainy, would it be raining again tomorrow? Now he is not certain about this, so the information he actually has are the probabilities of transitioning from from one weather to another the next day. Since we have 3 weathers, we can represent these probabilities as a 3x3 matrix.


```python
# Weather transition probabilities
#
#              ...then  
#              today it 
#             should be:  Sunny | Cloudy | Rainy
# If yesterday
# was:
# ---------------------------------------------
# Sunny                    .8      .2      0.
# ---------------------------------------------
# Cloudy                   .4      .4     .2
# ---------------------------------------------
# Rainy                    .2      .6     .2
# ---------------------------------------------

pt = np.array(
      [[.8, .2, 0.], 
      [.4, .4, .2],
      [.2, .6, .2]])
```

Now let's say we slept in our shelter for the night and woke up the next day. What is the weather today? What can we infer based only on the initial state (yesterday's weather) and the weather transition probabilities. First, we know that various transitions are possible, so we're not going to get one value, rather a probability distribution over different weathers. Since we know for sure that the previous weather was cloudy, the probability distribution is simply:

$[p(x_t='sunny'|x_{t-1}='cloudy'), $
$ p(x_t='cloudy'|x_{t-1}='cloudy'), $
$ p(x_t='rainy'|x_{t-1}='cloudy')] $

Here $x_t$ represents the state at time t and $x_{t-1}$ the previous state. These conditional probabilities become:

$[p(x_t='sunny') * p(x_{t-1}='cloudy'), $
$ p(x_t='cloudy') * p(x_{t-1}='cloudy'), $
$ p(x_t='rainy') * p(x_{t-1}='cloudy')] $

Since we have $p(x_{t-1}='cloudy')=1$, this simply gives us the second row of the transition table above:

= \[0.4, 0.4, 0.2\]

However, to generalize further, we don't want to assume that we always know the previous state for sure. Therefore, we will use the probability distribution of the previous state.

$$
[p(x_t='sunny'|x_{t-1}='sunny')+p(x_t='sunny'|x_{t-1}='cloudy')+p(x_t='sunny'|x_{t-1}='rainy'),
$$

$[p(x_t='sunny'|x_{t-1}='sunny')+p(x_t='sunny'|x_{t-1}='cloudy')+p(x_t='sunny'|x_{t-1}='rainy'), $
$ p(x_t='cloudy'|x_{t-1}='sunny')+p(x_t='cloudy'|x_{t-1}='cloudy')+p(x_t='cloudy'|t_{k-1}='rainy'), $
$ p(x_t='rainy'|x_{t-1}='sunny')+p(x_t='rainy'|x_{t-1}='cloudy')+p(x_t='rainy'|x_{t-1}='rainy')] $

Here, $x_{k-1}$ is 0 for both 'sunny' and 'rainy', so we are left with the middle terms, which gives us the same result \[0.4, 0.4, 0.2\]

Putting this in code:


```python
import numpy as np
p = np.matmul(p,pt)
print(p)
```

    [0.4 0.4 0.2]
    

This basically multiplies the initial belief (p) element-wise with each column of the transition probabilities table, pt, and adds up the resulting rows (it will be a good exercise to work this through by hand and verify that it is correct). So from our isolated shelter, we now infer that there is a 40% chance that today is sunny, 40% chance of it being cloudy and only 20% chance that it is rainy.

We assumed earlier that we can't directly observe the weather. But let's suppose we have a sensor which can measure the three weather conditions.


```python
sensor = ['sunny', 'cloudy', 'rainy'] # possible sensor outputs
```

You may note that the sensor outputs are the same as the world states, but I modeled them separately just for clarity and flexibility. In real life, sensors can be noisy. For example, our sensor may be really accurate in measuring rainy weather, but if it measures sunny, there is a chance that it is actually cloudy. How do we account for the error margin for our sensor? In fact, the error can differ for each measurement. We can model this uncertainty also as a probability table.


```python
# Sensor correctness probabilities.
#
#            ...then  
#          it really 
#                is:    Sunny | Cloudy | Rainy
# If sensor
# reads:
# ---------------------------------------------
# Sunny                    .6      .4      0.
# ---------------------------------------------
# Cloudy                   .3      .7     .0
# ---------------------------------------------
# Rainy                    0.      0.     1.
# ---------------------------------------------

pSensor = np.array(
           [[.6, .4, 0.],
           [.3, .7, 0.],
           [0., 0., 1.]])
```

So, for example, if the sensor reads that it is sunny (1st row), there is a 60% chance that it is correct (it is actually sunny), but a 40% chance that it is actually cloudy. If it reads cloudy (2nd row), there is 70% chance that it is correct, but a 30% chance that it is actually sunny. Finally, it is very accurate in sensing rain, so if it reads rainy (3rd row), we can be certain it is actually raining.

With this setup, we're now ready to incorporate sensor measurements into our estimate of the weather. Let's say that we look at the sensor and it reads "cloudy". 


```python
measurement1 = "cloudy"
```

How does this affect our estimate? We multiply the current estimate with the probability distribution of the sensor reading "cloudy".


```python
i = sensor.index(measurement1) # Get the index of the measured weather (cloudy)
                               # for the probability distribution matrix of the sensor
p = p*pSensor[i] # i=1 because our measurement was "cloudy". So we will use the 2nd row of the pSensor matrix.
s = np.sum(p) # Normalizer
p = [x/s if(s>0) else 0 for x in p]
print(p)
```

    [0.3, 0.7, 0.0]
    

So we are now 70% sure that the weather is cloudy. This makes sense, because our prior estimate was equal for cloudy and sunny, but since our weather sensor also said cloudy, then the chance of it actually being cloudy went up.

That's it! This is Bayes Filter. What if we went into day 2 and wanted to estimate the weather again? We would use the last probability distribution above as the previous state and repeat the process. Now you know why we wanted to model the initial state as a probability distribution, eventhough we were sure it was cloudy.

Alright, so we basically performed two updates:

1. Knowledge update: Also called 'predict', where we predict the weather based on the transition knowledge and the prior weather. The transition knowledge can also be called the "motion model".

2. Sensor update: Also called just 'sense', where we incorporate the reading from the sensor into our prediction.

This is called the "update rule" in Bayesian Filters. Below, I encode it as an algorithm in pseudo-code:

```
function update(p(x{t-1}, z{t}):

   # p(x{t-1}): probability distribution of previous state
   # zt: current measurement
   
   for all x{t} do: # for all possible current states
       p(x{t}) = sum [ p(x{t}|x{t-1}) * p(x{t-1}) ]
       p(x{t}) = norm * [ p(z{t}|p(x{t} ]
```

Now let's put this all together in code.


```python
world = ['sunny', 'cloudy', 'rainy']
sensor = ['sunny', 'cloudy', 'rainy']

import numpy as np

p = np.array([0., 1., 0.]) #initial state

# Weather transition probabilities
pt = np.array(
     [[.8, .2, 0.], 
      [.4, .4, .2],
      [.2, .6, .2]])

# Sensor probabilities
pSensor = np.array(
          [[.6, .4, 0.], 
           [.3, .7, 0.],
           [0., 0., 1.]])

def predict(p):
    q = np.matmul(p,pt)
    return q

def sense(p, Z):
    i = sensor.index(Z)
    q = pSensor[i]*p
    s = np.sum(q)
    q = [x/s if(s>0) else 0 for x in q]
    return q


```

Let's use this code to estimate updates for 5 days.


```python
measurements = ['cloudy', 'sunny', 'cloudy', 'rainy', 'sunny'] # Example sensor readings for 5 days

for i in range(len(measurements)): # Each iteration represents a new day
    p=predict(p)
    p=sense(p, measurements[i])
    print(p)
```

    [0.3, 0.7, 0.0]
    [0.6964285714285715, 0.30357142857142855, 0.0]
    [0.5272895467160037, 0.47271045328399625, 0.0]
    [0.0, 0.0, 1.0]
    [0.3333333333333333, 0.6666666666666666, 0.0]
    

And there you have it, 5 weather predictions for the 5 days. We have just implemented a basic Bayes Filter to estimate weather!

In the next installment, I will show how to incorporate actions.
