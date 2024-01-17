import numpy as np
import pandas as pd
import scipy


def generate_data(n=5000, seed=0, beta1=1.05, alpha1=0.4, alpha2=0.3):
  """The generative model for our bloodpressure dataset example.

  Example to generate the data
  
  >>> data = generate_data(n=5000, seed=0, beta1=1.05, alpha1=0.4, alpha2=0.3)
  >>> data.to_csv('./data/bloodpressure_sodium_age.csv', index=False)
  """
  np.random.seed(seed)
  age = np.random.normal(65, 5, n)
  sodium = age / 18 + np.random.normal(size=n)
  bloodpressure = beta1 * sodium + 2 * age + np.random.normal(size=n)
  return pd.DataFrame({'bloodpressure': bloodpressure, 'sodium': sodium,
                        'age': age})


class FixableDataFrame(pd.DataFrame):
  """Helper class for manipulating generative models."""

  def __init__(self, *args, fixed={}, **kwargs):
      self.__dict__["__fixed_var_dictionary"] = fixed
      super().__init__(*args, **kwargs)

  def __setitem__(self, key, value):
      out = super().__setitem__(key, value)
      if isinstance(key, str) and key in self.__dict__["__fixed_var_dictionary"]:
          out = super().__setitem__(key, self.__dict__["__fixed_var_dictionary"][key])
      return out


# generate the data
def generator(n, fixed={}, seed=0):
  """The generative model for our subscriber retention example.
  
  >>> data = generator(10000)
  >>> data.to_csv('./data/subscriber_retention.csv', index=False)
  """
  if seed is not None:
      np.random.seed(seed)
  X = FixableDataFrame(fixed=fixed)

  # the number of sales calls made to this customer
  X["Sales calls"] = np.random.uniform(0, 4, size=(n,)).round()

  # the number of sales calls made to this customer
  X["Interactions"] = X["Sales calls"] + np.random.poisson(0.2, size=(n,))

  # the health of the regional economy this customer is a part of
  X["Economy"] = np.random.uniform(0, 1, size=(n,))

  # the time since the last product upgrade when this customer came up for renewal
  X["Last upgrade"] = np.random.uniform(0, 20, size=(n,))

  # how much the user perceives that they need the product
  X["Product need"] = X["Sales calls"] * 0.1 + np.random.normal(0, 1, size=(n,))

  # the fractional discount offered to this customer upon renewal
  X["Discount"] = (
      (1 - scipy.special.expit(X["Product need"])) * 0.5
      + 0.5 * np.random.uniform(0, 1, size=(n,))
  ) / 2

  # What percent of the days in the last period was the user actively using the product
  X["Monthly usage"] = scipy.special.expit(
      X["Product need"] * 0.3 + np.random.normal(0, 1, size=(n,))
  )

  # how much ad money we spent per user targeted at this user (or a group this user is in)
  X["Ad spend"] = (
      X["Monthly usage"] * np.random.uniform(0.99, 0.9, size=(n,))
      + (X["Last upgrade"] < 1)
      + (X["Last upgrade"] < 2)
  )

  # how many bugs did this user encounter in the since their last renewal
  X["Bugs faced"] = np.array([np.random.poisson(v * 2) for v in X["Monthly usage"]])

  # how many bugs did the user report?
  X["Bugs reported"] = (
      X["Bugs faced"] * scipy.special.expit(X["Product need"])
  ).round()

  # did the user renew?
  X["Did renew"] = scipy.special.expit(
      7
      * (
          0.18 * X["Product need"]
          + 0.08 * X["Monthly usage"]
          + 0.1 * X["Economy"]
          + 0.05 * X["Discount"]
          + 0.05 * np.random.normal(0, 1, size=(n,))
          + 0.05 * (1 - X["Bugs faced"] / 20)
          + 0.005 * X["Sales calls"]
          + 0.015 * X["Interactions"]
          + 0.1 / (X["Last upgrade"] / 4 + 0.25)
          + X["Ad spend"] * 0.0
          - 0.45
      )
  )

  # in real life we would make a random draw to get either 0 or 1 for if the
  # customer did or did not renew. but here we leave the label as the probability
  # so that we can get less noise in our plots. Uncomment this line to get
  # noiser causal effect lines but the same basic results
  # X["Did renew"] = scipy.stats.bernoulli.rvs(X["Did renew"])

  return X
