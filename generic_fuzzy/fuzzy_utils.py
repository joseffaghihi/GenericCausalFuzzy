
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules


class GenericFuzzy:

  def __init__(self,
               data:pd.DataFrame,
               c:float = 0.5,
               num_fuzzy_sets:int = 6,
               num_intervals:int = 5,
               target:int='bloodpressure'
               ):
    self.data = data
    self.c = c
    self.num_fuzzy_sets = num_fuzzy_sets
    self.num_intervals  = num_intervals
    self.target = target
  
  def partition(self,
                variable_name:str,
                i,
                num_intervals:int=5):

    min = self.data[variable_name].min()
    max = self.data[variable_name].max()
    return  min + i*(max - min)/num_intervals
  
  def membership_function(self, variable:str, i, x):
    if i == 0:
      point_1 = self.partition(variable, i)
      point_2 = self.partition(variable, i + 1)
      mean = point_1
      std = self.c*(point_2 - point_1)
      return np.where(point_1 <= x, np.exp(-((x - mean)**2)/(2*(std**2))), 1)
    elif i == self.num_intervals:
      point_1 = self.partition(variable, i - 1)
      point_2 = self.partition(variable, i )
      mean = point_2
      std = self.c*(point_2 - point_1)
      return np.where(x <= point_2, np.exp(-((x - mean)**2)/(2*(std**2))), 1)
    else:
      point_1 = self.partition(variable, i - 1)
      point_2 = self.partition(variable, i)
      point_3 = self.partition(variable, i + 1)
      mean = point_2
      std = self.c*(point_3 - point_2)
      return np.exp(-((x - mean)**2)/(2*(std**2)))
  
  def apply_and_replace(self, variable:str, x):
      max_result = -float('inf')  # Initialize with negative infinity
      max_index = -1  # Initialize index
  
      for i in range(self.num_fuzzy_sets):
          result = self.membership_function(variable, i, x)
          if result > max_result:
              max_result = result
              max_index = i
      return max_index

  def apply_fuzzy_sets_to_variables(self, variables:list=[]):
    for variable in self.data.columns:
      self.data[variable] = self.data[variable].apply(
          lambda x: self.apply_and_replace(variable, x))
    return self.data
  
  def generate_rules(self, target_value:str = None):
    if not self.target and not target_value:
       raise ValueError('Target value is required!')
    elif target_value:
       self.target = target_value

    self.apply_fuzzy_sets_to_variables()
    self.Data = self.data.copy(deep=True)
  
    # Encode categorical data (convert to binary format)
    self.data_encoded = pd.get_dummies(self.Data, columns=self.data.columns)

    # Apply Apriori algorithm to find frequent itemsets
    self.frequent_itemsets = apriori(self.data_encoded, min_support=0.01, use_colnames=True)
    # Generate association rules
    rules = association_rules(self.frequent_itemsets, metric="lift", min_threshold=1.0)

    # List of all blood pressure related items
    levels = list(np.arange(1, self.num_fuzzy_sets))
    target_items = [f'{self.target}_{level}' for level in levels]

    # Filter rules where all items in the consequents are blood pressure related
    target_only_rules = rules[rules['consequents'].apply(lambda x: all(item in target_items for item in x))]

    self.rules_with_one_element = target_only_rules[target_only_rules['antecedents'].apply(lambda x: len(x) == 1)]
    self.rules_with_two_elements = target_only_rules[target_only_rules['antecedents'].apply(lambda x: len(x) == 2)]

    return self.rules_with_one_element, self.rules_with_two_elements
  
  def transform_item(self, item):
    # Split the item into parts and handle cases with more than one underscore
    parts = item.split('_')
    print(parts)
    if len(parts) < 2:
       return item
    if len(parts) > 2:
        # Rejoin all parts except the last as the variable name
        variable = '_'.join(parts[:-1])
        value = parts[-1]
    else:
        variable, value = parts

    # Return the transformed item as a normal text string
    return f"{variable}['{value}']"

  def transform_rule_set(self, rule_set):
      # Transform each item in the set
      return set(self.transform_item(item) for item in rule_set)
  
  def format_rules(self):
     for d in [self.rules_with_one_element, self.rules_with_two_elements]:
      d['antecedents'] = d['antecedents'].apply(self.transform_rule_set)
      d['consequents'] = d['consequents'].apply(self.transform_rule_set)

