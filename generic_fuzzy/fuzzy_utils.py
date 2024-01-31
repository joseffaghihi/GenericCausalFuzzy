import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class GenericFuzzy:

  def __init__(self,
               data:pd.DataFrame,
               c:float = 0.2,
               num_fuzzy_sets:int = 6,
               num_intervals:int = 5,
               target_consequent:int='bloodpressure',
               num_points:int= 100,
               min_support:float=0.01
               ):
    self.data = data
    self.c = c
    self.num_fuzzy_sets = num_fuzzy_sets
    self.num_intervals  = num_intervals
    self.target_consequent = target_consequent
    self.min_support = min_support

    self.apply_fuzzy_sets_to_variables()
    self.generate_rules()
    self.format_rules()

    self.num_points = num_points
    self.fuzzy_rules = []
    self.variables = {el: None for el in self.data.columns}
    self.provide_antecedent()
    self.provide_consequent()
    self.apply_membership_function()

    self.namespace = {el: self.antecedent[el] for el in self.antecedent}
    self.namespace[self.consequent.label] = self.consequent
  
  def partition(self, min, max, i):
    return  min + i*(max - min)/self.num_intervals

  def membership_function(self, min, max, i, x):
    if i == 0:
      point_1 = self.partition(min, max, i)
      point_2 = self.partition(min, max, i + 2)
      mean = point_1
      std = self.c*(point_2 - point_1)
      return np.where(point_1 <= x, np.exp(-((x - mean)**2)/(2*(std**2))), 1)
    elif i == self.num_intervals:
      point_1 = self.partition(min, max, i - 2)
      point_2 = self.partition(min, max, i )
      mean = point_2
      std = self.c*(point_2 - point_1)
      return np.where(x <= point_2, np.exp(-((x - mean)**2)/(2*(std**2))), 1)
    else:
      point_1 = self.partition(min, max, i - 1)
      point_2 = self.partition(min, max, i)
      point_3 = self.partition(min, max, i + 1)
      mean = point_2
      std = self.c*(point_3 - point_2)
      return np.exp(-((x - mean)**2)/(2*(std**2)))
  
  def apply_and_replace(self, min, max, x):
      max_result = -float('inf')  # Initialize with negative infinity
      max_index = -1  # Initialize index
  
      for i in range(self.num_fuzzy_sets):
          result = self.membership_function(min, max, i, x)
          if result > max_result:
              max_result = result
              max_index = i
      return max_index

  def apply_fuzzy_sets_to_variables(self, variables:list=[]):
    self.Data = self.data.copy(deep=True)
    for variable in self.Data.columns:
      self.min = self.Data[variable].min()
      self.max = self.Data[variable].max()
      self.Data[variable] = self.Data[variable].apply(
          lambda x: self.apply_and_replace(self.data[variable].min(), self.data[variable].max(), x))
    return self.Data
  
  def generate_rules(self, target_value:str = None):
    if not self.target_consequent and not target_value:
       raise ValueError('Target value is required!')
    elif target_value:
       self.target_consequent = target_value
  
    self.apply_fuzzy_sets_to_variables()
    # Encode categorical data (convert to binary format)
    self.data_encoded = pd.get_dummies(self.Data, columns=self.data.columns)

    # Apply Apriori algorithm to find frequent itemsets
    self.frequent_itemsets = apriori(self.data_encoded, min_support=self.min_support, use_colnames=True)
    # Generate association rules
    rules = association_rules(self.frequent_itemsets, metric="lift", min_threshold=1.0)

    # List of all blood pressure related items
    levels = list(np.arange(1, self.num_fuzzy_sets))
    target_items = [f'{self.target_consequent}_{level}' for level in levels]

    # Filter rules where all items in the consequents are blood pressure related
    target_only_rules = rules[rules['consequents'].apply(lambda x: all(item in target_items for item in x))]

    self.rules_with_elements = {}

    for i in range(1, len(self.data.columns)):
       self.rules_with_elements[i] = target_only_rules[
          target_only_rules['antecedents'].apply(lambda x: len(x) == i)]

    return self.rules_with_elements
  
  def transform_item(self, item):
    # Split the item into parts and handle cases with more than one underscore
    parts = item.split('_')
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
    for d in self.rules_with_elements:
      self.rules_with_elements[d]['antecedents'] = self.rules_with_elements[d]['antecedents'].apply(self.transform_rule_set)
      self.rules_with_elements[d]['consequents'] = self.rules_with_elements[d]['consequents'].apply(self.transform_rule_set)

  def provide_antecedent(self):
    self.antecedent = {
      el: ctrl.Antecedent(
            np.arange(
              self.data[el].min(), self.data[el].max(),
              (self.data[el].max() - self.data[el].min())/self.num_points), el)
      
      for el in self.data.columns if el != self.target_consequent
    }

  def provide_consequent(self):
    self.consequent = ctrl.Consequent(
      np.arange(
        self.data[self.target_consequent].min(), self.data[self.target_consequent].max(),
        (self.data[self.target_consequent].max() - self.data[self.target_consequent].min())/self.num_points),
      self.target_consequent
    )
  
  def membership_function_fuzzy_generic_control(self, variable, min, max):
    for i in range(self.num_fuzzy_sets):
      string = str(i)
      if i == 0:
        point_1 = self.partition(min, max, i)
        point_2 = self.partition(min, max, i + 2)
        variable[string] = fuzz.gaussmf(variable.universe, point_1, self.c*(point_2 - point_1))
      elif i == self.num_intervals:
        point_1 = self.partition(min, max, i - 2)
        point_2 = self.partition(min, max, i )
        variable[string] = fuzz.gaussmf(variable.universe, point_2, self.c*(point_2 - point_1))
      else:
        point_1 = self.partition(min, max, i - 1)
        point_2 = self.partition(min, max, i)
        point_3 = self.partition(min, max, i + 1)
        variable[string] = fuzz.gaussmf(variable.universe, point_2, self.c*(point_3 - point_1))
  
  def apply_membership_function(self):
    for el in self.antecedent.values():
      self.membership_function_fuzzy_generic_control(
        el, self.data[el.label].min(),
        self.data[el.label].max()
      )

    self.membership_function_fuzzy_generic_control(
      self.consequent, self.data[self.consequent.label].min(),
      self.data[self.consequent.label].max()
    )
  
  # Define the rules

  def parse_fuzzy_reference(self, reference_str, namespace):
      parts = reference_str.split('[')
      var_name = parts[0]
      # Correctly strip quotes and extra characters
      set_name = parts[1].replace("'", "").replace('"', "").replace("]", "")
      return namespace[var_name][set_name]

  def combine_conditions(self, conditions):
    combined = conditions[0]
    for cond in conditions[1:]:
        combined &= cond
    return combined

  def generate_fuzzy_rules(self, element_in_rule:int=2):
    for i in range(len(self.rules_with_elements[element_in_rule])):
      antecedents = list(self.rules_with_elements[element_in_rule].iloc[i]['antecedents'])
      consequents = list(self.rules_with_elements[element_in_rule].iloc[i]['consequents'])
      antecedent_conditions = [self.parse_fuzzy_reference(ant, self.namespace) for ant in antecedents]
      rule = ctrl.Rule(self.combine_conditions(antecedent_conditions), self.parse_fuzzy_reference(consequents[0], self.namespace))
      self.fuzzy_rules.append(rule)
