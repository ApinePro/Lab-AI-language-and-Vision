import json
from typing import Awaitable
from qwikidata.sparql import (get_subclasses_of_item,
                              return_sparql_query_results)

# send any sparql query to the wikidata query service and get full result back
# here we use an example that counts the number of humans
COUNT = 3
OUTPUT_FILE = "TreeFr-3.json"
QUERIED_ENTITY = "Q142"

def query(entity):
  part1 = """
  SELECT DISTINCT ?subdivision ?subdivisionLabel 
  WHERE
  {
  """
  part2 = "<" + entity + "> " + "wdt:P150 ?subdivision ."
  part3 = """
  ?subdivision rdfs:label ?subdivisionLabel .
  FILTER(lang(?subdivisionLabel)='en')
  FILTER NOT EXISTS{?subdivision wdt:P582|wdt:P576 ?end}
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }

  }
  ORDER BY ?subdivisionLabel
  """
  return part1 + part2 + part3


def get_label(address):
  label = address.lstrip("http://www.wikidata.org/entity")
  return label


def get_sub(hyper_area):
  dic = {}
  final_dic = {}
  label = get_label(hyper_area)
  sparql_query = query(hyper_area)
  try:
    res = return_sparql_query_results(sparql_query)
  except BaseException:
    pass
  else:
    if(len(res["results"]["bindings"]) != 0):
      subdivisions = res["results"]["bindings"]
      for division in subdivisions:
        div_label = division["subdivisionLabel"]["value"]
        div_entity = division["subdivision"]["value"]
        dic[div_label] = get_sub(div_entity)
      final_dic["label"] = label
      final_dic["subdivision"] = dic
      return final_dic
    return {}

def get_sub_count(hyper_area, count):
  label = get_label(hyper_area)
  count = count + 1
  final_dic = {}
  if(count >= COUNT):
    final_dic["label"] = label
    final_dic["subdivision"] = {}
    return final_dic
  dic = {}
  error = True
  sparql_query = query(hyper_area)
  while error == True:
    try:
      res = return_sparql_query_results(sparql_query)
    except BaseException:
      pass
    else:
      error = False
      if(len(res["results"]["bindings"]) != 0):
        subdivisions = res["results"]["bindings"]
        for division in subdivisions:
          div_label = division["subdivisionLabel"]["value"]
          print(div_label)
          div_entity = division["subdivision"]["value"]
          dic[div_label] = get_sub_count(div_entity, count)
        final_dic["label"] = label
        final_dic["subdivision"] = dic
        return final_dic
      final_dic["label"] = label
      final_dic["subdivision"] = {}
      return final_dic

def get_country_query(queried_entity):
  part1 = """
  SELECT DISTINCT ?countryLabel
  WHERE
  {
  """
  part2 = "<http://www.wikidata.org/entity/" + queried_entity + "> " + "rdfs:label ?countryLabel ."
  part3 = """
  FILTER(lang(?countryLabel)='en')
  }
  """
  return part1 + part2 + part3

def get_country_label(sparql_query):
  error = True
  while error == True:
    try:
      res = return_sparql_query_results(sparql_query)
    except BaseException:
      pass
    else:
      country_label = res["results"]["bindings"][0]["countryLabel"]["value"]
      return country_label

def get_tree(queried_entity):
  dic = get_sub_count('http://www.wikidata.org/entity/' + queried_entity, 0)

  sparql_query = get_country_query(queried_entity)
  country_label = get_country_label(sparql_query)
  
  final_dic = {}
  final_dic[country_label] = dic
  with open(OUTPUT_FILE,"w") as f:
    json.dump(final_dic,f)

def get_area_query(queried_entity):
  part1 = """
  SELECT DISTINCT ?area
  WHERE
  {
  """
  part2 = "<http://www.wikidata.org/entity/" + queried_entity + "> " + "wdt:P2046 ?area ."
  part3 = """
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
  }
  """
  return part1 + part2 + part3

#print(return_sparql_query_results(get_area_query("Q1017"))["results"]["bindings"][0]["area"]["value"])
get_tree("Q142")