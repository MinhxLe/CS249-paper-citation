"""
data_handler contains set of parsers for the dataset
"""
import numpy as np
#import factorgraph as fg


def Print3Entries(d):
  num = 0
  for x, y in d.items():
    if num > 2:
      break
    print(x, y)
    num = num + 1




class DataHandler:

  
  
  
  """
  Code below reads the file "papers", and stores (title, ID) pairs in titleToID
  """
  titleToID = {} #the dictionary that maps the title to its ID
  fpapers = open("papers", "r")
  for x in fpapers:

    t = x.split()[0:2]
    titleToID[t[1]] = t[0]
    
  print()
  print("titleToID looks like this:")
  Print3Entries(titleToID)
  print()
  print("There are", len(titleToID.values()), "papers with at least one title")
  
  """
  Code below reads the file "classifications", and stores (ID, label) pairs in labels
  """
  labels = {} #the dictionary that maps the ID to its label
  fclassifications = open("classifications", "r")
  i = 0
  for x in fclassifications:
    t = x.split()
    if len(t) == 0:
      break;
    if t[0] == 'keywords':
      continue
    topic = t[1].split("/")
    if t[0] in titleToID.keys():
      labels[titleToID[t[0]]] = topic[1]
    else:
      i = i + 1
  print()
  print("labels looks like this:")
  Print3Entries(labels)
  print()
  print("There are", len(labels), "labeled papers")
  print("There are", i, "papers with a label but doesn't have an ID (useless ones)")
  
  """
  Code below reads the file "citations", and stores (ID, cited ID) pairs in citations
  """
  citations = {} #the dictionary that maps the ID to a list of ID's it cites
  fcitations = open("citations", "r")
  counter = 0
  for x in fcitations:
    c = x.split()
    if c[0] not in citations.keys():
      citations[c[0]] = set()
      if c[0] in labels.keys():
        counter = counter + 1
    citations[c[0]].add(c[1])
    
  print()
  print("citations looks like this:")
  Print3Entries(citations)
  print()
  print("There are", len(citations), "papers whose citations are provided")
  print("Among them", counter, "papers have labels")

  """
  Code below removes the papers in the dictionary "labels" whose citations are not provided
  """
  removeKeys = []
  for x in labels:
    if x not in citations:
      removeKeys.append(x)
  for x in removeKeys:
    del labels[x]
  print(len(labels))
  
  """
  g = fg.Graph()

  for var in citations:
    if var in labels.keys():
      g.rv(var, 1)
      g.factor([var], potential = np.array([1]))
      for citedpapers in citations[var]:
        if citedpapers in citations.keys(): #only add a factor if it is a valid paper
          if citedpapers in labels:
            #var is labeled, citedpapers is labeled
            g.factor([var, citedpapers], potential = np.array([1]))
          else:
            #var is labeled, citedpapers is not labeled
            g.factor([var, citedpapers], potential = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
    else:
      g.rv(var, 10)
      g.factor([var], potential = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
      for citedpapers in citations[var]:
        if citedpapers in citations.keys(): #only add a factor if it is a valid paper
          if citedpapers in labels:
            #var is not labeled, citedpapers is labeled
            g.factor([var, citedpapers], potential = np.array([1], [1], [1], [1], [1], [1], [1], [1], [1], [1]))
          else:
            #var is not labeled, citedpapers is not labeled
            g.factor([var, citedpapers], potential = np.array(
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))

    
  iters, converged = g.lbp(normalize=True)
  print("LBP ran for", iters, "iterations. Converged =", converged)
  
  """
  
  
  
  
  