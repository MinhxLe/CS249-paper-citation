"""
data_handler contains set of parsers for the dataset
"""
import numpy as np
import factorgraph as fg
import time
import pickle

DATA = "./data/"
#the dictionary that maps the title to its ID
titleToID = {}
labels = {}
citations = {}
N_PAPERS = 27894 
N_TOPICS = 10
#Handles test
g = fg.Graph()

topic_label_dict = {'Information_Retrieval': 0, 'Databases': 1,
                     'Artificial_Intelligence': 2, 'Networking': 3,
                     'Encryption_and_Compression': 4, 'Operating_Systems': 5,
                     'Data_Structures__Algorithms_and_Theory': 6,
                     'Hardware_and_Architecture': 7,
                     'Programming': 8, 'Human_Computer_Interaction': 9}

def Print3Entries(d):
  num = 0
  for x, y in d.items():
    if num > 2:
      break
    print(x, y)
    num = num + 1


def loadTitles():
  """
  Code below reads the file "papers", and stores (title, ID) pairs in titleToID
  """
  #the dictionary that maps the title to its ID
  fpapers = open(DATA + "papers", "r")
  for x in fpapers:

    t = x.split()[0:2]
    titleToID[t[1]] = t[0]
    
  print()
  print("titleToID looks like this:")
  Print3Entries(titleToID)
  print()
  print("There are", len(titleToID.values()), "papers with at least one title")

def loadClasses():
  """
  Code below reads the file "classifications", and stores (ID, label) pairs in labels
  """
  #the dictionary that maps the ID to its label
  fclassifications = open(DATA + "classifications", "r")
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

def loadCitations():
  """
  Code below reads the file "citations", and stores (ID, cited ID) pairs in citations
  """
  #the dictionary that maps the ID to a list of ID's it cites
  fcitations = open(DATA + "citations", "r")
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

def removeKeys():
  """
  Code below removes the papers in the dictionary "labels" whose citations are not provided
  """
  removeKeys = []
  topics = {}
  for x in labels:
    topics[labels[x]] = -1
    if x not in citations:
      removeKeys.append(x)
  for x in removeKeys:
    del labels[x]
  print(len(labels))


def graphTest():
  #Add all rvs to graph
  print("Adding unary factors to graph...")
  for var in citations:
    if var in labels.keys():
      g.rv(var, 10)
      topic = topic_label_dict[labels[var]]
      g.factor([var], potential = np.array([1.0 if x is topic else 0.0 for x in range(10)]))
    else:
      g.rv(var,10)
      g.factor([var], potential = \
                     np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))

  print("Adding binary factors to graph...")
  num_citations = len(citations)
  count = 1
  start = time.process_time()

  for var in citations:
    #Just for timing purposes
    if count % 4500 == 0:
      time_passed = (time.process_time()-start)/60
      print('{} factors added'.format(count), 'in {:.3} minutes'.format(time_passed))
      remaining_time = time_passed/count*(num_citations-count)
      print( '\t\tApproximately {:.3} minutes remaining'.format(remaining_time))
    count = count + 1

    if var in labels.keys():
      topic = topic_label_dict[labels[var]]
      
      for citedpapers in citations[var]:
        if citedpapers in citations.keys() and citedpapers != var: 
          #only add a factor if it is a valid paper
          if citedpapers in labels:
            #var is labeled, citedpapers is labeled
            topic2 = topic_label_dict[labels[citedpapers]]

            fact_val = 1.0
            if topic != topic2:
              fact_val = 0.5
            g.factor([var, citedpapers], potential = \
                     np.array([[fact_val if (x is topic or x is topic2) else 0.0 for x in range(10)] for i in range(10)]))
          else:
            #var is labeled, citedpapers is not labeled
            g.factor([var, citedpapers], potential = \
                     np.array([[0.55 if x is topic else 0.05 for x in range(10)] for i in range(10)]))
    else:  
      for citedpapers in citations[var]:
        if citedpapers in citations.keys() and citedpapers != var: 
          #only add a factor if it is a valid paper
          if citedpapers in labels:
            topic = topic_label_dict[labels[citedpapers]]
            #var is not labeled, citedpapers is labeled
            g.factor([var, citedpapers], potential = \
                     np.array([[0.55 if x is topic else 0.05 for x in range(10)] for i in range(10)]))
          else:
            #var is not labeled, citedpapers is not labeled
            g.factor([var, citedpapers], potential = \
                     np.array([[0.1 for x in range(10)] for i in range(10)]))


  print("Handling Loopy BP...")
  iters, converged = g.lbp(normalize=True, progress=True)
  print("BP Completed")
  print("LBP ran for", iters, "iterations. Converged =", converged)

  #print("Saving to pickle...")
  #pickle.dump(g, open(DATA + "saved_model.p", "wb"))
  

def main():
  loadTitles()
  loadClasses()
  loadCitations()
  removeKeys()
  #graphTest()
  
if __name__== "__main__":
  main()

  
  
  
