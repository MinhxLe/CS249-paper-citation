from src import mrf


n_papers=10
n_topics=2

references = {}
labels = {}


for i in range(0,n_papers,2):
    references[i] = set([i+1])

for i in range(0, n_papers-2, 2):
    labels[i] = 0
    labels[i+1] = 1
#labels[n_papers-2] = 1


model = mrf.PaperMRF(
    n_papers=n_papers,
    n_topics=n_topics,
    references=references,
    labels=labels,
) 

model.run_EM_algorthm(1000)
inferer = model.get_inferer()
graph = inferer.graph

graph.print_rv_marginals()
#graph.print_messages()
print(model.unary_parameters)
print(model.reference_parameters)