class pycodes:
	def program1():
		print(f"""
import heapq

def aStarAlgo(start_node, stop_node):
    open_set = [(0, start_node)]
    closed_set = set()
    g = {start_node: 0}
    parents = {start_node: start_node}

    while open_set:
        _, n = heapq.heappop(open_set)

        if n == stop_node:
            path = []
            while parents[n] != n:
                path.append(n)
                n = parents[n]
            path.append(start_node)
            path.reverse()
            print(f"Path found: {path}")
            return path
        
        closed_set.add(n)
        for m, weight in get_neighbors(n) or []:
            tentative_g = g[n] + weight
            if m not in g or tentative_g < g[m]:
                g[m] = tentative_g
                heapq.heappush(open_set, (tentative_g + heuristic(m), m))
                parents[m] = n
    
    print("Path doesn't exist")
    return None

def get_neighbors(v):
    return Graph_nodes.get(v, [])

def heuristic(n):
    H_dist = {
        'A': 10, 'B': 8, 'C': 5, 'D': 7, 'E': 3,
        'F': 6, 'G': 5, 'H': 3, 'I': 1, 'J': 0
    }
    return H_dist.get(n, 0)


Graph_nodes = {
    'A': [('B', 6), ('F', 3)],
    'B': [('C', 3), ('D', 2)],
    'C': [('D', 1), ('E', 5)],
    'D': [('C', 1), ('E', 8)],
    'E': [('I', 5), ('J', 5)],
    'F': [('G', 1), ('H', 7)],
    'G': [('I', 3)],
    'H': [('I', 2)],
    'I': [('E', 5), ('J', 3)]
}

aStarAlgo('A', 'J')
	""")

	def Program2():
		print("""
from queue import PriorityQueue

class Graph:
    def __init__(self, graph, heuristic, start):
        self.graph = graph
        self.heuristic = heuristic
        self.start = start
        self.parent = {}
        self.solutionGraph = {}

    def applyAOStar(self):
        priority_queue = PriorityQueue()
        priority_queue.put((self.heuristic[self.start], self.start))

        while not priority_queue.empty():
            cost, current = priority_queue.get()
            if current == 'T':
                break

            for neighbor, weight in self.graph.get(current, {}).items():
                new_cost = self.heuristic[neighbor] + weight
                priority_queue.put((new_cost, neighbor))
                self.parent[neighbor] = current

        for node, parent in self.parent.items():
            if parent in self.solutionGraph:
                self.solutionGraph[parent].append(node)
            else:
                self.solutionGraph[parent] = [node]

    def printSolution(self):
        print("FOR GRAPH SOLUTION, TRAVERSE THE GRAPH FROM THE START NODE:", self.start)
        print("--------------------------------------------------------------------------")
        print(f"{self.solutionGraph}")
        print("--------------------------------------------------------------------------")


heuristic_values = {'A': 1, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1, 'T': 3}
graph_edges = {
    'A': {'B': 1, 'C': 1, 'D': 1},
    'B': {'G': 1, 'H': 1},
    'C': {'J': 1},
    'D': {'E': 1, 'F': 1},
    'G': {'I': 1}
}

G1 = Graph(graph_edges, heuristic_values, 'A')
G1.applyAOStar()
G1.printSolution()
  """)

	def program3():
		print("""
import csv
with open ('./Datasets/trainingexamples.csv') as f:
    csv_file = csv.reader(f)
    data = list(csv_file)

    specific = data[0][:-1]
    general = [['?' for _ in specific] for _ in specific]

    for i in data:
        attr_values = i[:-1]
        outcome = i[-1]

        for j in range(len(specific)):
            if outcome == "Yes":
                if attr_values[j] != specific[j]:
                    specific[j] = '?'
                    general[j][j] = '?'
            elif outcome == "No":
                if attr_values[j] != specific[j]:
                    general[j][j] = specific [j]
                else:
                    general[j][j] = '?'

        print(f"Step {data.index(i)} of Candidate Elimination Algorithm")
        print(f"Specific Hypothesis: {specific} ")
        print (f"General Hypothesis: {general}")

    generalHypothesis = [list(filter(lambda x: x != '?', i)) for i in general]
    print(f"Final General Hypothesis: {specific}")
    print(f"Final General Hypothesis: {generalHypothesis}")
  """)

	def program4():
		print("""
import numpy as np
import pandas as pd
PlayTennis = pd.read_csv("./Datasets/Play tennis.csv")
from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()

PlayTennis['Outlook'] = Le.fit_transform(PlayTennis['Outlook'])
PlayTennis['Temperature'] = Le.fit_transform(PlayTennis['Temperature'])
PlayTennis['Humidity'] = Le.fit_transform(PlayTennis['Humidity'])
PlayTennis['Windy'] = Le.fit_transform(PlayTennis['Windy'])
y = PlayTennis['PlayTennis']
x = PlayTennis.drop(['PlayTennis'], axis = 1)
from sklearn import tree
classifier = tree.DecisionTreeClassifier(criterion = 'entropy')
classifier = classifier.fit(x, y)
tree.plot_tree(classifier) 

X_pred = classifier.predict(x)
X_pred == y
  """)

	def program5():
		print("""
import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Input data
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float) / 9
y = np.array(([92], [86], [89]), dtype=float) / 100

# Neural network parameters
input_neurons, hidden_neurons, output_neurons = 2, 3, 1
epoch, lr = 7000, 0.1

# Initialize weights and biases
wh = np.random.uniform(size=(input_neurons, hidden_neurons))
wout = np.random.uniform(size=(hidden_neurons, output_neurons))
bh = np.random.uniform(size=(1, hidden_neurons))
bout = np.random.uniform(size=(1, output_neurons))

# Training
for i in range(epoch):
    # Forward propagation
    hinp = np.dot(X, wh) + bh
    hlayer_act = sigmoid(hinp)
    outinp = np.dot(hlayer_act, wout) + bout
    output = sigmoid(outinp)

    # Backpropagation
    EO = y - output
    d_output = EO * output * (1 - output)
    EH = d_output.dot(wout.T) * hlayer_act * (1 - hlayer_act)

    # Update weights and biases
    wout += hlayer_act.T.dot(d_output) * lr
    wh += X.T.dot(EH) * lr

# Print results
print(f"Input:\n{X}")
print(f"Actual Output:\n{y}")
print(f"Predicted Output:\n{output}")
  """)

	def program6():
		print("""
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
data = pd.read_csv("./Datasets/Play tennis.csv")
print(f"The first 5 data of training input: \n {data.head()}")
X = data.iloc[:, :-1]
print (f"The first 5 values of train data: \n{X.head()}")
Y = data.iloc[:, -1]
print(f"The first 5 values of train data: \n{Y.head()}")
leOutlook, leTemperature, leHumidity, leWindy = LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder()

X.Outlook = leOutlook.fit_transform(X.Outlook)
X.Temperature = leTemperature.fit_transform(X.Temperature)
X.Humidity = leHumidity.fit_transform(X.Humidity)
X.Windy = leWindy.fit_transform(X.Windy)
print(f"Now the trained data is: \n{X.head()}")
lePlayTennis = LabelEncoder()
Y.PlayTennis = lePlayTennis.fit_transform(Y)
print(f"Now the Test Data is: \n{Y}")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
classifier = GaussianNB()
classifier.fit(x_train, y_train)
from sklearn.metrics import accuracy_score
print(f"Accuracy: {accuracy_score(classifier.predict(x_test), y_test)}")
  """)

	def program7():
		print("""
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
iris = datasets.load_iris()
print(iris)
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target)
model = KMeans(n_clusters = 3)
model.fit(X_train, Y_train)
model.score
print(f"K-Means: {metrics.accuracy_score(Y_test, model.predict(X_test))}")
from sklearn.mixture import GaussianMixture
model2 = GaussianMixture (n_components = 4)
model2.fit(X_train, Y_train)
model2.score
print(f"EM Algorithm: {metrics.accuracy_score(Y_test, model2.predict(X_test))}")


# To Plot the Graph

# from sklearn.cluster import KMeans
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# data = pd.read_csv("./Datasets/7.csv")

# x1 = data['x'].values
# x2 = data['y'].values

# X = np.asarray(list(zip(x1, x2))) # Convert to numpy array
# plt.scatter(x1, x2)
# plt.show()

# markers = ['s', 'o', 'v']
# k = 3
# clusters = KMeans(n_clusters=k).fit(X)
# for i, l in enumerate(clusters.labels_):
#     plt.plot(x1[i], x2[i], marker=markers[l])
# plt.tight_layout()
# plt.show()

  """)

	def program8():
		print("""
# from sklearn.model_selection import train_test_split 
# from sklearn.neighbors import KNeighborsClassifier 
# from sklearn import datasets
# iris=datasets.load_iris()
# print("Iris Data set loaded...")
# x_train, x_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size=0.1)
# #random_state=0
# for i in range(len(iris.target_names)):
#     print("Label", i , "-",str(iris.target_names[i]))
# classifier = KNeighborsClassifier(n_neighbors=5)
# classifier.fit(x_train, y_train)
# y_pred=classifier.predict(x_test)
# print("Results of Classification using K-nn with K=5 ")
# for r in range(0,len(x_test)):
     print(f"Sample: {str(x_test[r])}\nActual-label: {str(y_test[r])}\nPredicted-label: {str(y_pred[r])}\nClassification Accuracy : {classifier.score(x_test,y_test)}")
        """)


	def program9():
 		print("""
import numpy as np
import matplotlib.pyplot as plt
def local_regression(x0, X, Y, tau):
    x0 = [1, x0]
    X = [[1, i] for i in X]
    X = np.asarray(X)
    xw = (X.T) * np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * tau))
    beta = np.linalg.pinv(xw @ X) @ xw @ Y @ x0
    return beta
def draw(tau):
    prediction = [local_regression(x0, X, Y, tau) for x0 in domain]
    plt.plot(X, Y, 'o', color='black')
    plt.plot(domain, prediction, color='red')
    plt.show()

X = np.linspace(-3, 3, num=1000)
domain = X
Y = np.log(np.abs(X ** 2 - 1) + .5)

draw(10)
draw(0.1)
draw(0.001)

        """)
