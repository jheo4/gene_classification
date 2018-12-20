
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import numpy as np
import imutils
import sklearn
import data.gene_dataset as gd

train_data, train_labels, test_data, test_labels =\
    gd.GeneDataset(classic=True).prepare_calssic_data()

classes = ['EI', 'IE', 'N']
best_k = 0
best_accuracy = 0

for k in range(1, 100, 1):
  model = KNeighborsClassifier(n_neighbors=k)
  model.fit(train_data, train_labels)

  predictions = model.predict(test_data)
  accuracy = f1_score(test_labels, predictions, average='weighted')
  print("k-%d accuracy : %.3f%%" % (k, accuracy * 100))
  if best_accuracy < accuracy:
    best_accuracy = accuracy
    best_k = k

print("The best value of k(%d)'s accuracy : %.3f" %
    (best_k, best_accuracy * 100))

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(train_data, train_labels)
predictions = model.predict(test_data)
print()
print("                           REPORT")
print(classification_report(test_labels, predictions, target_names=classes))


#print("best accuracies : %.2f" % (accuracies_k[best_practice_index]))
