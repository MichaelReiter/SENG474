from util2 import Arff2Skl
from sklearn import tree

cvt = Arff2Skl('contact-lenses.arff')
label = cvt.meta.names()[-1]
X, y = cvt.transform(label)

dtc = tree.DecisionTreeClassifier(criterion='entropy')
dtc.fit(X, y)

tree.export_graphviz(dtc)
