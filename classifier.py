#using first classifier
from sklearn import ensemble

clf = ensemble.RandomForestClassifier()

X=[[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y= ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

clf=clf.fit(X, Y)

prediction=clf.predict([[190, 55, 32]])#user nodes

print(prediction)
#result = female

#using second classifier
from sklearn import neighbors

clf = neighbors.KNeighborsClassifier()

X=[[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y= ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

clf=clf.fit(X, Y)

prediction=clf.predict([[190, 55, 32]])

print(prediction)
#result = male

#using third classifier(algo)
from sklearn import neural_network
 	
clf = neural_network.MLPClassifier()

X=[[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y= ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

clf=clf.fit(X, Y)

prediction=clf.predict([[80, 55, 32]])

print(prediction)

