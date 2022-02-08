"""Importing Libraries"""

from flask import Flask, request, render_template
import pickle

# load the model from the disk
filename1 = 'model1.pkl'
filename2 = 'model2.pkl'
filename3 = 'model3.pkl'

clf1 = pickle.load(open(filename1, 'rb'))
clf2 = pickle.load(open(filename2, 'rb'))
clf3 = pickle.load(open(filename3, 'rb'))
cv = pickle.load(open('transform.pkl','rb'))

print()
print('Model loaded successfully...')
print()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Predict1', methods = ['POST'])
def predict1():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf1.predict(vect)
	return render_template('result.html', prediction = my_prediction)

@app.route('/Predict2',methods=['POST'])
def predict2():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf2.predict(vect)
	return render_template('result.html',prediction = my_prediction)

@app.route('/Predict3',methods=['POST'])
def predict3():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf3.predict(vect)
	return render_template('result.html',prediction = my_prediction)

if __name__ == "__main__":
    app.run(debug=True)

