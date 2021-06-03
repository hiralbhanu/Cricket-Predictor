from flask import Flask, request, jsonify,redirect,url_for,render_template
import pickles_algos
from algos import *

app = Flask(__name__)


@app.route('/')
def root():
    return app.send_static_file("index.html")

@app.route('/success/<name>')
def success(name):
   return render_template('output.html', name=name)

@app.route('/predict')
def prediction():
    X_test_input = []
    X_test_input1 = []
    innings1_runs = request.values['innings1_runs']
    X_test_input.append(innings1_runs)
    X_test_input1.append(innings1_runs)
    innings1_wickets = request.values['innings1_wickets']
    X_test_input.append(innings1_wickets)
    X_test_input1.append(innings1_wickets)
    innings1_overs_batted = request.values['innings1_overs_batted']
    X_test_input.append(innings1_overs_batted)
    X_test_input1.append(innings1_overs_batted)
    print(X_test_input)
    print(X_test_input1)
    '''DL_method = request.values['DL_method']
    X_test_input.append(DL_method)
    X_test_input1.append(DL_method)
    print(X_test_input)
    print(X_test_input1)'''
    algo = request.values['algo']
    data = dict()
    if algo == 'randForest':
        data['prediction'] = randforest(X_test_input)
    elif algo == 'naiveBayes':
        data['prediction'] = naivebayes(X_test_input)
    elif algo == 'knn':
        data['prediction'] = k_nearest_neighbours(X_test_input)
    elif algo == 'decisionTree':
        data['prediction'] = decision_tree(X_test_input)
    else:
        data['prediction'] = 'None'
    print(data['prediction'])
    if data['prediction'] == True:
        return redirect(url_for('success',name = "Team batting 2nd will be most probable to win!"))
    elif data['prediction'] == False:
        return redirect(url_for('success',name = "Team batting 1st will be most probable to win!"))
    

if __name__ == '__main__':
    app.run(debug=True)
