import pickle
from flask import Flask,request,render_template
import pandas as pd
app= Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

ll=[]

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods = ['GET','POST'])
def index():
    if request.method=='POST':
        ind=request.form                  # returns an immutable dictionary
        
        l=dict(ind)
        in_str =[[float(l['online_order']), float(l['book_table']),float(l['votes']), float(l['location']),float(l['rest_type']),float(l['cuisine']), float(l['book_table']) ]]
        
        in_val = model.predict(pd.DataFrame(in_str))
        
        ll.append(in_val[0])
    
    return render_template('index.html',result=ll)


@app.route('/final')
def final():
    return render_template('solution.html',result=ll)
        
if __name__=='__main__':
    app.run(debug=False,port=4000)
