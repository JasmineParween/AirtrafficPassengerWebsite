from flask import Flask, render_template,request,redirect
import pandas as pd
import numpy as np
import json
import pickle

app = Flask(__name__)

file = open('ML/filtered_data.json')
filtered_data = json.load(file)
file.close()

file = open('ML/raw_data.json')
raw_data = json.load(file)
file.close()


model = pickle.load(open('ML/model.pkl','rb'))
global df
df = pd.read_csv('ML/final_df.csv')
print("df created")
@app.route('/', methods = ['GET' , "POST"])
def home():
    if request.method == "POST":
        year = request.form['year']
        month =  raw_data['Month'][request.form['month']]
        geo_summary = raw_data['GEO Summary'][request.form['geo_summary']]
        geo_region = raw_data['GEO Region'][request.form['geo_region']]
        operating_airline = raw_data['Operating Airline'][request.form['operating_airline']]
        published_airline = raw_data['Published Airline'][request.form['published_airline']]
        activity_type_code = raw_data['Activity Type Code'][request.form['activity_type_code']]
        terminal = raw_data['Terminal'][request.form['terminal']]
        boarding_area = raw_data['Boarding Area'][request.form['boarding_area']]
        price_cat_code = raw_data['Price Category Code'][request.form['price_cat_code']]

        if operating_airline not in filtered_data['Operating Airline']:
            operating_airline = 'Rare_var'
        if published_airline not in filtered_data['Published Airline']:
            published_airline = 'Rare_var'
        # print(request.form)
        d={
            'Operating Airline': operating_airline,
            'Published Airline' : published_airline,
            'GEO Summary' : geo_summary,
            'GEO Region' : geo_region,
            'Activity Type Code' : activity_type_code,
            'Price Category Code' : price_cat_code,
            'Terminal' : terminal,
            'Boarding Area' : boarding_area,
            'Adjusted Passenger Count' : 2000,   #just taking it will not affect
            'Year' : int(year),
            'Month' : month
        }
        global df
        df1 = df.append(d,ignore_index = True)
        # print(df.iloc[-1,:])
        df1 = pd.get_dummies(df1)
        t = df1.iloc[-1,1:]
        t = np.array(t).reshape(1,-1)
        pred_value = model.predict(t)
        print(pred_value)
        return render_template('result.html', pred_value = int(pred_value))
        
    return render_template('index.html', raw_data=raw_data)

if __name__ == '__main__':
    app.run(debug=True) 