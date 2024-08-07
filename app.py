from flask import Flask, request, render_template # type: ignore
import pickle
import pandas as pd # type: ignore

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        area = request.form['area']
        bath = request.form['bath']
        bedroom = request.form['bedroom']
        type = request.form['type']
        location = request.form['location']

        # Process the data here
        # For example, let's assume we have a function to calculate the price
        price = calculate_price(area, bath, bedroom, type, location)

        return render_template('result.html', price=price)
    return render_template('index.html')

def calculate_price(area, bath, bedroom, type1, location):

    with open('model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    with open('column_transformer.pkl', 'rb') as f:
        loaded_column_transformer = pickle.load(f)

    new_data=pd.DataFrame([[type1, location, area, bath, bedroom]], columns=['Type', 'Location_Category', 'Area', 'Bath', 'Bedroom'])

    new_data_transformed = loaded_column_transformer.transform(new_data)
    new_predictions = loaded_model.predict(new_data_transformed)

    return new_predictions[0]

if __name__ == '__main__':
    app.run(debug=True)