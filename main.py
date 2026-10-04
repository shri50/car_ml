from flask import Flask, jsonify, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


def predict_price(Brand, 
                Variant, 
                Model_Year, 
                City, 
                Kilometer, 
                Fuel_Type, 
                Drive_Mode,
                CURRENT_YEAR = 2026
                ):
    input_data = pd.DataFrame({
                 'Brand': [Brand],
                 'Variant': [Variant],
                 'Model Year': [Model_Year],
                 'City': [City],
                 'Kilometer': [Kilometer],
                 'Fuel_Type': [Fuel_Type],
                 'Drive_Mode': [Drive_Mode],
        
    })
    input_data["Car_Age"] = CURRENT_YEAR - input_data["Model Year"]
    input_data["km_per_year"] = round(input_data["Kilometer"] / input_data["Car_Age"] ,1)

    with open(r"model/xgb_model.pkl","rb") as model_file:
        model = pickle.load(model_file)

    predict = model.predict(input_data)[0]
    return  predict   

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    fields = ['Brand', 'Variant', 'Model_Year', 'City', 'Kilometer', 'Fuel_Type', 'Drive_Mode']
    if not all(field in data for field in fields):
        return jsonify({'error': 'Missing required fields'}), 400

    price = predict_price (
                     Brand =data['Brand'],
                     Variant = data['Variant'],
                     Model_Year =data['Model_Year'],
                     City = data['City'],
                     Kilometer = data['Kilometer'],
                     Fuel_Type = data['Fuel_Type'],
                     Drive_Mode = data['Drive_Mode'],
                     CURRENT_YEAR = 2026
                )
    print(price)

    return f"predicted_price = {price}"


if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=8080)

   