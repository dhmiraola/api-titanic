from flask import Flask
from flask import Flask, request
import pandas as pd
import joblib
app = Flask(__name__)
@app.route('/predice', methods=['POST'])
def predict():
    json_ = request.json
    query_df = pd.DataFrame([json_])  # construye DataFrame desde el JSON
    # asegúrate de tener las columnas correctas
    query_df = query_df[['Age', 'C', 'Fare', 'Parch', 'Pclass', 'Q', 'S', 'SibSp', 'female', 'male']]
    
    classifier = joblib.load('classifier.pkl')
    prediction = classifier.predict(query_df)
    if prediction[0]:
        return "TRUE: El pasajero pudo haber sobrevivido"
    else:
        return "FALSE: El pasajero pudo NO haber sobrevivido"
if __name__ == "__main__":
    app.run(port=8000, debug=True)