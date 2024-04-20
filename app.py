from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle

# Create Flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

# Map definitions for converting form inputs
general_health_map = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
race_map = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}
diabetic_map = {'0': 0, '1': 1, '2': 0, '3': 1}
sex_map = {"0": 0, "1": 1}
diff_walking_map = {"0": 0, "1": 1}
age_category_map = {
    "18-24": 21, "25-29": 27, "30-34": 32, "35-39": 37,
    "45-49": 47, "50-54": 52, "55-59": 57, "60-64": 62,
    "65-69": 67, "70-74": 72, "75-79": 77, "80 or older": 85
}

@flask_app.route("/")
def home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    try:
        # Convert form inputs using the maps
        age_category = age_category_map[request.form['ageCategory']]
        gen_health = general_health_map[request.form['genHealth']]
        bmi = float(request.form['bmi'])
        physical_health = float(request.form['physicalHealth'])
        sleep_time = float(request.form['sleepTime'])
        race = race_map[request.form['race']]
        diabetic = diabetic_map[request.form['diabetic']]
        sex = sex_map[request.form['sex']]
        diff_walking = diff_walking_map[request.form['diffWalking']]
        
        # Prepare feature array for prediction
        features = np.array([[
            age_category, gen_health, bmi, physical_health,
            sleep_time, race, diabetic, sex, diff_walking
        ]])
        
        # Predict using a loaded model
        prediction = model.predict(features)
        prediction_text = "Heart Disease: Yes" if prediction[0] == 1 else "Heart Disease: No"
    except Exception as e:
        return jsonify(error=str(e)), 400
    
    return jsonify(prediction_text=prediction_text)

if __name__ == "__main__":
    flask_app.run(debug=True)
