from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = joblib.load(r'D:\ChildTiming\models\user_behavior_svm_model.pkl')

# Define the route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    try:
        device_model = request.form['device_model']
        os = request.form['os']
        app_usage_time = float(request.form['app_usage_time'])
        screen_on_time = float(request.form['screen_on_time'])
        battery_drain = float(request.form['battery_drain'])
        num_apps_installed = int(request.form['num_apps_installed'])
        data_usage = float(request.form['data_usage'])
        age = int(request.form['age'])
        gender = request.form['gender']

        # Preprocess inputs to match model format
        user_input = np.array([[device_model, os, app_usage_time, screen_on_time, battery_drain, 
                                num_apps_installed, data_usage, age, gender]])

        # Predict using the model
        prediction = model.predict(user_input)
        result_text = interpret_prediction(prediction[0])

    except Exception as e:
        result_text = f"Error in input or prediction: {e}"

    # Display result
    return render_template('index.html', prediction_text=result_text)

def interpret_prediction(pred):
    """Interpret the prediction value with actionable feedback."""
    if pred == 5:
        return "Alert: High screen time detected! Informing parents."
    elif pred == 4:
        return "Warning: Consider reducing screen time."
    elif pred == 3:
        return "Notice: Moderate screen time."
    else:
        return "Good job! Reward for balanced screen time."

if __name__ == '__main__':
    app.run(debug=True)
