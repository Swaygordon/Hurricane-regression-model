import joblib
import pandas as pd

# Load the trained models
try:
    regression_model = joblib.load("hurricane_regression_model.pkl")
    classification_model = joblib.load("hurricane_classification_model.pkl")
    print("Hurricane ZERO: Project Prototype One")
    print("Models loaded successfully!")
    print("\nNote model was trained with ONLY Atlantic Data Parameters, Recommend using Parameters from that Region!")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

def categorize_hurricane(predicted_wind_speed):
    if predicted_wind_speed < 39:
        return "Tropical Depression", "Stay informed about weather updates. Secure loose outdoor items. Prepare an emergency kit."
    elif 39 <= predicted_wind_speed <= 73:
        return "Tropical Storm", "Avoid unnecessary travel. Charge electronic devices. Reinforce windows and doors."
    elif 74 <= predicted_wind_speed <= 95:
        return "Category 1", "Stay indoors and away from windows. Have a battery-powered radio. Consider evacuating flood-prone areas."
    elif 96 <= predicted_wind_speed <= 110:
        return "Category 2", "Evacuate if advised. Turn off gas, electricity, and water if instructed. Move to a safe shelter."
    elif 111 <= predicted_wind_speed <= 129:
        return "Category 3", "Expect significant damage. Evacuate if possible. Keep emergency supplies accessible."
    elif 130 <= predicted_wind_speed <= 156:
        return "Category 4", "Seek sturdy shelter or evacuate immediately. Stay away from low-lying areas. Prepare for extended power outages."
    else:
        return "Category 5", "Evacuate if possible—these storms are catastrophic. Follow emergency broadcasts closely. Have a survival plan in place."

while True:
    try:
        # User Input for Hurricane Parameters
        print("\nEnter real-world hurricane parameters: \nNote Pressure in MilliBars (mb), Temperature in Celsius (°C) & Prediction of Wind Speed in Miles per Hour (MPH)")
        latitude = float(input("Latitude: "))
        longitude = float(input("Longitude: "))
        pressure = float(input("Pressure (mb): "))
        temperature = float(input("Temperature (°C): "))

        # Convert Input to DataFrame for Regression Model
        regression_input = pd.DataFrame([[latitude, longitude, pressure, temperature]], 
                                        columns=['Latitude', 'Longitude', 'Pressure', 'Temperature'])

        # Predict Wind Speed
        predicted_wind_speed = regression_model.predict(regression_input)[0]
        predicted_category, safety_tips = categorize_hurricane(predicted_wind_speed)

        # Display Results
        print("\n Hurricane Prediction Results ")
        print(f" Predicted Wind Speed: {predicted_wind_speed:.2f} mph")
        print(f" Predicted Hurricane Category: {predicted_category}")
        print(f" Safety Tips: {safety_tips}")

    except ValueError:
        print("Invalid input. Please enter numeric values.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    # Exit condition
    exit_choice = input("\nDo you want to exit? (yes/no): ").strip().lower()
    if exit_choice == 'yes':
        print("Exiting the program. Stay safe!")
        break