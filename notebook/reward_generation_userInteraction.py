import joblib
import random

# Load the pre-trained model for User Behavior Prediction
svm_model_user_behavior = joblib.load(r'D:\ChildTiming\models\user_behavior_svm_model.pkl')
def get_user_input():
    user_data = {}
    print("Please enter the following details:")
    user_data['Device Model'] = int(input("Enter device model (e.g., 1 for phone, 2 for tablet): "))
    user_data['Operating System'] = int(input("Enter operating system (0 for Android, 1 for iOS): "))
    user_data['App Usage Time (min/day)'] = float(input("Enter daily app usage time in minutes: "))
    user_data['Screen On Time (hours/day)'] = float(input("Enter daily screen on time in hours: "))
    user_data['Battery Drain (mAh/day)'] = float(input("Enter daily battery drain in mAh: "))
    user_data['Number of Apps Installed'] = int(input("Enter the number of apps installed on the device: "))
    user_data['Data Usage (MB/day)'] = float(input("Enter daily data usage in MB: "))
    user_data['Age'] = int(input("Enter the user's age: "))
    user_data['Gender'] = int(input("Enter gender (0 for male, 1 for female): "))

    return user_data

def give_rewards_or_suggestions(user_data):
    user_features = [
        user_data['Device Model'],
        user_data['Operating System'],
        user_data['App Usage Time (min/day)'],
        user_data['Screen On Time (hours/day)'],
        user_data['Battery Drain (mAh/day)'],
        user_data['Number of Apps Installed'],
        user_data['Data Usage (MB/day)'],
        user_data['Age'],
        user_data['Gender']
    ]

    prediction = svm_model_user_behavior.predict([user_features])[0]
    print(f"Predicted user behavior class: {prediction}")

    if prediction == 5:
        print("Alert: Your child is highly engaged with the device. Please ensure a healthy balance of screen time. Informing parents for review.")
        print("Parents are notified. Please make sure to engage your child in offline activities.")
    elif prediction == 4:
        print("Warning: The screen time is a bit high. Consider reducing it to promote a more balanced lifestyle.")
        print("Take short breaks and spend time engaging in creative offline activities.")
    elif prediction == 3:
        print("Moderate warning: Try to balance screen time with more offline activities. Your child is moderately engaged.")
        print("We recommend taking breaks and engaging in physical activities.")
    else:
        coins = random.randint(5, 20)  # Reward coins for engagement below 3
        print(f"Thank you for maintaining a good balance! You've earned {coins} coins as a reward!")
        print("Here are some suggestions to improve engagement with offline activities:")
        print("- Encourage your child to engage in educational games or reading.")
        print("- Try balancing screen time with physical outdoor activities.")
        print(f"Total coins earned: {coins}")

# Main function to interact with the user
def main():
    user_data = get_user_input()
    give_rewards_or_suggestions(user_data)

# Run the program
if __name__ == "__main__":
    main()
