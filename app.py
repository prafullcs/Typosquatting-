from flask import Flask, render_template, request
import pickle

# Load your machine learning model and preprocessor
try:
    model = pickle.load(open('model.pkl', 'rb'))
    preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
    print("Model and preprocessor loaded successfully.")
except Exception as e:
    print(f"Error loading model or preprocessor: {e}")
    model, preprocessor = None, None

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/data', methods=['POST'])
def process_data():
    try:
        input_value = request.form['textInput']
        if preprocessor:
            # Transform the input using the loaded preprocessor
            input_transformed = preprocessor.transform([input_value])
            if model:
                # Predict using the loaded model
                prediction = model.predict(input_transformed)[0]
                result = 'Good TypoDomain' if prediction == 1 else 'Bad TypoDomain'
            else:
                result = 'Model is not loaded, unable to predict.'
        else:
            result = 'Preprocessor is not loaded, unable to preprocess input.'

        return render_template('result.html', result=result)

    except KeyError:
        return "Form field 'textInput' is missing", 400
    except ValueError as ve:
        return f"Error during prediction: {ve}", 500

if __name__ == '__main__':
    app.run(debug=True)
