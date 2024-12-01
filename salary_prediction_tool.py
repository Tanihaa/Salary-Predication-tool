import pandas as pd
import gradio as gr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from gtts import gTTS
import tempfile

# Load and preprocess your dataset
df = pd.read_csv('your_data.csv')  # Replace with the path to your dataset
x = df.drop("Salary", axis=1)
y = df['Salary']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=42)

# Train the model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Define the prediction function with formatted output and audio generation
def predict_salary(*features):
    # Convert input features to a DataFrame for prediction
    input_data = pd.DataFrame([features], columns=x.columns)
    prediction = lr.predict(input_data)
    salary_in_rupees = round(prediction[0])  # Round to whole number for rupees
    years_experience = features[0]  # Assuming "Years of Experience" is the first input feature
    output_text = f"Predicted Salary: ₹{salary_in_rupees} Rupees\nYears of Experience: {years_experience} years"

    # Convert the text to speech
    tts = gTTS(output_text)
    audio_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tts.save(audio_file.name)
    
    return output_text, audio_file.name

# Set up the Gradio interface
inputs = [gr.Number(label=col) for col in x.columns]  # Create input fields for each feature
output_text = gr.Textbox(label="Salary Prediction")
output_audio = gr.Audio(label="Audio Prediction")

# Launch the Gradio app with text and audio output
app = gr.Interface(fn=predict_salary, inputs=inputs, outputs=[output_text, output_audio], title="Salary Prediction")
app.launch()
