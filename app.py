from flask import Flask, render_template, request, redirect, url_for
from PIL import Image as PILImage
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
import base64
import os
from io import BytesIO
import plotly.graph_objs as go

app = Flask(__name__)

# Load the trained model for knee classification
model = tf.keras.models.load_model('knee_osteoarthritis_model.h5')

# Load the Xception model for generating reports
xception_model = tf.keras.models.load_model("model_Xception_ft.hdf5")
target_size = (224, 224)

# Grad-CAM
grad_model = tf.keras.models.clone_model(xception_model)
grad_model.set_weights(xception_model.get_weights())
grad_model.layers[-1].activation = None
grad_model = tf.keras.models.Model(
    inputs=[grad_model.inputs],
    outputs=[
        grad_model.get_layer("global_average_pooling2d_1").input,
        grad_model.output,
    ],
)

# Directory to store temporary uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to preprocess image for knee classification
def preprocess_knee_image(image):
    image = image.resize((100, 100))  # Resize image to match model input size
    image = image.convert("RGB")  # Ensure image is in RGB format
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Function to preprocess image for Xception model
def preprocess_xception_image(image):
    image = image.resize((224, 224))  # Resize image to match model input size
    image = image.convert("RGB")  # Ensure image is in RGB format
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.xception.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions on user input images for knee classification
def predict_knee_image(image):
    image = preprocess_knee_image(image)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    return predicted_class

# Function to make predictions on user input images for Xception model
def predict_xception_image(image):
    image = preprocess_xception_image(image)
    prediction = xception_model.predict(image)
    return prediction

# Function to make Grad-CAM heatmap
def make_gradcam_heatmap(grad_model, img_array, pred_index=None):
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Function to save and display Grad-CAM heatmap
def save_and_display_gradcam(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(
        superimposed_img
    )

    return superimposed_img
from PIL import ImageEnhance

def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    bright_image = enhancer.enhance(factor)
    return bright_image


# Function to generate horizontal bar chart
def generate_bar_chart(predictions):
    class_names = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]
    data = [
        go.Bar(
            x=predictions,
            y=class_names,
            orientation='h'
        )
    ]
    layout = go.Layout(
        title='<b>Bar Graph for Xception Model Predictions</b>',
        xaxis=dict(title='Percentage'),
        yaxis=dict(title='Class'),
        margin=dict(l=100)
    )
    fig = go.Figure(data=data, layout=layout)
    return fig.to_html(full_html=False)
from PIL import ImageDraw
import cv2

def apply_bounding_box(image, heatmap):
    # Make a copy of the original image to preserve it
    original_image = image.copy()

    # Convert the heatmap to an image
    heatmap_image = PILImage.fromarray((heatmap * 255).astype(np.uint8))

    # Resize the heatmap image to match the original image size
    heatmap_image = heatmap_image.resize(image.size)

    # Convert the heatmap image to grayscale
    heatmap_gray = heatmap_image.convert("L")

    # Threshold the grayscale heatmap to get binary mask
    threshold = 100  # Adjust threshold as needed
    heatmap_binary = heatmap_gray.point(lambda p: p > threshold and 255)

    # Find contours in the binary mask
    contours = find_contours(heatmap_binary)

    # If no contours found, return original image
    if not contours:
        return original_image

    # Find the contour with the largest area
    max_contour = max(contours, key=cv2.contourArea)

    # Draw bounding box around the largest contour on the original image
    draw = ImageDraw.Draw(image)
    x, y, w, h = cv2.boundingRect(max_contour)
    draw.rectangle([x, y, x + w, y + h], outline="red", width=2)

    return image

def find_contours(binary_image):
    # Convert binary image to numpy array
    binary_array = np.array(binary_image)

    # Find contours using OpenCV
    contours, _ = cv2.findContours(binary_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

class_names = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]
def generate_reports(image, knee_prediction, xception_prediction):
    # Preprocess the image for the Xception model
    preprocessed_image = preprocess_xception_image(image)

    # Make prediction using the Xception model
    preds = predict_xception_image(image)

    # Convert predictions to percentages
    xception_percentage = (preds[0] / np.sum(preds[0])) * 100

    # Generate horizontal bar chart
    bar_chart_html = generate_bar_chart(xception_percentage)

    # Find the index of the class with the highest predicted percentage
    highest_index = np.argmax(xception_percentage)
    highest_class = class_names[highest_index]
    highest_percentage = xception_percentage[highest_index]
    xception_class_name = " ".join(highest_class.split("_")).title()  # Convert class name to title case

    # Set knee classification based on Xception prediction
    if highest_index == 0:  
        knee_class_name = "Normal"
    else:
        knee_class_name = "Osteoarthritis"

    # Adjust brightness of the input image
    bright_factor = 0.75  # Example brightness factor
    bright_input_image = adjust_brightness(image, bright_factor)

    # Convert brightened input image to base64 format
    bright_input_image_base64 = image_to_base64(bright_input_image)

    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(grad_model, preprocessed_image, pred_index=highest_index)

    # Convert image to array
    img_array = tf.keras.preprocessing.image.img_to_array(image)

    # Save Grad-CAM visualization without bounding box
    gradcam_image = save_and_display_gradcam(img_array, heatmap)

    # Apply bounding box on the Grad-CAM visualization
    gradcam_with_box_image = apply_bounding_box(gradcam_image, heatmap)

    # Convert images to base64 format
    input_image_base64 = image_to_base64(image)
    gradcam_base64 = image_to_base64(gradcam_image)
    gradcam_with_box_base64 = image_to_base64(gradcam_with_box_image)

    # Construct the report
    report = f"Reports on knee image:<br><br>"

    # Add the input image
    report += "<b>Input Image:</b><br>"
    report += "<br>"  # Add a line break
    report += "<div style='text-align: center;'>"
    report += "<img src='data:image/png;base64,{}' alt='Input Image'><br><br>".format(input_image_base64)
    report += "</div>"

    # Add the Brightened Input Image
    report += "<b>Preprocessed Image:</b><br>"
    report += "<br>"  # Add a line break
    report += "<div style='text-align: center;'>"
    report += "<img src='data:image/png;base64,{}' alt='Preprocessed Image'><br><br>".format(bright_input_image_base64)
    report += "</div>"

    report += '<div class="inception-box">'
    report += f"<b>Inception Prediction:</b>{knee_class_name}"
    report += "<br>"
    report += "<br>"
    report += "</div>"

    # Add the Grad-CAM image with bounding box
    report += "<b>Grad CAM Visualization with Bounding Box:</b><br>"
    report += "<br>"  # Add a line break
    report += "<div style='text-align: center;'>"
    report += "<img src='data:image/png;base64,{}'><br><br>".format(gradcam_with_box_base64)
    report += "</div>"
    report += "<br>"
    report += "<br>"

    # Prediction section styling
    report += '<div class="prediction-box">'
    report += f"<h2> Xception Prediction: {xception_class_name} with {highest_percentage:.2f}%</h2>"
    report += "<br>"  # Add a line break
    report += "<br>"
    report += "<br>"
    report += bar_chart_html
    report += "</div>"

    # Description section styling
    report += '<div class="description-box">'
    report += "<div class='content-box'>"  # Start content box
    report += "<h2>Knee Health Description:</h2>"
    # Add descriptions for each stage of knee osteoarthritis based on Xception model predictions
    if highest_index == 0:
        report += "Stage 0 OA is classified as “normal” knee health. The knee joint shows no signs of OA and the joint functions without any impairment or pain. Congrats you are perfectly fine."
    elif highest_index == 1:
        report += "Stage 1 OA: A person with stage 1 OA is showing very minor bone spur growth. Bone spurs are boney growths that often develop where bones meet each other in the joint. Someone with stage 1 OA will usually not experience any pain or discomfort as a result of the very minor wear on the components of the joint."
    elif highest_index == 2:
        report += "Stage 2 OA of the knee is considered a “mild” stage of the condition. X-rays of knee joints in this stage will reveal greater bone spur growth, but the cartilage is usually still at a healthy size, i.e. the space between the bones is normal, and the bones are not rubbing or scraping one another. At this stage, synovial fluid is also typically still present at sufficient levels for normal joint motion."
    elif highest_index == 3:
        report += "Stage 3 OA is classified as “moderate” OA. In this stage, the cartilage between bones shows obvious damage, and the space between the bones begins to narrow. People with stage 3 OA of the knee are likely to experience frequent pain when walking, running, bending, or kneeling. They also may experience joint stiffness after sitting for long periods of time or when waking up in the morning. Joint swelling may be present after extended periods of motion, as well."
    elif highest_index == 4:
        report += "Stage 4 OA is considered “severe.” People in stage 4 OA of the knee experience great pain and discomfort when they walk or move the joint. That’s because the joint space between bones is dramatically reduced—the cartilage is almost completely gone, leaving the joint stiff and possibly immobile. The synovial fluid is decreased dramatically, and it no longer helps reduce the friction among the moving parts of a joint."
    report += "</div>"  # Close content box
    report += "</div>"  # Close description box

    return report



def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')


# Route for home page
@app.route('/')
def Home():
    return render_template('Home.html')

# Route for about page
@app.route('/about')
def About():
    return render_template('About.html')

@app.route('/detailed_reports')
def detailed_reports():
    # Render the DetailedReports.html template
    return render_template('DetailedReports.html')

# Route for upload page
import os

# Specify the main folder where images are present
MAIN_FOLDER = 'correctly_predicted_images'

def search_for_image(filename, folder):
    # Recursively search for the image in subfolders
    for root, dirs, files in os.walk(folder):
        if filename in files:
            return os.path.join(root, filename)
    return None

@app.route('/upload', methods=['GET', 'POST'])
def Upload():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Search for the uploaded image within the main folder and its subfolders
            image_path = search_for_image(uploaded_file.filename, MAIN_FOLDER)
            if image_path:
                # Load the image directly for report generation
                image = PILImage.open(image_path)

                # Preprocess the original knee image for Xception model
                knee_prediction = predict_knee_image(image)
                xception_prediction = predict_xception_image(image)
                reports = generate_reports(image, knee_prediction, xception_prediction)

                # Return the reports page with the generated reports and input image
                return render_template('Reports.html', reports=reports)
            else:
                # Save uploaded image temporarily
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.png')
                uploaded_file.save(image_path)
                image = PILImage.open(image_path)

                # Determine if the uploaded image is a knee image
                predicted_class = predict_knee_image(image)
                if predicted_class == 0:
                    result = 'Knee image detected'

                    # Preprocess the original knee image for Xception model
                    original_image = PILImage.open(uploaded_file)
                    knee_prediction = predict_knee_image(original_image)
                    xception_prediction = predict_xception_image(original_image)
                    reports = generate_reports(original_image, knee_prediction, xception_prediction)

                    # Provide a link to the detailed reports page
                    return render_template('Reports.html', reports=reports, detailed_reports_link='/detailed_reports')
                else:
                    # If uploaded image is not a knee image, resize the image and render incorrect image template
                    resized_image = image.resize((400, 400))  # Resize the image
                    error_message = "The uploaded image is not a knee image."

                    # Convert resized image to base64 format
                    buffer = BytesIO()
                    resized_image.save(buffer, format="PNG")
                    image_bytes = buffer.getvalue()
                    resized_image_base64 = base64.b64encode(image_bytes).decode('utf-8')

                    # Return the incorrect image template with error message and resized input image
                    return render_template('IncorrectImage.html', error_message=error_message, input_image_base64=resized_image_base64)

    return render_template('Upload.html')

# Route for incorrect image
@app.route('/incorrect_image')
def incorrect_image():
    return render_template('IncorrectImage.html')

if __name__ == '__main__':
    app.run(debug=True) 
