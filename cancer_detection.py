import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image as PILImage
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Table, TableStyle
import textwrap

# Load the trained model
st.set_page_config(
    page_title="Brain Tumor Prediction App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(r"C:\Users\HP\Downloads\tumor_model.h5")
    return model

model = load_model()

# Define class labels
class_labels = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']

# Tumor-specific recommendations
recommendations = {
    'Glioma': (
        "A glioma diagnosis requires immediate consultation with a neurologist or neurosurgeon for further evaluation. "
        "Advanced imaging, such as contrast-enhanced MRI, should be performed to determine tumor size, location, and grade. "
        "Treatment options may include surgical resection, radiation therapy, chemotherapy, or targeted therapy. "
        "Regular follow-ups with MRI scans and neurological assessments are essential to monitor tumor progression."
    ),
    'Meningioma': (
        "Meningiomas are generally slow-growing and often benign. Small, asymptomatic meningiomas may not require immediate intervention, "
        "and periodic imaging may be recommended. For symptomatic tumors, surgical resection is the primary treatment option. "
        "Radiation therapy may be used if complete removal is not feasible. Regular follow-ups with a neurologist or neurosurgeon are advised."
    ),
    'No tumor': (
        "No evidence of a brain tumor was detected. However, if symptoms persist, further evaluation by a neurologist is recommended. "
        "Additional tests such as EEG or vascular imaging may help identify underlying neurological conditions. "
        "A healthy lifestyle with a balanced diet, exercise, and stress management is encouraged."
    ),
    'Pituitary': (
        "Pituitary tumors can affect hormone production, requiring evaluation by an endocrinologist. "
        "A full endocrine workup and MRI scan should be performed. Treatment options depend on tumor type: non-functioning adenomas may be monitored, "
        "while hormone-secreting tumors may require medication, surgery, or radiation therapy. Long-term management includes hormone assessments."
    )
}

# Function to generate PDF report
def generate_pdf_report(patient_name, age, sex, symptoms, prediction, confidence, img, recommendation):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    # Define custom styles
    title_style = ParagraphStyle(name="TitleStyle", fontSize=18, spaceAfter=10, alignment=1, textColor=colors.darkblue)
    section_title_style = ParagraphStyle(name="SectionTitle", fontSize=14, spaceAfter=5, textColor=colors.darkred)
    text_style = ParagraphStyle(name="TextStyle", fontSize=11, spaceAfter=5, leading=16)  # 1.5x line spacing
    prediction_style = ParagraphStyle(name="PredictionStyle", fontSize=14, textColor=colors.green if prediction == "notumor" else colors.red)

    story = []

    # Title
    story.append(Paragraph("Brain MRI Tumor Classification Report", title_style))
    story.append(Spacer(1, 12))

    # Patient Details Table (without Symptoms)
    data = [
        ["Patient Name:", patient_name],
        ["Age:", str(age)],
        ["Sex:", sex],
        ["Prediction:", prediction],
        ["Confidence:", f"{confidence:.2f}%"]
    ]
    table = Table(data, colWidths=[120, 300])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey)
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    # Symptoms Section (Now Separate)
    if symptoms.strip():
        story.append(Paragraph("Symptoms", section_title_style))
        wrapped_symptoms = textwrap.fill(symptoms, width=80)
        story.append(Paragraph(wrapped_symptoms, text_style))
        story.append(Spacer(1, 10))

    # Recommendations
    story.append(Paragraph("Recommendations", section_title_style))
    wrapped_text = textwrap.fill(recommendation, width=80)
    story.append(Paragraph(wrapped_text, text_style))
    story.append(Spacer(1, 10))

    # MRI Scan Image
    img_path = "temp_image.jpg"
    img = img.resize((300, 200))
    img.save(img_path)
    story.append(ReportLabImage(img_path, width=300, height=200))

    doc.build(story)
    buffer.seek(0)
    return buffer

# Streamlit app layout
st.title("Brain Tumor Classification App")
st.write("Upload an MRI scan and the model will predict the tumor type. A detailed report will be generated.")

# Sidebar for patient details
with st.sidebar:
    st.title("Patient's  Information")
    patient_name = st.text_input("Name")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Sex", ["Male", "Female", "Other"])
    symptoms = st.text_area("Symptoms", height=100)

uploaded_file = st.file_uploader("Choose an MRI scan image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = PILImage.open(uploaded_file)

    # Display smaller image in Streamlit
    st.image(image, caption="Uploaded Image", width=300)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display prediction in color
    color = "green" if predicted_class == "No tumor" else "red"
    st.markdown(f"<h3 style='color: {color};'>Prediction: {predicted_class}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {color};'>Confidence: {confidence:.2f}%</h3>", unsafe_allow_html=True)

    # Generate PDF report
    pdf_buffer = generate_pdf_report(
        patient_name, age, sex, symptoms, predicted_class, confidence, image, recommendations[predicted_class]
    )

    st.download_button(
        label="Download Report as PDF",
        data=pdf_buffer,
        file_name=f"{patient_name}_MRI_report.pdf",
        mime="application/pdf"
    )
