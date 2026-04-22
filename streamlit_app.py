import streamlit as st
from PIL import Image
from utils.detector import YOLOModel
from utils.visualization import draw_boxes

st.set_page_config(page_title="Football Player Detection", page_icon="⚽", layout="wide")

st.title("Football Player Detection ⚽")
st.markdown("Upload an image to detect football players, referees, goalkeepers, and the ball.")

@st.cache_resource
def load_model():
    return YOLOModel(model_path="model/best.pt", labels_path="model/labels.txt")

try:
    with st.spinner("Loading model..."):
        model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

col1, col2 = st.columns(2)

with st.sidebar:
    st.header("Settings")
    conf_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    st.header("Input")
    use_demo = st.checkbox("Use Demo Image", value=True)
    uploaded_file = st.file_uploader("Or upload your own image...", type=["jpg", "jpeg", "png"])

image = None
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Error loading uploaded image: {e}")
elif use_demo:
    try:
        image = Image.open("assets/demo.png").convert("RGB")
    except Exception as e:
        st.error(f"Error loading demo image: {e}")

if image is not None:
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    
    if st.button("Run Detection", type="primary"):
        with st.spinner("Running detection..."):
            try:
                detections = model.predict(image, conf=conf_threshold)
                
                if not detections:
                    st.warning("No objects detected! Try lowering the confidence threshold.")
                else:
                    annotated_image = draw_boxes(image, detections)
                    with col2:
                        st.subheader("Annotated Image")
                        st.image(annotated_image, use_container_width=True)
                        st.success(f"Detected {len(detections)} objects!")
                        
                        # Display detection breakdown
                        st.write("### Detection Details")
                        for det in detections:
                            st.write(f"- **{det['class_name']}**: {det['confidence']:.2f} ({det['box'][0]:.1f}, {det['box'][1]:.1f}, {det['box'][2]:.1f}, {det['box'][3]:.1f})")
            except Exception as e:
                st.error(f"Error during inference: {e}")
