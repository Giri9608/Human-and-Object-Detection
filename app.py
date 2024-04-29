import cv2
import streamlit as st
from ultralytics import YOLO
import PIL
import tempfile

# Replace the relative path to your weight file
model_path = 'C:/Users/girim/anaconda3/envs/HumanandObjectDetection/YoloV8 Tracking/weights/best.pt'

# Setting page layout
st.set_page_config(
    page_title="Object Detection",  # Setting page title
    page_icon="🤖",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded"    # Expanding sidebar by default
)

# Creating sidebar for images
with st.sidebar:
    st.header("Image Config")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader(
        "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    # Model Options
    confidence_img = float(st.slider(
        "Select Model Confidence (Image)", 25, 100, 40)) / 100

# Creating main page heading for images
st.title("Object Detection")

# Creating two columns on the main page for images
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(source_img)
        # Adding the uploaded image to the page with a caption
        st.image(uploaded_image,
                 caption="Uploaded Image",
                 use_column_width=True)

        try:
            model_img = YOLO(model_path)
            if st.sidebar.button('Detect Objects (Image)'):
                res = model_img.predict(uploaded_image, conf=confidence_img)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                with col2:
                    st.image(res_plotted,
                             caption='Detected Image',
                             use_column_width=True)
                    try:
                        with st.expander("Detection Results (Image)"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
        except Exception as ex:
            st.error(f"Unable to load model. Check the specified path: {model_path}")
            st.error(ex)

# Creating sidebar for videos
with st.sidebar:
    st.header("Video Config")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting videos
    source_vid = st.file_uploader("Upload a video...", type=("mp4", "avi"))

    # Model Options
    confidence_vid = float(st.slider(
        "Select Model Confidence (Video)", 25, 100, 40)) / 100

# Creating main page heading for videos
st.title("Object Detection (Video)")

# Processing video if uploaded
if source_vid is not None:
    with st.spinner('Detected video...'):
        if hasattr(source_vid, 'name'):  # If uploaded via file_uploader
            video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            video_path.write(source_vid.read())
            source_vid.close()
            video_path.close()
        else:
            video_path = source_vid  # If passed as a path

        try:
            model_vid = YOLO(model_path)
            vid_cap = cv2.VideoCapture(video_path.name)

            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    image = cv2.resize(image, (720, int(720*(9/16))))
                    res = model_vid.predict(image, conf=confidence_vid)
                    result_tensor = res[0].boxes
                    res_plotted = res[0].plot()
                    st_frame.image(res_plotted,
                                   caption='Detected Video',
                                   channels="BGR",
                                   use_column_width=True)
                else:
                    vid_cap.release()
                    break
        except Exception as ex:
            st.error(f"Unable to load model. Check the specified path: {model_path}")
            st.error(ex)

        try:
            if st.sidebar.button('Show Detection Results (Video)'):
                with st.expander("Detection Results (Video)"):
                    for box in result_tensor:
                        st.write(box.xywh)
        except Exception as ex:
            st.write("No video is uploaded yet!")
