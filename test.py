import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

ModelPath = os.path.join(os.path.dirname(__file__), "best.pt")
model = YOLO(ModelPath)

st.title("Car Detection")
st.title("using :blue[YOLO-V8 & Roboflow] :sunglasses:")

st.header("Upload your Image")
uploaded_files = st.file_uploader(
    "Choose an Image file", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)


if uploaded_files:
    uploaded_images = [Image.open(uploaded_file).convert("RGB") for uploaded_file in uploaded_files]

    # Display uploaded Images
    st.subheader("Uploaded Images")
    num_cols = 3
    col = st.columns(num_cols)

    for i, uploaded_image in enumerate(uploaded_images):
        with col[i % num_cols]:
            st.image(uploaded_image, caption=f"Uploaded Image {i + 1}", use_column_width=True)

    # Add a button to proceed with detection
    if st.button("Proceed with Detection"):
        detected_images = []
        
        for uploaded_image in uploaded_images:
            image_np = np.array(uploaded_image)

            try:
                result = model.predict(image_np)

                if result:
                    detected_image = result[0].plot()
                    detected_image_pil = Image.fromarray(detected_image)
                    detected_images.append(detected_image_pil)
                else:
                    detected_images.append(None)
                    
            except Exception as e:
                st.error(f"Error processing an uploaded image: {e}")
                
        #progress bar 
        import time
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
            my_bar.success("Sucessfully processed all uploaded images!")
                
        # Display detected Images        
        st.subheader("Detected Images")
        col = st.columns(num_cols)

        for i, detected_image in enumerate(detected_images):
            with col[i % num_cols]:
                if detected_image is not None:
                    st.image(detected_image, caption=f"Detected Image {i + 1}", use_column_width=True)
                else:
                    st.write(f"No detections found in uploaded image {i + 1}!")
