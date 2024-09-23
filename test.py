# # if uploaded_files:
# #     for uploaded_file in uploaded_files:
# #         uploaded_image = PILImage.open(uploaded_files)
# #         # Image = Image.open(uploaded_files)
# #         image_np = np.array(uploaded_image)
# #         st.image(image_np, caption='Uploaded Image:{uploaded_file.name}',use_column_width=True)
    
# #         results = model.predict(image_np)
    
# #         st.subheader("Detected Image : {uploaded_file.name}")
    
# #         # Check if results contain images
# #         if results:
# #             # Get the first detected image
# #             detected_image = results[0].plot()  # Use plot() to get the image with detections
        
# #             #Convert to PIL Image
# #             detected_image_pil = PILImage.fromarray(detected_image)
        
# #             # Display the detected image
# #             st.image(detected_image_pil, caption='Detected Image', use_column_width=True)
# #         else:
# #             st.error(f"No detections found in {uploaded_file.name}! Try with another image.")




# import streamlit as st
# import numpy as np
# from PIL import Image
# from ultralytics import YOLO
# from PIL import Image as PILImage

# ModelPath = r"C:\Users\IBE\OneDrive\Desktop\Harish\YOLO-V8\old1\yolov8n.pt"

# try:
#     model = YOLO(ModelPath)
# except:
#     st.error("error in loading model")


# st.title("Car Detection")
# st.title("using :blue[YOLO-V8 & Roboflow] :sunglasses:")

# st.header("Upload your Image")
# uploaded_files = st.file_uploader(
#     "Choose an Image file", type=["jpg", "jpeg", "png"]
# )

# if uploaded_files:
#     uploaded_images=[]
#     detected_images=[]
    
#     for uploaded_file in uploaded_files:
#         try:
#             uploaded_image = PILImage.open(uploaded_file)
#             image_np = np.array(uploaded_image)
#             uploaded_images.append(uploaded_image)
        
#             result = model.predict(image_np)
        
#             if result:
#                 detected_image = result[0].plot()
#                 detected_image_pil = Image.fromarray(detected_image)
#                 detected_images.append(detected_image_pil)
#             else:
#                 detected_images.append(None)
#         except Exception as e:
#             file_name = uploaded_file.name if hasattr(uploaded_file, 'name') else "an uploaded file"
#             st.error(f"Error processing {file_name}: {e}")
            
#     #Display uploaded Images
#     st.subheader("Uploaded Images")
#     num_cols =3
#     col = st.columns(num_cols)
    
#     for i,uploaded_image in enumerate(uploaded_images):
#         with col[i % num_cols]:
#             st.image(uploaded_image,caption=f"Uploaded Images{i+1}",use_column_width=True)
    
#     #Display detected Images        
#     st.subheader("Detected Images")
#     col = st.columns(num_cols)
    
#     for i,detected_image in enumerate(detected_images):
#         with col[i % num_cols]:
#             if detected_image is not None:
#                 st.image(detected_image,caption=f"Detected Images{i+1}",use_column_width=True)
#             else:
#                 st.write(f"No detection is found in uploaded images{i+1}!")
        
    
    
    
    
import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

ModelPath = r"C:\Users\IBE\OneDrive\Desktop\Harish\YOLO-V8\best.pt"
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