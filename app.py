# Import dependencies
import streamlit as st
import tensorflow as tf
import keras
import numpy as np
import os
import time

from layers import L1Dist
# pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 opencv-python

model = tf.keras.models.load_model('siamesemodelv2.h5', custom_objects={'L1Dist':L1Dist})

def preprocess(file_path):
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)
        
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))
    # Scale image to be between 0 and 1 
    img = img / 255.0
        
    # Return image
    return img

# Verification function to verify person
def verify():
    per_completed = 0
    progress_bar = col2.progress(per_completed)
    
    # Specify thresholds
    detection_threshold = 0.99
    verification_threshold = 0.8

    # Build results array
    results = []

    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_images', image))
        
        # Make Predictions 
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
        per_completed+=2
        progress_bar.progress(per_completed)

    # Detection Threshold: Metric above which a prediciton is considered positive 
    detection = np.sum(np.array(results) > detection_threshold)
        
    # Verification Threshold: Proportion of positive predictions / total positive samples 
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
    verified = verification > verification_threshold

    # Set verification text 
    typed = 'Verified' if verified == True else 'Unverified '

    # Log out details
    #col2.markdown(results)
    # col2.markdown(detection)
    # col2.markdown(verification)
    # col2.markdown(verified)
    col2.markdown(typed)

        
    return results, verified, detection, verification

col1, col2, col3 = st.columns([1,2,1])
col1.markdown("# Welcome to my app! ")
col1.markdown("Some info")


img_file_buffer = col2.camera_input("Facial Recognition")

if img_file_buffer is not None:
    col2.success("Photo uploaded successfully!")

    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    #frame = cv2_img[120:120+250, 200:200+250, :]
    frame = cv2_img[29:29+250, 91:91+250, :]
    # st.write(type(frame))
    SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
    cv2.imwrite(SAVE_PATH, frame)

    # Check the type of cv2_img:
    # Should output: <class 'numpy.ndarray'>
    #st.write(type(frame))

    # Check the shape of cv2_img:
    # Should output shape: (height, width, channels)
    
    results, verified, detection, verification = verify()
    # for per_completed in range(50):
    #     time.sleep(0.05)
    #     progress_bar.progress(per_completed+1)

    with st.expander("Click to read more"):
        # st.markdown(results)
        st.markdown(detection)
        st.markdown(verification)
        st.markdown("*Streamlit* is **really** ***cool***.")
        st.markdown('''
    :red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in]
    :gray[pretty] :rainbow[colors].''')
        st.markdown("Here's a bouquet &mdash;\
            :tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:")

    col3.caption('This is a string that explains something above.')
        