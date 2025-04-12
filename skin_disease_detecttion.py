import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import os

# Define constants
IMG_SIZE = 224
BATCH_SIZE = 32
dataset_path = r"E:\Project\skin_disease_dataset"  # Update with your dataset path
MODEL_PATH = "skin_disease_model.h5"  # Path to save/load the model
CLASS_INDICES_PATH = "class_indices.npy"  # Path to save/load class indices

# Dictionary for prevention, cure suggestions, and related website links
suggestions = {
    "Acne": {
        "prevention": "1. Keep your skin clean by washing it twice a day with a gentle cleanser.\n"
                      "2. Avoid touching your face with dirty hands.\n"
                      "3. Use non-comedogenic skincare and makeup products.\n"
                      "4. Manage stress, as it can worsen acne.",
        "cure": "1. Use over-the-counter treatments containing benzoyl peroxide or salicylic acid.\n"
                "2. For severe acne, consult a dermatologist for prescription medications like retinoids or antibiotics.",
        "link": "https://www.mayoclinic.org/diseases-conditions/acne/symptoms-causes/syc-20368047"
    },
    "Eczema": {
        "prevention": "1. Moisturize your skin regularly to prevent dryness.\n"
                      "2. Avoid irritants like harsh soaps, detergents, and fragrances.\n"
                      "3. Wear soft, breathable fabrics like cotton.\n"
                      "4. Manage stress, as it can trigger flare-ups.",
        "cure": "1. Use corticosteroid creams or ointments to reduce inflammation.\n"
                "2. Take antihistamines to relieve itching.\n"
                "3. For severe cases, consult a dermatologist for advanced treatments like immunosuppressants.",
        "link": "https://www.mayoclinic.org/diseases-conditions/atopic-dermatitis-eczema/symptoms-causes/syc-20353273"
    },
    "Melanoma": {
        "prevention": "1. Avoid excessive sun exposure, especially during peak hours (10 AM to 4 PM).\n"
                      "2. Use broad-spectrum sunscreen with SPF 30 or higher.\n"
                      "3. Wear protective clothing, hats, and sunglasses.\n"
                      "4. Avoid tanning beds and lamps.",
        "cure": "1. Early detection is crucial. Consult a dermatologist immediately if you notice suspicious moles or skin changes.\n"
                "2. Treatment options include surgery, radiation therapy, immunotherapy, and targeted therapy.",
        "link": "https://www.mayoclinic.org/diseases-conditions/melanoma/symptoms-causes/syc-20374884"
    }
}

# Function to build and compile the model
def build_model(num_classes):
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False  # Freeze base model

    # Custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)

    # Define model
    model = Model(inputs=base_model.input, outputs=output_layer)

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model
def train_model():
    # Image data augmentation
    data_gen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    # Load training and validation data
    train_data = data_gen.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_data = data_gen.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Save class indices
    np.save(CLASS_INDICES_PATH, train_data.class_indices)

    # Build the model
    model = build_model(num_classes=len(train_data.class_indices))

    # Define checkpoint to save best model
    checkpoint_callback = ModelCheckpoint(
        MODEL_PATH,  # Save in Keras format
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1
    )

    # Train model with checkpointing
    history = model.fit(train_data, validation_data=val_data, epochs=20, callbacks=[checkpoint_callback])
    return model

# Function to preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to model input size
    img = img.astype("float32") / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Expand to match batch size
    return img

# Function to predict skin disease
def predict_skin_disease(image_path, model):
    img = preprocess_image(image_path)  # Preprocess image
    prediction = model.predict(img)  # Get predictions
    class_index = np.argmax(prediction)  # Get class with highest probability

    # Load class indices
    class_indices = np.load(CLASS_INDICES_PATH, allow_pickle=True).item()
    class_names = list(class_indices.keys())  # Get class names
    return class_names[class_index]

# Streamlit UI for web app
def main():
    st.title("Skin Disease Detection")

    # Check if the model and class indices exist
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_INDICES_PATH):
        st.write("Model or class indices not found. Training the model...")
        train_model()
        st.write("Model trained and saved.")

    # Load the saved model
    model = load_model(MODEL_PATH)

    uploaded_file = st.file_uploader("Upload an image of a skin lesion", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Classifying...")

        # Save temporary image
        temp_image_path = "temp_image.jpg"
        image.save(temp_image_path)

        # Make prediction
        prediction = predict_skin_disease(temp_image_path, model)
        st.write(f"Prediction: {prediction}")

        # Display suggestions and related website link
        if prediction in suggestions:
            st.write("### Prevention Tips")
            st.write(suggestions[prediction]["prevention"])
            st.write("### Cure Suggestions")
            st.write(suggestions[prediction]["cure"])
            st.write("### Learn More")
            st.write(f"For more information about {prediction}, visit: [{prediction.capitalize()} Information]({suggestions[prediction]['link']})")
        else:
            st.write("No specific prevention or cure suggestions available for this condition.")

if __name__ == "__main__":
    main()