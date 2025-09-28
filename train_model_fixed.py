# train_model_fixed.py - Complete fixed training script
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def check_data_availability():
    """Check if training data is available"""
    train_dir = 'dataset/augmented/train'
    test_dir = 'dataset/augmented/test'
    classes = ['up', 'down', 'left', 'right']
    
    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found: {train_dir}")
        print("Please run the following steps first:")
        print("1. python capture_gestures.py")
        print("2. python augment_images.py")
        return False
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found: {test_dir}")
        print("Please run augment_images.py first")
        return False
    
    # Check each class directory
    total_train_images = 0
    total_test_images = 0
    
    for class_name in classes:
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        
        if os.path.exists(train_class_dir):
            train_count = len([f for f in os.listdir(train_class_dir) if f.endswith('.jpg')])
            total_train_images += train_count
            print(f"Training {class_name}: {train_count} images")
        else:
            print(f"Warning: {train_class_dir} not found")
        
        if os.path.exists(test_class_dir):
            test_count = len([f for f in os.listdir(test_class_dir) if f.endswith('.jpg')])
            total_test_images += test_count
            print(f"Test {class_name}: {test_count} images")
        else:
            print(f"Warning: {test_class_dir} not found")
    
    print(f"Total training images: {total_train_images}")
    print(f"Total test images: {total_test_images}")
    
    if total_train_images < 100:
        print("Warning: Very few training images. Consider capturing more gestures.")
        return input("Continue with training anyway? (y/n): ").lower() == 'y'
    
    return True

def create_enhanced_model(input_shape, num_classes):
    """Create enhanced CNN model for gesture recognition"""
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Dense Layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_gesture_model():
    """Complete training function with error handling"""
    print("Starting Enhanced Gesture Recognition Training")
    print("=" * 50)
    
    # Configuration
    IMG_SIZE = (64, 64)
    BATCH_SIZE = 32
    EPOCHS = 25
    CLASSES = ['up', 'down', 'left', 'right']
    
    train_dir = 'dataset/augmented/train'
    test_dir = 'dataset/augmented/test'
    
    # Check data availability
    if not check_data_availability():
        return False
    
    print("\nCreating data generators...")
    
    # Enhanced data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Only rescaling for validation/test
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    try:
        # Create data generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            classes=CLASSES,
            shuffle=True,
            seed=42
        )
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            classes=CLASSES,
            shuffle=False,
            seed=42
        )
        
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {test_generator.samples}")
        print(f"Classes found: {list(train_generator.class_indices.keys())}")
        
    except Exception as e:
        print(f"Error creating data generators: {e}")
        print("Please check your dataset structure")
        return False
    
    # Create model
    print("\nBuilding enhanced CNN model...")
    model = create_enhanced_model((*IMG_SIZE, 3), len(CLASSES))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001, decay=1e-6),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model Architecture:")
    model.summary()
    
    # Create callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        ModelCheckpoint(
            'best_gesture_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\nStarting training for up to {EPOCHS} epochs...")
    try:
        history = model.fit(
            train_generator,
            steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
            epochs=EPOCHS,
            validation_data=test_generator,
            validation_steps=max(1, test_generator.samples // BATCH_SIZE),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        model.save('gesture_cnn_model.h5')
        print("Final model saved as 'gesture_cnn_model.h5'")
        
        # Evaluate model
        print("\nEvaluating model...")
        test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
        
        # Print results
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED!")
        print("=" * 50)
        print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Final Test Loss: {test_loss:.4f}")
        
        # Training history summary
        if history.history:
            final_train_acc = history.history['accuracy'][-1]
            final_val_acc = history.history.get('val_accuracy', [0])[-1]
            print(f"Final Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
            print(f"Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
        
        # Performance evaluation
        print("\nPERFORMANCE ASSESSMENT:")
        if test_accuracy > 0.9:
            print("Excellent! Model should work very well for gesture control.")
        elif test_accuracy > 0.8:
            print("Good performance. Should work well for most gestures.")
        elif test_accuracy > 0.7:
            print("Moderate performance. May need clearer gestures during gaming.")
        elif test_accuracy > 0.6:
            print("Fair performance. Consider capturing more diverse training data.")
        else:
            print("Low performance. Please recapture gestures with better lighting/clarity.")
        
        # Per-class evaluation
        print("\nTesting individual gesture recognition...")
        test_generator.reset()
        predictions = model.predict(test_generator, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes[:len(predicted_classes)]
        
        print("Per-class accuracy:")
        for i, class_name in enumerate(CLASSES):
            class_mask = (true_classes == i)
            if np.sum(class_mask) > 0:
                class_accuracy = np.mean(predicted_classes[class_mask] == true_classes[class_mask])
                print(f"  {class_name.upper()}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
            else:
                print(f"  {class_name.upper()}: No test samples")
        
        print(f"\nModel files created:")
        print(f"- gesture_cnn_model.h5 (final model)")
        print(f"- best_gesture_model.h5 (best during training)")
        
        return True
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model():
    """Test the trained model with webcam"""
    print("Testing trained model...")
    
    try:
        from tensorflow.keras.models import load_model
        import cv2
        
        model = load_model('gesture_cnn_model.h5')
        print("Model loaded successfully!")
        
        # Test with webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera for testing")
            return
        
        print("Show gestures to camera. Press 'q' to quit test.")
        classes = ['up', 'down', 'left', 'right']
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Extract ROI (same as game will use)
            roi = frame[50:300, 50:300]
            
            # Preprocess
            img = cv2.resize(roi, (64, 64))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Predict
            pred = model.predict(img, verbose=0)
            idx = np.argmax(pred[0])
            confidence = pred[0][idx]
            
            # Draw results
            cv2.rectangle(frame, (50, 50), (300, 300), (0, 255, 0), 2)
            cv2.putText(frame, f"Prediction: {classes[idx]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {confidence:.3f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Model Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Model test completed!")
        
    except Exception as e:
        print(f"Model testing failed: {e}")

def main():
    """Main function"""
    print("Gesture Recognition Model Training")
    print("Please ensure you have completed:")
    print("1. Captured gesture images (capture_gestures.py)")
    print("2. Augmented the dataset (augment_images.py)")
    print()
    
    choice = input("Proceed with training? (y/n): ")
    if choice.lower() != 'y':
        print("Training cancelled.")
        return
    
    success = train_gesture_model()
    
    if success:
        print("\nTraining successful!")
        test_choice = input("Test the model with webcam? (y/n): ")
        if test_choice.lower() == 'y':
            test_model()
        
        print("\nYou can now run your snake game!")
        print("python snake_game.py")
    else:
        print("\nTraining failed. Please check the errors above.")

if __name__ == "__main__":
    main()