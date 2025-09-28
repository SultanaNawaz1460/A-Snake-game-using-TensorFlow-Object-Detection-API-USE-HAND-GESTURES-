# augment_images.py - FIXED VERSION
import cv2
import os
import numpy as np
import random
import glob

def augment_image(image):
    """Create multiple variations of one image"""
    augmented = []
    
    # Original image
    augmented.append(image)
    
    try:
        # Brightness variations
        for alpha in [0.7, 0.9, 1.1, 1.3]:
            bright = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
            augmented.append(bright)
        
        # Small rotations
        for angle in [-15, -5, 5, 15]:
            h, w = image.shape[:2]
            center = (w//2, h//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h))
            augmented.append(rotated)
        
        # Flip horizontally
        flipped = cv2.flip(image, 1)
        augmented.append(flipped)
        
        # Add different types of blur
        blurred1 = cv2.GaussianBlur(image, (3, 3), 0)
        blurred2 = cv2.GaussianBlur(image, (5, 5), 0)
        augmented.extend([blurred1, blurred2])
        
        # Add noise
        noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
        noisy = cv2.add(image, noise)
        augmented.append(noisy)
        
    except Exception as e:
        print(f"⚠️  Augmentation warning: {e}")
    
    return augmented

def split_train_test(files, test_ratio=0.2):
    """Simple train/test split without sklearn dependency"""
    random.shuffle(files)
    split_index = int(len(files) * (1 - test_ratio))
    return files[:split_index], files[split_index:]

def main():
    print("🔄 STARTING DATA AUGMENTATION...")
    
    classes = ['up', 'down', 'left', 'right']
    input_dir = 'dataset/raw_images'
    output_dir = 'dataset/augmented'
    
    # Create output directories
    for cls in classes:
        os.makedirs(os.path.join(output_dir, 'train', cls), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test', cls), exist_ok=True)
    
    total_original = 0
    total_augmented = 0
    
    for cls in classes:
        print(f"\n📁 Processing {cls} gestures...")
        
        # Get all original images
        class_dir = os.path.join(input_dir, cls)
        if not os.path.exists(class_dir):
            print(f"⚠️  Directory not found: {class_dir}")
            continue
            
        image_files = glob.glob(os.path.join(class_dir, '*.jpg'))
        if not image_files:
            print(f"⚠️  No JPG images found in {class_dir}")
            print("💡 Run capture_gestures.py first to capture your hand gestures!")
            continue
            
        total_original += len(image_files)
        print(f"📸 Found {len(image_files)} original images")
        
        # Split into train/test
        train_files, test_files = split_train_test(image_files, test_ratio=0.2)
        
        # Process training images (augment them)
        train_count = 0
        for img_path in train_files:
            try:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"⚠️  Could not read image: {img_path}")
                    continue
                
                # Create augmented versions
                augmented_images = augment_image(image)
                
                # Save each augmented version
                for i, aug_img in enumerate(augmented_images):
                    filename = f"{cls}_aug_{train_count:04d}_{i}.jpg"
                    output_path = os.path.join(output_dir, 'train', cls, filename)
                    success = cv2.imwrite(output_path, aug_img)
                    if success:
                        train_count += 1
                    else:
                        print(f"⚠️  Failed to save: {filename}")
                        
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
        
        # Process test images (keep original)
        test_count = 0
        for img_path in test_files:
            try:
                image = cv2.imread(img_path)
                if image is not None:
                    filename = f"{cls}_test_{test_count:04d}.jpg"
                    output_path = os.path.join(output_dir, 'test', cls, filename)
                    if cv2.imwrite(output_path, image):
                        test_count += 1
            except Exception as e:
                print(f"❌ Error saving test image {img_path}: {e}")
        
        print(f"✅ {cls}: {len(image_files)} originals → {train_count + test_count} total images")
        total_augmented += (train_count + test_count)
    
    if total_original > 0:
        print(f"\n🎉 AUGMENTATION COMPLETE!")
        print(f"📊 Original photos: {total_original}")
        print(f"📊 After augmentation: {total_augmented}")
        print(f"📈 Multiplied by: {total_augmented/total_original:.1f}x")
        print(f"📁 Output folder: {output_dir}/")
        
        # Show folder structure
        print(f"\n📂 FOLDER STRUCTURE:")
        for cls in classes:
            train_dir = os.path.join(output_dir, 'train', cls)
            test_dir = os.path.join(output_dir, 'test', cls)
            train_count = len(glob.glob(os.path.join(train_dir, '*.jpg'))) if os.path.exists(train_dir) else 0
            test_count = len(glob.glob(os.path.join(test_dir, '*.jpg'))) if os.path.exists(test_dir) else 0
            print(f"  {cls}: {train_count} training + {test_count} test images")
    else:
        print(f"\n❌ No images found. Please run capture_gestures.py first!")

if __name__ == "__main__":
    main()