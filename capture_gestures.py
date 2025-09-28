# capture_gestures.py
import cv2
import os
import time

# Create folders for your gestures
classes = ['up', 'down', 'left', 'right']
base_dir = 'dataset/raw_images'
for cls in classes:
    os.makedirs(os.path.join(base_dir, cls), exist_ok=True)

# Webcam capture
cap = cv2.VideoCapture(0)
count = {cls: 0 for cls in classes}
current_class = None

print("\n🎯 GESTURE CAPTURE INSTRUCTIONS:")
print("Press 'u' = UP gesture (show thumbs up)")
print("Press 'd' = DOWN gesture (show fist)")
print("Press 'l' = LEFT gesture (show peace sign pointing left)")
print("Press 'r' = RIGHT gesture (show open palm)")
print("Press SPACE = Capture current gesture")
print("Press 'q' = Quit")
print("\nAim for 30-50 photos per gesture (we'll multiply them later)")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Display instructions
    if current_class:
        cv2.putText(frame, f"Capturing: {current_class} ({count[current_class]} photos)", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Press u/d/l/r to select gesture", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Gesture Capture - Your Own Photos', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('u'):
        current_class = 'up'
        print("Selected: UP gesture")
    elif key == ord('d'):
        current_class = 'down'
        print("Selected: DOWN gesture")
    elif key == ord('l'):
        current_class = 'left'
        print("Selected: LEFT gesture")
    elif key == ord('r'):
        current_class = 'right'
        print("Selected: RIGHT gesture")
    elif key == ord(' '):  # Space to capture
        if current_class:
            filename = f"{current_class}_{count[current_class]:03d}.jpg"
            filepath = os.path.join(base_dir, current_class, filename)
            cv2.imwrite(filepath, frame)
            count[current_class] += 1
            print(f"✅ Captured {filename}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Summary
print("\n📊 CAPTURE SUMMARY:")
for cls in classes:
    print(f"{cls}: {count[cls]} photos")
    
total = sum(count.values())
print(f"\n🎯 Total captured: {total} photos")
print("Folder: dataset/raw_images/")
if total < 100:
    print("💡 Tip: You can run this again later to add more photos!")