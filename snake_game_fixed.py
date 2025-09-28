# snake_game_fixed.py - Complete fixed snake game with all features
import cv2
import numpy as np
import time
import os
from collections import deque
from tensorflow.keras.models import load_model

# Sound support - optional
try:
    import pygame
    SOUND_AVAILABLE = True
    pygame.mixer.init()
except ImportError:
    SOUND_AVAILABLE = False

class SoundManager:
    """Handles sound effects and background music"""
    
    def __init__(self):
        self.enabled = SOUND_AVAILABLE
        self.music_playing = False
        
        if self.enabled:
            self.create_sounds()
    
    def create_sounds(self):
        """Create simple sound effects"""
        try:
            # Create eating sound
            sample_rate = 22050
            duration = 0.1
            frames = int(duration * sample_rate)
            
            # Simple beep sound
            frequency = 800
            arr = np.zeros((frames, 2), dtype=np.int16)
            for i in range(frames):
                wave = 4096 * np.sin(frequency * 2 * np.pi * i / sample_rate)
                fade = 1.0 - (i / frames)  # Fade out
                arr[i] = [wave * fade, wave * fade]
            
            self.eat_sound = pygame.sndarray.make_sound(arr)
        except Exception as e:
            print(f"Sound creation failed: {e}")
            self.enabled = False
    
    def play_eat_sound(self):
        """Play eating sound"""
        if self.enabled and hasattr(self, 'eat_sound'):
            try:
                self.eat_sound.play()
            except:
                pass

class ImprovedGestureSmoothing:
    """Enhanced gesture smoothing to prevent direction locking"""
    
    def __init__(self, window_size=5, confidence_threshold=0.6):
        self.window_size = window_size
        self.base_confidence_threshold = confidence_threshold
        self.gesture_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)
        self.current_stable_gesture = None
        self.last_gesture_change = 0
        self.min_change_interval = 0.2  # Faster response
        
        # Dynamic confidence adjustment
        self.confidence_threshold = confidence_threshold
        
    def add_prediction(self, gesture, confidence):
        """Add new prediction with dynamic threshold adjustment"""
        self.gesture_history.append(gesture)
        self.confidence_history.append(confidence)
        
        # Adjust confidence threshold based on recent performance
        if len(self.confidence_history) >= self.window_size:
            avg_confidence = np.mean(list(self.confidence_history))
            if avg_confidence > 0.8:
                self.confidence_threshold = min(0.75, self.base_confidence_threshold + 0.1)
            elif avg_confidence < 0.5:
                self.confidence_threshold = max(0.4, self.base_confidence_threshold - 0.1)
            else:
                self.confidence_threshold = self.base_confidence_threshold
    
    def get_stable_gesture(self):
        """Get stable gesture with improved responsiveness"""
        if len(self.gesture_history) < 3:  # Faster initial response
            return self.current_stable_gesture
        
        current_time = time.time()
        
        # Only check recent predictions for faster response
        recent_window = min(3, len(self.gesture_history))
        recent_gestures = list(self.gesture_history)[-recent_window:]
        recent_confidences = list(self.confidence_history)[-recent_window:]
        
        # Filter by confidence
        valid_predictions = []
        for i in range(len(recent_gestures)):
            if recent_confidences[i] >= self.confidence_threshold:
                valid_predictions.append((recent_gestures[i], recent_confidences[i]))
        
        if not valid_predictions:
            return self.current_stable_gesture
        
        # Get the most confident prediction
        best_gesture, best_confidence = max(valid_predictions, key=lambda x: x[1])
        
        # Count how many times this gesture appears
        gesture_count = sum(1 for g, c in valid_predictions if g == best_gesture)
        
        # More lenient stability requirement
        required_count = max(1, len(valid_predictions) // 2)
        
        if gesture_count >= required_count:
            time_since_change = current_time - self.last_gesture_change
            
            # Allow change if different gesture and enough time passed OR very confident
            if (best_gesture != self.current_stable_gesture and 
                (time_since_change >= self.min_change_interval or best_confidence > 0.85)):
                self.current_stable_gesture = best_gesture
                self.last_gesture_change = current_time
            elif best_gesture == self.current_stable_gesture:
                # Same gesture, keep it
                pass
        
        return self.current_stable_gesture
    
    def get_debug_info(self):
        """Get debugging information"""
        return {
            'threshold': f"{self.confidence_threshold:.3f}",
            'recent_gestures': list(self.gesture_history)[-3:],
            'recent_confidences': [f"{c:.3f}" for c in list(self.confidence_history)[-3:]],
            'stable_gesture': self.current_stable_gesture or 'None'
        }

class SnakeGame:
    """Enhanced snake game with wrap-around walls and smooth movement"""
    
    def __init__(self, width=600, height=600, cell_size=20):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.cols = width // cell_size
        self.rows = height // cell_size
        self.sound_manager = SoundManager()
        self.speed = 0.25  # Default speed
        self.reset()
    
    def set_speed(self, speed_level):
        """Set game speed (1=Slow, 2=Medium, 3=Fast)"""
        speed_map = {1: 0.4, 2: 0.25, 3: 0.15}
        self.speed = speed_map.get(speed_level, 0.25)
    
    def reset(self):
        """Reset game state"""
        center_x = (self.cols // 2) * self.cell_size
        center_y = (self.rows // 2) * self.cell_size
        self.snake = [(center_x, center_y)]
        self.direction = 'right'
        self.pending_direction = None
        self.food = self.spawn_food()
        self.score = 0
        self.game_over = False
        self.paused = False
        self.last_direction_change = 0
    
    def spawn_food(self):
        """Spawn food at random location"""
        while True:
            food_x = np.random.randint(0, self.cols) * self.cell_size
            food_y = np.random.randint(0, self.rows) * self.cell_size
            food = (food_x, food_y)
            if food not in self.snake:
                return food
    
    def is_opposite_direction(self, new_direction, current_direction):
        """Check if directions are opposite"""
        opposites = {
            'up': 'down', 'down': 'up',
            'left': 'right', 'right': 'left'
        }
        return opposites.get(current_direction) == new_direction
    
    def set_direction(self, new_direction):
        """Set new direction with validation"""
        current_time = time.time()
        
        if (new_direction and 
            new_direction != self.direction and 
            not self.is_opposite_direction(new_direction, self.direction) and
            current_time - self.last_direction_change >= 0.1):
            
            self.pending_direction = new_direction
            self.last_direction_change = current_time
            return True
        return False
    
    def wrap_position(self, x, y):
        """Handle wrap-around walls"""
        # Wrap horizontally
        if x < 0:
            x = self.width - self.cell_size
        elif x >= self.width:
            x = 0
        
        # Wrap vertically
        if y < 0:
            y = self.height - self.cell_size
        elif y >= self.height:
            y = 0
        
        return x, y
    
    def move(self):
        """Move snake with wrap-around walls"""
        if self.game_over or self.paused:
            return
        
        # Apply pending direction
        if self.pending_direction:
            self.direction = self.pending_direction
            self.pending_direction = None
        
        # Calculate new head position
        head_x, head_y = self.snake[0]
        
        direction_moves = {
            'up': (0, -self.cell_size),
            'down': (0, self.cell_size),
            'left': (-self.cell_size, 0),
            'right': (self.cell_size, 0)
        }
        
        dx, dy = direction_moves[self.direction]
        new_x, new_y = self.wrap_position(head_x + dx, head_y + dy)
        new_head = (new_x, new_y)
        
        # Check self collision
        if new_head in self.snake:
            self.game_over = True
            return
        
        # Add new head
        self.snake.insert(0, new_head)
        
        # Check food collision
        if new_head == self.food:
            self.score += 1
            self.food = self.spawn_food()
            self.sound_manager.play_eat_sound()
            # Gradually increase speed
            self.speed = max(0.1, self.speed - 0.005)
        else:
            self.snake.pop()
    
    def draw(self, img):
        """Draw the game"""
        # Background gradient
        for y in range(self.height):
            intensity = int(20 + (y / self.height) * 15)
            img[y:y+1, :] = [intensity * 0.7, intensity * 0.8, intensity]
        
        # Draw snake
        for i, segment in enumerate(self.snake):
            if i == 0:  # Head
                color = (50, 255, 50)
                cv2.rectangle(img, segment, 
                             (segment[0] + self.cell_size - 1, segment[1] + self.cell_size - 1),
                             color, -1)
                
                # Direction indicator
                center = (segment[0] + self.cell_size//2, segment[1] + self.cell_size//2)
                direction_arrows = {
                    'up': (center[0], center[1] - 6),
                    'down': (center[0], center[1] + 6),
                    'left': (center[0] - 6, center[1]),
                    'right': (center[0] + 6, center[1])
                }
                if self.direction in direction_arrows:
                    arrow_pos = direction_arrows[self.direction]
                    cv2.circle(img, arrow_pos, 3, (255, 255, 255), -1)
            else:  # Body
                intensity = max(120, 255 - i * 8)
                color = (0, intensity, 0)
                cv2.rectangle(img, segment, 
                             (segment[0] + self.cell_size - 1, segment[1] + self.cell_size - 1),
                             color, -1)
                cv2.rectangle(img, segment, 
                             (segment[0] + self.cell_size - 1, segment[1] + self.cell_size - 1),
                             (0, min(255, intensity + 30), 0), 1)
        
        # Draw food with pulsing effect
        food_x, food_y = self.food
        pulse = int(abs(np.sin(time.time() * 6)) * 50) + 200
        cv2.rectangle(img, (food_x, food_y), 
                     (food_x + self.cell_size - 1, food_y + self.cell_size - 1),
                     (0, 0, pulse), -1)
        
        # UI elements
        cv2.putText(img, f"Score: {self.score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f"Length: {len(self.snake)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Direction and speed info
        direction_symbols = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
        cv2.putText(img, f"Direction: {direction_symbols.get(self.direction, '?')}", 
                   (self.width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(img, "WRAP-AROUND WALLS", (self.width - 220, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        
        if self.paused:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
            cv2.putText(img, "PAUSED", (self.width//2 - 70, self.height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
            cv2.putText(img, "Press SPACE to continue", (self.width//2 - 120, self.height//2 + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if self.game_over:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
            
            cv2.putText(img, "GAME OVER!", (self.width//2 - 120, self.height//2 - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(img, f"Final Score: {self.score}", (self.width//2 - 90, self.height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, "Press 'R' to restart", (self.width//2 - 100, self.height//2 + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def show_speed_selection():
    """Show speed selection menu"""
    print("\nSPEED SELECTION")
    print("1 - SLOW (Relaxed)")
    print("2 - MEDIUM (Normal)")
    print("3 - FAST (Challenge)")
    print("ESC - Exit")
    
    # Create visual menu
    menu_img = np.zeros((400, 600, 3), dtype=np.uint8)
    menu_img.fill(30)
    
    cv2.putText(menu_img, "SELECT GAME SPEED", (150, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    speeds = [
        ("1 - SLOW", (100, 255, 100)),
        ("2 - MEDIUM", (255, 255, 100)),
        ("3 - FAST", (255, 100, 100))
    ]
    
    for i, (text, color) in enumerate(speeds):
        y_pos = 160 + i * 60
        cv2.putText(menu_img, text, (200, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.putText(menu_img, "Press ESC to exit", (190, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    cv2.imshow('Speed Selection', menu_img)
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('1'):
            cv2.destroyWindow('Speed Selection')
            return 1
        elif key == ord('2'):
            cv2.destroyWindow('Speed Selection')
            return 2
        elif key == ord('3'):
            cv2.destroyWindow('Speed Selection')
            return 3
        elif key == 27:  # ESC
            cv2.destroyWindow('Speed Selection')
            return None

def load_gesture_model(model_path='gesture_cnn_model.h5'):
    """Load gesture model with comprehensive error handling"""
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please train the model first by running:")
        print("1. python capture_gestures.py")
        print("2. python augment_images.py")
        print("3. python train_model_fixed.py")
        return None
    
    try:
        print(f"Loading model: {model_path}")
        model = load_model(model_path)
        print("Model loaded successfully!")
        
        # Test model
        test_input = np.random.random((1, 64, 64, 3)).astype('float32')
        test_pred = model.predict(test_input, verbose=0)
        print("Model test successful!")
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Model file may be corrupted. Please retrain.")
        return None

def predict_gesture_enhanced(model, frame, classes):
    """Enhanced gesture prediction with better preprocessing"""
    if model is None:
        return "right", 0.0, np.array([0.25, 0.25, 0.25, 0.25])
    
    try:
        # Enhanced preprocessing
        img = cv2.resize(frame, (64, 64))
        
        # Apply slight blur to reduce noise
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Normalize
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Predict
        pred = model.predict(img, verbose=0)
        confidence_scores = pred[0]
        
        # Get best prediction
        idx = np.argmax(confidence_scores)
        confidence = float(confidence_scores[idx])
        gesture = classes[idx]
        
        return gesture, confidence, confidence_scores
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return "right", 0.0, np.array([0.25, 0.25, 0.25, 0.25])

def draw_enhanced_gesture_info(frame, roi_coords, gesture, confidence, stable_gesture, 
                              gesture_smoother, all_confidences=None):
    """Draw comprehensive gesture information"""
    x1, y1, x2, y2 = roi_coords
    
    # Dynamic ROI color based on confidence
    if confidence > 0.8:
        roi_color = (0, 255, 0)  # Green
    elif confidence > 0.6:
        roi_color = (0, 255, 255)  # Yellow
    elif confidence > 0.4:
        roi_color = (0, 150, 255)  # Orange
    else:
        roi_color = (0, 0, 255)  # Red
    
    # Draw ROI rectangle
    thickness = max(2, int(confidence * 4))
    cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, thickness)
    
    # Confidence bar
    bar_width = x2 - x1
    bar_fill = int(bar_width * confidence)
    
    # Background bar
    cv2.rectangle(frame, (x1, y1 - 25), (x2, y1 - 15), (50, 50, 50), -1)
    # Filled portion
    cv2.rectangle(frame, (x1, y1 - 25), (x1 + bar_fill, y1 - 15), roi_color, -1)
    # Border
    cv2.rectangle(frame, (x1, y1 - 25), (x2, y1 - 15), roi_color, 1)
    
    # Confidence percentage
    cv2.putText(frame, f"{confidence*100:.0f}%", (x2 + 5, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1)
    
    # ROI label
    cv2.putText(frame, "GESTURE ZONE", (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Info panel background
    info_height = 220
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (450, 10 + info_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Current prediction
    y_pos = 35
    cv2.putText(frame, f"Current: {gesture.upper()}", (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    y_pos += 25
    cv2.putText(frame, f"Confidence: {confidence:.3f}", (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Stable gesture
    y_pos += 30
    stable_color = (0, 255, 0) if stable_gesture else (100, 100, 100)
    stable_text = stable_gesture.upper() if stable_gesture else "NONE"
    cv2.putText(frame, f"Stable: {stable_text}", (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, stable_color, 2)
    
    # All predictions
    if all_confidences is not None:
        y_pos += 25
        cv2.putText(frame, "All Predictions:", (15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        classes = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        for i, (cls, conf) in enumerate(zip(classes, all_confidences)):
            y_pos += 20
            color = (0, 255, 0) if i == np.argmax(all_confidences) else (150, 150, 150)
            cv2.putText(frame, f"  {cls}: {conf:.3f}", (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Debug info
    debug_info = gesture_smoother.get_debug_info()
    y_pos += 25
    cv2.putText(frame, f"Threshold: {debug_info['threshold']}", (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Instructions
    instructions = [
        "ESC: Quit | R: Restart | SPACE: Pause",
        "Show clear gestures in the colored box",
        "Green=Great, Yellow=Good, Red=Poor"
    ]
    
    start_y = frame.shape[0] - 70
    for i, instruction in enumerate(instructions):
        cv2.putText(frame, instruction, (10, start_y + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

def main():
    """Main game function with all fixes applied"""
    print("GESTURE-CONTROLLED SNAKE GAME")
    print("=" * 40)
    
    # Speed selection
    selected_speed = show_speed_selection()
    if selected_speed is None:
        print("Game cancelled.")
        return
    
    speed_names = {1: "SLOW", 2: "MEDIUM", 3: "FAST"}
    print(f"Selected speed: {speed_names[selected_speed]}")
    
    # Configuration
    CLASSES = ['up', 'down', 'left', 'right']
    MODEL_PATH = 'gesture_cnn_model.h5'
    
    # Larger ROI for better detection
    ROI_X1, ROI_Y1 = 100, 100
    ROI_X2, ROI_Y2 = 400, 400
    
    # Load model
    print("Loading gesture recognition model...")
    model = load_gesture_model(MODEL_PATH)
    
    if model is None:
        print("Running without gesture recognition")
        print("Use WASD or arrow keys to control snake")
        use_gestures = False
    else:
        print("Model loaded successfully!")
        use_gestures = True
    
    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    
    # Initialize game
    game = SnakeGame()
    game.set_speed(selected_speed)
    
    gesture_smoother = ImprovedGestureSmoothing(
        window_size=5,
        confidence_threshold=0.6
    )
    
    # Timing variables
    last_move_time = time.time()
    last_fps_time = time.time()
    fps_counter = 0
    fps_display = 0
    
    print(f"\nGame started with {speed_names[selected_speed]} speed!")
    print("\nCONTROLS:")
    if use_gestures:
        print("GESTURE CONTROLS:")
        print("- UP: Thumbs up gesture")
        print("- DOWN: Fist gesture")
        print("- LEFT: Peace sign pointing left")
        print("- RIGHT: Open palm gesture")
    print("KEYBOARD CONTROLS:")
    print("- WASD or Arrow Keys: Direct control")
    print("- ESC: Quit | R: Restart | SPACE: Pause")
    print(f"- Sound: {'ON' if game.sound_manager.enabled else 'OFF'}")
    
    try:
        while True:
            current_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Camera error")
                break
            
            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)
            
            # Gesture recognition
            if use_gestures:
                roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
                gesture, confidence, all_confidences = predict_gesture_enhanced(model, roi, CLASSES)
                gesture_smoother.add_prediction(gesture, confidence)
                stable_gesture = gesture_smoother.get_stable_gesture()
                
                # Apply gesture to game
                if stable_gesture and not game.game_over and not game.paused:
                    game.set_direction(stable_gesture)
            else:
                gesture, confidence, stable_gesture, all_confidences = "none", 0.0, None, None
            
            # Move snake
            if current_time - last_move_time >= game.speed:
                game.move()
                last_move_time = current_time
            
            # Draw game
            game_img = np.zeros((game.height, game.width, 3), dtype=np.uint8)
            game.draw(game_img)
            
            # Draw gesture info
            if use_gestures:
                draw_enhanced_gesture_info(frame, (ROI_X1, ROI_Y1, ROI_X2, ROI_Y2),
                                         gesture, confidence, stable_gesture,
                                         gesture_smoother, all_confidences)
            else:
                cv2.putText(frame, "KEYBOARD MODE", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(frame, "Use WASD or arrow keys", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            # FPS calculation
            fps_counter += 1
            if current_time - last_fps_time >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                last_fps_time = current_time
            
            # Display FPS
            cv2.putText(frame, f"FPS: {fps_display}", (frame.shape[1] - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show windows
            cv2.imshow('Gesture Control Camera', frame)
            cv2.imshow('Snake Game', game_img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('r') or key == ord('R'):
                game.reset()
                gesture_smoother = ImprovedGestureSmoothing()
                print("Game restarted!")
            elif key == ord(' '):  # SPACE
                game.paused = not game.paused
                print("Game paused" if game.paused else "Game resumed")
            
            # Manual controls
            elif key == ord('w') or key == 82:  # W or Up arrow
                game.set_direction('up')
            elif key == ord('s') or key == 84:  # S or Down arrow
                game.set_direction('down')
            elif key == ord('a') or key == 81:  # A or Left arrow
                game.set_direction('left')
            elif key == ord('d') or key == 83:  # D or Right arrow
                game.set_direction('right')
    
    except KeyboardInterrupt:
        print("\nGame interrupted")
    except Exception as e:
        print(f"Game error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nFinal Score: {game.score}")
        print("Thanks for playing!")

if __name__ == "__main__":
    main()