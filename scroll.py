import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configure webcam capture
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

# Improved configuration
SCROLL_SMOOTHING = 0.5  # Lower = smoother transitions (0-1)
SCROLL_ACCELERATION = 1.2  # Speed multiplier for faster scrolling
SCROLL_DEADZONE = 0.15  # Minimum movement to trigger scroll
HISTORY_BUFFER = 5  # Number of frames for gesture averaging

# Initialize variables
scroll_speed = 0
scroll_history = []
last_scroll_time = time.time()
is_scrolling = False
scroll_direction = 0

# Set up MediaPipe Hands
with mp_hands.Hands(
    model_complexity=0,  # Reduced complexity for better performance
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1) as hands:
    
    print("Enhanced hand gesture scrolling activated!")
    print("- Point finger up/down: scroll (speed based on angle)")
    print("- Make a fist: immediate stop")
    print("- Open palm: gradual stop")
    print("- Press 'q' to quit")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to get camera feed.")
            break
            
        # Flip the image horizontally for natural movement
        image = cv2.flip(image, 1)
        debug_image = image.copy()
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = hands.process(image_rgb)
        
        # Reset detection flags
        current_speed = 0
        current_direction = 0
        status = "READY"
        gesture = "none"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get key landmarks
                landmarks = hand_landmarks.landmark
                wrist = landmarks[mp_hands.HandLandmark.WRIST]
                
                # Finger state detection
                finger_states = {
                    'index': landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
                    'middle': landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
                    'ring': landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y,
                    'pinky': landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_PIP].y,
                    'thumb': landmarks[mp_hands.HandLandmark.THUMB_TIP].x < landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
                }

                                # Gesture classification
                if all(not v for v in finger_states.values()):  # Fist
                    gesture = "fist"
                    scroll_speed = 0
                    status = "FIST - STOPPED"
                
                elif finger_states['index']:  # Pointing
                    gesture = "pointing"
                    
                    # Get index finger position
                    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                    
                    # Calculate direction
                    screen_middle = 0.5  # Middle of the screen (normalized)
                    if index_tip.y < screen_middle:
                        current_direction = 1  # Scroll Up
                    else:
                        current_direction = -1  # Scroll Down
                    
                    # Adjust speed based on extra fingers
                    base_speed = 30
                    if finger_states['middle']:  # Boost speed
                        base_speed *= 1.8
                    
                    current_speed = int(base_speed * SCROLL_ACCELERATION)
                    status = f"SCROLLING {'UP' if current_direction > 0 else 'DOWN'} (Speed: {current_speed})"
                
                elif all(finger_states.values()):  # Open Hand
                    gesture = "open"
                    status = "OPEN HAND - STOPPING"
                    current_speed = 0

                # Smooth speed transitions
                if gesture == "pointing":
                    scroll_speed = scroll_speed * SCROLL_SMOOTHING + current_speed * (1 - SCROLL_SMOOTHING)
                else:
                    scroll_speed = scroll_speed * 0.7  # Gradual slowdown

                # Apply scrolling
                if time.time() - last_scroll_time > 0.05 and abs(scroll_speed) > 5:
                    pyautogui.scroll(int(current_direction * scroll_speed))
                    last_scroll_time = time.time()


                # Draw visual feedback
                cv2.rectangle(debug_image, (20, 20), (40, 200), (50, 50, 50), -1)
                scroll_bar_height = int(180 * (scroll_speed / 100))
                cv2.rectangle(debug_image, (20, 200 - scroll_bar_height), 
                             (40, 200), (0, 255, 0), -1)
                
                # Draw gesture information
                mp_drawing.draw_landmarks(
                    debug_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # Display status
        cv2.putText(debug_image, f"Status: {status}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_image, f"Speed: {int(scroll_speed)}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the image
        cv2.imshow('Gesture Scrolling', debug_image)
        
        # Exit on 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Gesture scrolling deactivated.")