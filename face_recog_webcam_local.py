import os
import cv2
import pickle
import face_recognition
import numpy as np
from datetime import datetime, timedelta
from collections import deque, defaultdict
from scipy.spatial.distance import euclidean
import csv
import time
import re
import shutil

# Constants - Modified for local storage
LOG_DIR = "face_logs"  # Directory for storing logs
ENCODINGS_DIR = "face_encodings"  # Directory for storing face data
TARGET_WIDTH = 860
TARGET_HEIGHT = 480
FRAME_SKIP = 5
FACE_DISTANCE_THRESHOLD = 0.5
MIN_ENCODINGS_TO_SAVE = 2
MAX_ENCODINGS_PER_USER = 20
BOX_DISPLAY_DURATION = 1.0
ENTRY_COOLDOWN = timedelta(seconds=3)
PRESENCE_THRESHOLD = timedelta(seconds=5)

# Global state for tracking user presence
user_presence = defaultdict(lambda: {
    'last_seen': None,
    'last_logged': None,
    'status': 'exited',
    'last_confidence': None
})

def initialize_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(ENCODINGS_DIR, exist_ok=True)

def initialize_log_file():
    """Initialize the log file with headers if it doesn't exist"""
    log_file = os.path.join(LOG_DIR, "face_logs.csv")
    if not os.path.exists(log_file):
        with open(log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Username", "Status", "Confidence", "Duration"])

def log_face_event(username, status, confidence, duration=None):
    """
    Log face recognition events to local file
    Args:
        username: Identified user
        status: 'entered' or 'exited'
        confidence: Face recognition confidence score
        duration: Time spent (for exit events)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = os.path.join(LOG_DIR, "face_logs.csv")
    
    try:
        # Update local presence tracking
        timestamp_obj = datetime.now()
        if status == "entered":
            user_presence[username]['last_seen'] = timestamp_obj
        user_presence[username]['last_logged'] = timestamp_obj
        user_presence[username]['status'] = status
        user_presence[username]['last_confidence'] = confidence
        
        # Write to log file
        with open(log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, 
                username, 
                status, 
                f"{confidence:.3f}", 
                f"{duration:.1f}s" if duration else ""
            ])
    except Exception as e:
        print(f"[ERROR] Failed to write to log file: {str(e)}")

def load_local_encodings():
    """
    Load face encodings from local directory
    Returns:
        dict: {username: [encodings]} or empty dict on failure
    """
    known_data = {}
    try:
        for filename in os.listdir(ENCODINGS_DIR):
            if filename.endswith(".pkl"):
                username = os.path.splitext(filename)[0]
                with open(os.path.join(ENCODINGS_DIR, filename), "rb") as f:
                    known_data[username] = pickle.load(f)
        return known_data
    except Exception as e:
        print(f"[ERROR] Failed to load local encodings: {str(e)}")
        return {}
        
def is_blurry(image, threshold=40.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap < threshold
    
def is_encoding_unique(new_enc, enc_list, threshold=0.45):
    return all(euclidean(new_enc, existing) > threshold for existing in enc_list)

def capture_and_save_face(username):
    """
    Capture face images and generate encodings for a new user
    Args:
        username: Unique identifier for the user
    Returns:
        bool: True if successful, False otherwise
    """
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] Couldn't open webcam")
        return False

    # Configure camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    encodings = deque(maxlen=MAX_ENCODINGS_PER_USER)  # This sets the maximum limit (20)
    frame_count = 0
    last_box_time = 0
    box_coordinates = None
    last_encoding_time = 0
    encoding_interval = 0.5  # Seconds between capturing new encodings

    print(f"[INFO] Capturing face encodings for {username}. Press 'q' to stop.")
    cv2.namedWindow("Register Face", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break

        frame_count += 1
        current_time = time.time()
        processed_frame = frame.copy()

        # Process every FRAME_SKIP-th frame for efficiency
        if frame_count % FRAME_SKIP == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
            boxes = face_recognition.face_locations(rgb, model="hog")
            boxes = [(top*2, right*2, bottom*2, left*2) for (top, right, bottom, left) in boxes]

            if len(boxes) == 1:  # Only process if exactly one face is detected
                if is_blurry(frame):
                    continue

                # Only try to capture new encoding if enough time has passed
                if current_time - last_encoding_time > encoding_interval:
                    new_encs = face_recognition.face_encodings(
                        rgb, 
                        [(top//2, right//2, bottom//2, left//2) for (top, right, bottom, left) in boxes]
                    )
                    
                    if new_encs:  # Only proceed if we got an encoding
                        enc = new_encs[0]  # Take the first encoding
                        
                        # MODIFIED: Always add the encoding if we're below MIN_ENCODINGS_TO_SAVE
                        # After that, only add if it's different enough
                        if len(encodings) < MIN_ENCODINGS_TO_SAVE:
                            encodings.append(enc)
                            last_encoding_time = current_time
                        else:
                            # Only compare if we have existing encodings
                            if is_encoding_unique(enc, encodings):
                                encodings.append(enc)
                                last_encoding_time = current_time
                        
                        box_coordinates = ((boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]), username)
                        last_box_time = current_time

        # Display face bounding box if recently detected
        if box_coordinates and (current_time - last_box_time) < BOX_DISPLAY_DURATION:
            (top, right, bottom, left), name = box_coordinates
            cv2.rectangle(processed_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(processed_frame, name, (left, top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(processed_frame, f"Encodings: {len(encodings)}/{MAX_ENCODINGS_PER_USER}", 
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(processed_frame, "Move head slightly in all directions", 
            (10, TARGET_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, (0, 255, 255), 1)
            

        # Resize display frame for better viewing
        display_scale = 0.7
        if display_scale != 1.0:
            processed_frame = cv2.resize(processed_frame, (0, 0), fx=display_scale, fy=display_scale)

        cv2.imshow("Register Face", processed_frame)
        
        # Exit conditions
        if (cv2.getWindowProperty("Register Face", cv2.WND_PROP_VISIBLE) < 1 or 
            cv2.waitKey(1) & 0xFF == ord('q') or 
            len(encodings) >= MAX_ENCODINGS_PER_USER):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save encodings if minimum threshold met
    if len(encodings) >= MIN_ENCODINGS_TO_SAVE:
        print(f"\n[INFO] Captured {len(encodings)} encodings for {username}")
        return save_local_encodings(username, encodings)
    else:
        print(f"[ERROR] Need at least {MIN_ENCODINGS_TO_SAVE} distinct encodings")
        return False

def save_local_encodings(username, encodings):
    """
    Save face encodings to local file
    Args:
        username: User identifier
        encodings: List of face encodings
    Returns:
        bool: True if save succeeded
    """
    try:
        filepath = os.path.join(ENCODINGS_DIR, f"{username}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(list(encodings), f)
        print(f"[SUCCESS] Saved encodings to {filepath}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save encodings: {str(e)}")
        return False
        
def verify_user_face():
    """Main face verification function using local storage"""
    initialize_directories()
    initialize_log_file()
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] Couldn't open webcam.")
        return

    # Camera configuration
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

    # Load and prepare known face encodings
    known_faces = load_local_encodings()
    if not known_faces:
        print("[WARNING] No registered faces found. Please register users first.")
        return

    # Convert encodings to a consistent format
    known_face_names = []
    known_face_encodings = []
    for name, encodings in known_faces.items():
        for encoding in encodings:
            # Ensure each encoding is a numpy array with proper shape
            if isinstance(encoding, np.ndarray):
                known_face_encodings.append(encoding)
                known_face_names.append(name)

    if not known_face_encodings:
        print("[ERROR] No valid face encodings found in the database.")
        return

    frame_count = 0
    current_detections = set()
    processing = False
    last_boxes = {}
    print("[INFO] Verifying faces. Press 'q' to stop.")
    cv2.namedWindow("Verification", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        frame_count += 1
        display_frame = frame.copy()
        current_time = time.time()

        # Process every FRAME_SKIP-th frame
        if frame_count % (FRAME_SKIP + 1) == 0:
            processing = True
            current_detections.clear()

            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Detect face locations and encodings
            face_locations = face_recognition.face_locations(rgb_small, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
            
            # Scale face locations back to original frame size
            face_locations = [(top*2, right*2, bottom*2, left*2) 
                            for (top, right, bottom, left) in face_locations]

            for (face_location, face_encoding) in zip(face_locations, face_encodings):
                # Compare faces only if we have a valid encoding
                if face_encoding.size == 0:
                    continue

                # Compare with known faces
                matches = face_recognition.compare_faces(
                    known_face_encodings, 
                    face_encoding, 
                    tolerance=FACE_DISTANCE_THRESHOLD
                )
                
                name = "Unknown"
                distance = 1.0
                
                # If we found a match, get the best one
                if True in matches:
                    matched_indices = [i for i, match in enumerate(matches) if match]
                    face_distances = face_recognition.face_distance(
                        [known_face_encodings[i] for i in matched_indices],
                        face_encoding
                    )
                    best_match_index = matched_indices[np.argmin(face_distances)]
                    name = known_face_names[best_match_index]
                    distance = face_distances[np.argmin(face_distances)]

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                top, right, bottom, left = face_location

                if name != "Unknown":
                    current_detections.add(name)
                    now = datetime.now()
                    if user_presence[name]['status'] == "exited":
                        last_exit = user_presence[name].get('last_logged')
                        if (last_exit is None or (now - last_exit) > PRESENCE_THRESHOLD):
                            log_face_event(name, "entered", distance)
                    user_presence[name]['last_seen'] = now
                    user_presence[name]['last_confidence'] = distance

                last_boxes[name] = {
                    'coords': (top, right, bottom, left),
                    'color': color,
                    'distance': distance,
                    'time': current_time
                }

            processing = False

        # Rest of the function remains the same...
        # Check for exited users
        for name in list(user_presence.keys()):
            if (name not in current_detections and 
                user_presence[name]['last_seen'] and 
                (datetime.now() - user_presence[name]['last_seen']) > PRESENCE_THRESHOLD and
                user_presence[name]['status'] == "entered"):
                log_face_event(name, "exited", user_presence[name].get('last_confidence', 0))

        # Display bounding boxes
        for name, box_data in last_boxes.items():
            if current_time - box_data['time'] < 1.0:
                top, right, bottom, left = box_data['coords']
                color = box_data['color']
                distance = box_data['distance']
                status = user_presence.get(name, {}).get('status', 'unknown')
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                cv2.putText(display_frame, f"{name} ({status}, {distance:.2f})", 
                            (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if processing:
            cv2.putText(display_frame, "Processing...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        display_scale = 0.7
        if display_scale != 1.0:
            display_frame = cv2.resize(display_frame, (0, 0), fx=display_scale, fy=display_scale)

        cv2.imshow("Verification", display_frame)

        if cv2.getWindowProperty("Verification", cv2.WND_PROP_VISIBLE) < 1 or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Log all remaining users as exited
    for name in list(user_presence.keys()):
        if user_presence[name]['status'] == "entered":
            log_face_event(name, "exited", user_presence[name].get('last_confidence', 0))

    cap.release()
    cv2.destroyAllWindows()

def list_users():
    """List registered users from local storage"""
    try:
        if not os.path.exists(ENCODINGS_DIR):
            print("[INFO] No users registered yet.")
            return

        users = []
        for filename in os.listdir(ENCODINGS_DIR):
            if filename.endswith(".pkl"):
                users.append(os.path.splitext(filename)[0])

        if not users:
            print("[INFO] No registered users found.")
            return

        print("\n[INFO] Registered Users:")
        for i, user in enumerate(sorted(users), 1):
            print(f"{i}. {user}")
        print(f"\nTotal: {len(users)} users")

    except Exception as e:
        print(f"[ERROR] Failed to list users: {str(e)}")

def delete_user(username):
    """Delete user data from local storage"""
    try:
        filepath = os.path.join(ENCODINGS_DIR, f"{username}.pkl")
        
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"[INFO] Successfully deleted user '{username}'.")
            return True
        else:
            print(f"[INFO] No face data found for '{username}'.")
            return False

    except Exception as e:
        print(f"[ERROR] Failed to delete user: {str(e)}")
        return False

def view_logs():
    """Display verification logs from local file"""
    try:
        log_file = os.path.join(LOG_DIR, "face_logs.csv")
        
        if not os.path.exists(log_file):
            print("[INFO] No log file found.")
            return

        with open(log_file, mode="r") as f:
            reader = csv.reader(f)
            print("\n[INFO] Verification Logs:")
            for i, row in enumerate(reader, 1):
                if i == 1 and row[0] == "Timestamp":
                    print(f"{'#':<3} | {row[0]:<19} | {row[1]:<10} | {row[2]:<8} | {row[3]}")
                    print("-" * 60)
                    continue
                print(f"{i:<3} | {row[0]:<19} | {row[1]:<10} | {row[2]:<8} | {row[3]}")

    except Exception as e:
        print(f"[ERROR] Failed to read logs: {str(e)}")
        
def clear_logs():
    """Clear all verification logs"""
    try:
        log_file = os.path.join(LOG_DIR, "face_logs.csv")
        if os.path.exists(log_file):
            os.remove(log_file)
            initialize_log_file()  # Recreate with headers
            print("[INFO] Successfully cleared all logs.")
            return True
        else:
            print("[INFO] No log file found to clear.")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to clear logs: {str(e)}")
        return False

def main():
    """Main menu interface with all options"""
    # Initialize directories first
    initialize_directories()
    
    while True:
        print("\n==== LOCAL FACE RECOGNITION SYSTEM ====")
        print("1. Register a new user")
        print("2. Verify faces (webcam)")
        print("3. List registered users")
        print("4. Delete a user")
        print("5. View verification logs")
        print("6. Clear all logs")
        print("7. Exit")
        
        try:
            choice = input("\nChoose an option (1-7): ").strip()
            
            if choice == '1':
                username = input("Enter username: ").strip().lower()
                if username:
                    if not re.match("^[a-z0-9_]+$", username):
                        print("[ERROR] Username can only contain letters, numbers and underscores")
                    else:
                        capture_and_save_face(username)
                else:
                    print("[ERROR] Username cannot be empty")
            
            elif choice == '2':
                verify_user_face()
            
            elif choice == '3':
                list_users()
            
            elif choice == '4':
                username = input("Enter username to delete: ").strip().lower()
                if username:
                    delete_user(username)
                else:
                    print("[ERROR] Username cannot be empty")
            
            elif choice == '5':
                view_logs()
            
            elif choice == '6':
                confirm = input("Are you sure you want to clear ALL logs? (y/n): ").lower()
                if confirm == 'y':
                    clear_logs()
            
            elif choice == '7':
                print("Goodbye!")
                break
            
            else:
                print("[ERROR] Invalid choice. Please enter 1-9")
        
        except KeyboardInterrupt:
            print("\n[INFO] Operation cancelled by user")
        
        except Exception as e:
            print(f"[ERROR] An error occurred: {str(e)}")

if __name__ == "__main__":
    
    # Check if face_recognition is available
    try:
        face_recognition.face_encodings(np.zeros((100, 100, 3), dtype=np.uint8))
    except Exception as e:
        print(f"[CRITICAL] Face recognition library not working: {str(e)}")
        exit(1)
    
    # Start the application
    main()        
