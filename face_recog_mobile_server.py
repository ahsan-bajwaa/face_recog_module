import os
import cv2
import pickle
import face_recognition
import numpy as np
from datetime import datetime, timedelta
from scipy.spatial.distance import euclidean
from flask import Flask, request, jsonify
from collections import deque, defaultdict
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import hashlib
import ssl
import csv
import time
import requests
import re
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Constants
LOG_FILE = "face_logs.csv"
TARGET_WIDTH = 860
TARGET_HEIGHT = 480
FRAME_SKIP = 5
FACE_DISTANCE_THRESHOLD = 0.5
MIN_ENCODINGS_TO_SAVE = 2
MAX_ENCODINGS_PER_USER = 20
BOX_DISPLAY_DURATION = 1.0
ENTRY_COOLDOWN = timedelta(seconds=3)
PRESENCE_THRESHOLD = timedelta(seconds=5)
CHUNK_SIZE = 8192  # 8KB per read chunk (safe and efficient)
SERVER_BASE_URL = "http://192.168.2.186:1234/uploads"  # Updated server IP
FLASK_API_URL = "http://192.168.2.186:5001/verify"   # HTTPS for secure verification
ip_address = "http://Aaa:98760@192.168.193.182:7788/video"
ENCODINGS_DIR = "encodings"

# Global state for tracking user presence
user_presence = defaultdict(lambda: {
    'last_seen': None,
    'last_logged': None,
    'status': 'exited',
    'last_confidence': None
})

def initialize_log_file():
    """Initialize the log file with headers if it doesn't exist"""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Username", "Status", "Confidence", "Duration"])

def log_face_event(username, status, confidence, duration=None):
    """
    Log face recognition events to both local file and remote server
    Args:
        username: Identified user
        status: 'entered' or 'exited'
        confidence: Face recognition confidence score
        duration: Time spent (for exit events)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp},{username},{status},{confidence:.3f},{duration or ''}\n"
    
    try:
        load_dotenv()
        auth = HTTPBasicAuth(os.getenv('WEBDAV_USER'), os.getenv('WEBDAV_PASS'))
        log_url = f"{os.getenv('WEBDAV_URL').rstrip('/')}/face_logs.csv"
        
        # Fetch existing logs with SSL verification disabled for testing
        response = requests.get(log_url, auth=auth)
        existing_logs = response.text if response.status_code == 200 else "Timestamp,Username,Status,Confidence,Duration\n"
        
        # Append new entry and upload
        updated_logs = existing_logs + log_entry
        requests.put(
            log_url,
            data=updated_logs,
            auth=auth,
            headers={'Content-Type': 'text/csv'},
        )
        
        # Update local presence tracking
        timestamp_obj = datetime.now()
        if status == "entered":
            user_presence[username]['last_seen'] = timestamp_obj
        user_presence[username]['last_logged'] = timestamp_obj
        user_presence[username]['status'] = status
        user_presence[username]['last_confidence'] = confidence
        
    except Exception as e:
        print(f"[WARNING] Failed to update server logs: {str(e)}")
        # Fallback to local logging
        with open(LOG_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, username, status, f"{confidence:.3f}", f"{duration:.1f}s" if duration else ""])

def fetch_remote_encodings():
    """
    Fetch face encodings from remote server with error handling
    Returns:
        dict: {username: [encodings]} or empty dict on failure
    """
    try:
        response = requests.get(SERVER_BASE_URL)
        if response.status_code != 200:
            print("[ERROR] Unable to list users from server.")
            return {}

        # Try to automatically discover users if possible
        usernames = []
        try:
            # Attempt to parse directory listing if server allows
            if "html" in response.text.lower():
                # Simple HTML parsing for demo - replace with proper parsing in production
                usernames = list(set(re.findall(r'href="([^"]+\.pkl)"', response.text)))
                usernames = [u.split('.')[0] for u in usernames]
            else:
                usernames = ["lawliet", "kira"]  # Fallback
        except:
            usernames = ["lawliet", "kira"]  # Hardcoded fallback

        known_data = {}
        for username in usernames:
            file_url = f"{SERVER_BASE_URL}/{username}.pkl"
            r = requests.get(file_url)
            if r.status_code == 200:
                encodings = pickle.loads(r.content)
                known_data[username] = encodings
            else:
                print(f"[WARNING] Could not retrieve encoding for {username}")
        return known_data

    except Exception as e:
        print(f"[ERROR] Failed to fetch encodings from server: {str(e)}")
        return {}
       
def is_blurry(image, threshold=50.0):
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
    encodings = deque(maxlen=MAX_ENCODINGS_PER_USER)
    frame_count = 0
    last_box_time = 0
    box_coordinates = None

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

                new_encs = face_recognition.face_encodings(
                    rgb, 
                    [(top//2, right//2, bottom//2, left//2) for (top, right, bottom, left) in boxes]
                )
                
                for (top, right, bottom, left), enc in zip(boxes, new_encs):
                    # Only add encoding if it's sufficiently different from existing ones
                    if len(encodings) > 0:
                        if is_encoding_unique(enc, encodings):
                            encodings.append(enc)
                    else:
                        encodings.append(enc)
                        
                    box_coordinates = ((top, right, bottom, left), username)
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
        
        # Exit on 'q' key or window close
        if cv2.getWindowProperty("Register Face", cv2.WND_PROP_VISIBLE) < 1 or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save encodings if minimum threshold met
    if len(encodings) >= MIN_ENCODINGS_TO_SAVE:
        print(f"\n[INFO] Captured {len(encodings)} encodings for {username}")
        return secure_upload(username, encodings)
    else:
        print(f"[ERROR] Need at least {MIN_ENCODINGS_TO_SAVE} distinct encodings")
        return False

def calculate_file_hash(file_path):
    """
    Generate SHA-256 hash of a file for integrity verification
    Args:
        file_path: Path to the file
    Returns:
        str: Hexadecimal hash string
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(CHUNK_SIZE):
            sha256.update(chunk)
    return sha256.hexdigest()

def secure_upload(username, encodings):
    """
    Securely upload face encodings to server with integrity checking
    Args:
        username: User identifier
        encodings: List of face encodings
    Returns:
        bool: True if upload succeeded (locally or remotely)
    """
    try:
        # Ensure local directory exists
        os.makedirs(ENCODINGS_DIR, exist_ok=True)
        
        # Save local copy first
        local_path = os.path.join(ENCODINGS_DIR, f"{username}.pkl")
        with open(local_path, 'wb') as f:
            pickle.dump(list(encodings), f)
        
        # Calculate hash for integrity verification
        file_hash = calculate_file_hash(local_path)
        
        # Get server credentials
        load_dotenv()
        auth = HTTPBasicAuth(
            os.getenv('WEBDAV_USER'), 
            os.getenv('WEBDAV_PASS')
        )
        server_url = os.getenv('WEBDAV_URL')
        
        if not all([auth.username, auth.password, server_url]):
            print("[ERROR] Missing server credentials in .env")
            return False
            
        # Prepare secure upload headers
        headers = {
            'X-File-Hash': file_hash,
            'Content-Type': 'application/octet-stream'
        }
        server_path = f"{server_url.rstrip('/')}/{username}.pkl"
        
        # Upload with hash verification
        with open(local_path, 'rb') as f:
            response = requests.put(
                server_path,
                data=f,
                auth=auth,
                headers=headers,
            )
        
        if response.status_code in (200, 201, 204):
            print(f"[SUCCESS] Uploaded {username}.pkl to server")
            return True
        else:
            print(f"[ERROR] Upload failed (HTTP {response.status_code})")
            return False
            
    except Exception as e:
        print(f"[ERROR] Upload error: {str(e)}")
        return False
        
def verify_user_face():
    """Main face verification function with server communication"""
    initialize_log_file()
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] Couldn't open webcam.")
        return

    # Camera configuration
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

    frame_count = 0
    current_detections = set()
    processing = False
    last_boxes = {}
    print("[INFO] Verifying faces via server. Press 'q' to stop.")
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

            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb_small, model="hog")
            encodings = face_recognition.face_encodings(rgb_small, boxes)
            boxes = [(top*2, right*2, bottom*2, left*2) for (top, right, bottom, left) in boxes]

            for (top, right, bottom, left), face_enc in zip(boxes, encodings):
                try:
                    # Send to verification server with timeout
                    response = requests.post(
                        FLASK_API_URL,
                        json={"encoding": face_enc.tolist()},
                        timeout=3
                    )
                    if response.status_code == 200:
                        result = response.json()
                        name = result.get("match", "Unknown")
                        distance = result.get("distance", 1.0)
                    else:
                        name = "Unknown"
                        distance = 1.0
                except requests.Timeout:
                    print("[WARNING] Server verification timeout")
                    name = "Unknown"
                    distance = 1.0
                except Exception as e:
                    print(f"[ERROR] Server error: {e}")
                    name = "Unknown"
                    distance = 1.0

                color = (0, 255, 0) if distance < FACE_DISTANCE_THRESHOLD else (0, 0, 255)

                if distance < FACE_DISTANCE_THRESHOLD:
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
        # [Previous display and cleanup code here]

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
    """List registered users from WebDAV server with XML parsing"""
    try:
        load_dotenv()
        auth = HTTPBasicAuth(os.getenv('WEBDAV_USER'), os.getenv('WEBDAV_PASS'))
        url = os.getenv('WEBDAV_URL').rstrip('/') + '/'

        response = requests.request(
            'PROPFIND',
            url,
            auth=auth,
            headers={'Depth': '1'},
        )

        if response.status_code != 207:  # 207 = Multi-Status
            print(f"[ERROR] WebDAV error: HTTP {response.status_code}")
            return

        from xml.etree import ElementTree
        root = ElementTree.fromstring(response.content)
        users = set()

        for elem in root.findall('.//{DAV:}href'):
            path = elem.text.strip()
            if path.lower().endswith('.pkl'):
                user = path.split('/')[-1].split('.')[0]
                if user:
                    users.add(user)

        if not users:
            print("[INFO] No users found via WebDAV.")
            return

        print("\n[INFO] Registered Users (WebDAV):")
        for i, user in enumerate(sorted(users), 1):
            print(f"{i}. {user}")
        print(f"\nTotal: {len(users)} users")

    except Exception as e:
        print(f"[ERROR] WebDAV query failed: {str(e)}")

def delete_user(username):
    """Delete user data from both local and remote storage"""
    try:
        # Local deletion
        local_path = os.path.join(ENCODINGS_DIR, f"{username}.pkl")
        local_deleted = False
        
        if os.path.exists(local_path):
            os.remove(local_path)
            print(f"[INFO] Deleted local face data for '{username}'.")
            local_deleted = True
        else:
            print(f"[INFO] No local encoding found for '{username}'.")

        # Remote deletion
        load_dotenv()
        auth = HTTPBasicAuth(os.getenv('WEBDAV_USER'), os.getenv('WEBDAV_PASS'))
        server_url = f"{os.getenv('WEBDAV_URL').rstrip('/')}/{username}.pkl"
        
        head_resp = requests.head(server_url, auth=auth)
        if head_resp.status_code == 200:
            del_resp = requests.delete(server_url, auth=auth)
            if del_resp.status_code in (200, 204):
                print(f"[INFO] Deleted server copy of '{username}.pkl'.")
                return True
            else:
                print(f"[ERROR] Server deletion failed (HTTP {del_resp.status_code})")
                return local_deleted
        else:
            print(f"[INFO] No server copy found for '{username}'.")
            return local_deleted

    except Exception as e:
        print(f"[ERROR] Deletion error: {str(e)}")
        return False

def view_logs():
    """Display verification logs from server"""
    try:
        load_dotenv()
        auth = HTTPBasicAuth(os.getenv('WEBDAV_USER'), os.getenv('WEBDAV_PASS'))
        log_url = f"{os.getenv('WEBDAV_URL').rstrip('/')}/face_logs.csv"

        response = requests.get(log_url, auth=auth)
        
        if response.status_code == 404:
            print("[INFO] No log file found on server.")
            return
        elif response.status_code != 200:
            print(f"[ERROR] Server returned HTTP {response.status_code}")
            return

        from io import StringIO
        csv_data = StringIO(response.text)
        reader = csv.reader(csv_data)
        
        print("\n[INFO] Server Verification Logs:")
        for i, row in enumerate(reader, 1):
            if i == 1 and row[0] == "Timestamp":
                print(f"{'#':<3} | {row[0]:<19} | {row[1]:<10} | {row[2]:<8} | {row[3]}")
                print("-" * 60)
                continue
            print(f"{i:<3} | {row[0]:<19} | {row[1]:<10} | {row[2]:<8} | {row[3]}")

    except Exception as e:
        print(f"[ERROR] Failed to process logs: {str(e)}")

def main():
    """Main menu interface with all options"""
    while True:
        print("\n==== FACE RECOGNITION SYSTEM ====")
        print("1. Register a new user")
        print("2. Verify a face")
        print("3. List registered users")
        print("4. Delete a user")
        print("5. View verification logs")
        print("6. Exit")
        
        try:
            choice = input("Choose an option (1-6): ").strip()
            
            if choice == '1':
                username = input("Enter username: ").strip().lower()
                if username:
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
                print("Goodbye!")
                break
            else:
                print("[ERROR] Invalid choice. Please enter 1-6")
        except KeyboardInterrupt:
            print("\n[INFO] Operation cancelled by user")
        except Exception as e:
            print(f"[ERROR] An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
