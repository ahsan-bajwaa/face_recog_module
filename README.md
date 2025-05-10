MAke sure that you set executable permission to setup.sh file!

Also also change it ip address of your server.

For Mobile connection install IP Webcam application from playstore.

Run this script in virtual envirnment!
You’ll need CMake, Boost, and some development headers:
```
sudo apt update
sudo apt install -y build-essential cmake \
    libopenblas-dev liblapack-dev libx11-dev \
    libgtk-3-dev libboost-python-dev \
    python3-dev
sudo apt install python3.13-dev
pip install --upgrade pip setuptools wheel
```
This may take a while.
```
pip install dlib
```
Then,

```
pip install opencv-python
pip install face_recognition
pip install numpy
pip install python-dateutil
pip install scipy
pip install Flask
pip install requests
pip install python-dotenv
pip install hashlib
pip install urllib3
```
