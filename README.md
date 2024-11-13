## Matrix Glitch FX

A simple Matrix-like motion-triggered visual effect using OpenCV and Pygame.

Overview

Matrix Glitch FX is a lightweight and fun visual effect inspired by the famous "falling green code" scene from The Matrix. When movement is detected in the video feed, the moving regions are replaced with a "Matrix code rain" effect, creating a striking glitch-like transition.

The effect uses Python with OpenCV and Pygame, making it an easily extensible and accessible tool for beginner or advanced developers interested in video and visual effects.

## Features

Motion detection with OpenCV.

Matrix-style falling characters effect for glitch transitions.

## Requirements

To run this script, you need the following dependencies installed:

Python 3.8+

OpenCV

Pygame

Numpy

You can install all the dependencies using the provided requirements.txt file.

pip install -r requirements.txt

## Installation & Usage

Clone the Repository

git clone https://github.com/anttiluode/MatrixGlitchFX.git

cd MatrixGlitchFX

Install Dependencies

Use the provided requirements.txt to install all the necessary libraries:

pip install -r requirements.txt

Run the Application

Simply run app.py to start the Matrix Glitch FX:

python app.py

You can adjust the webcam index or resolution as needed by changing the values in app.py:

desired_camera_index = 0  # Change this to the appropriate camera index
resolution = (640, 480)    # Modify resolution if needed

## Control the Application

Press Spacebar to pause or resume the effect.

Press Up Arrow or Down Arrow to increase or decrease the motion detection threshold.

Press Q or close the window to quit.

## Requirements.txt

A requirements.txt is provided to make setup easy.

numpy
opencv-python
pygame

## License

This project is released under the MIT License.
