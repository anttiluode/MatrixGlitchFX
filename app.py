import cv2
import numpy as np
import pygame
import time
import random

# ==========================
# Define DynamicDepthApp Class with Matrix Glitch Effect
# ==========================
class DynamicDepthApp:
    def __init__(self, resolution=(640, 480), camera_index=0):
        # Parameters
        self.width, self.height = resolution
        self.input_shape = (self.height, self.width)
        self.camera_index = camera_index  # Set the desired camera index here
        self.is_paused = False
        self.threshold = 25  # Initial threshold for motion detection
        self.min_area = 500  # Minimum area for a region to be considered motion

        # Initialize Webcam
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open webcam with index {self.camera_index}. Please check the camera index.")

        # Initialize Background Subtractor
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("Dynamic Matrix Glitch Effect with Motion-Based Display")

        # Matrix Effect Parameters
        self.matrix_chars = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.num_columns = self.width // 10  # Width of each character block
        self.matrix_drops = [0 for _ in range(self.num_columns)]

    def run(self):
        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.is_paused = not self.is_paused
                        print(f"{'Paused' if self.is_paused else 'Resumed'}")
                    elif event.key == pygame.K_UP:
                        # Increase threshold
                        self.threshold = min(self.threshold + 5, 255)
                        print(f"Threshold Increased to: {self.threshold}")
                    elif event.key == pygame.K_DOWN:
                        # Decrease threshold
                        self.threshold = max(self.threshold - 5, 0)
                        print(f"Threshold Decreased to: {self.threshold}")

            if not self.is_paused:
                # Capture frame-by-frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Resize frame if necessary
                frame = cv2.resize(frame, (self.width, self.height))

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Apply background subtractor to get foreground mask
                fgMask = self.backSub.apply(rgb_frame)

                # Threshold the mask to remove shadows (gray pixels)
                _, fgMask = cv2.threshold(fgMask, self.threshold, 255, cv2.THRESH_BINARY)

                # Optional: Apply morphological operations to remove noise
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=2)
                fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_DILATE, kernel, iterations=2)

                # Find contours to identify moving objects
                contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Create a mask for moving regions
                motion_mask = np.zeros_like(fgMask)

                for contour in contours:
                    if cv2.contourArea(contour) > self.min_area:
                        cv2.drawContours(motion_mask, [contour], -1, 255, -1)  # Fill the contour

                # Generate Matrix code background
                matrix_layer = self.generate_matrix_layer()

                # Normalize motion_mask to [0,1]
                motion_mask_normalized = motion_mask / 255.0

                # Expand mask to three channels
                motion_mask_3ch = np.stack([motion_mask_normalized] * 3, axis=2)

                # Blend the matrix layer only in moving regions
                glitch_frame = (rgb_frame * (1 - motion_mask_3ch) + matrix_layer * motion_mask_3ch).astype(np.uint8)

                # Resize to window size if necessary
                window_size = self.screen.get_size()
                resized_image = cv2.resize(glitch_frame, window_size)

                # Convert to Pygame surface
                pygame_surface = pygame.surfarray.make_surface(resized_image.swapaxes(0, 1))
                self.screen.blit(pygame_surface, (0, 0))

            # Refresh display
            pygame.display.flip()
            clock.tick(30)  # Limit to 30 FPS

        # Cleanup
        self.cap.release()
        pygame.quit()

    def generate_matrix_layer(self):
        """Generate the Matrix-like falling text layer."""
        matrix_layer = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for i in range(self.num_columns):
            char_index = random.randint(0, len(self.matrix_chars) - 1)
            char = self.matrix_chars[char_index]
            char_x = i * 10
            char_y = self.matrix_drops[i] * 10

            # Draw character in bright green color
            if char_y < self.height:
                cv2.putText(matrix_layer, char, (char_x, char_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

            # Randomly reset the drop position to create falling effect
            if random.random() > 0.95:
                self.matrix_drops[i] = 0
            else:
                self.matrix_drops[i] += 1

        return matrix_layer

# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    desired_camera_index = 0  # Set to the desired camera index usually 0 
    app = DynamicDepthApp(resolution=(640, 480), camera_index=desired_camera_index)
    app.run()
