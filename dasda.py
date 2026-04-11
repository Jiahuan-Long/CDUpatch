

import cv2
import numpy as np

def remove_black_borders_and_add_padding(frame, padding=10):
    # Convert frame to numpy array (if not already)
    height, width, channels = frame.shape
    middle = width // 2

    # Split the frame into infrared and visible parts (left for infrared, right for visible)
    infrared_image = frame[:, :middle, :]
    visible_image = frame[:, middle:, :]

    # Remove black borders by cropping non-black regions
    infrared_non_black = np.any(infrared_image > 10, axis=-1)
    visible_non_black = np.any(visible_image > 10, axis=-1)

    # Find the bounding box for the non-black pixels in both images
    infrared_top, infrared_bottom = np.where(infrared_non_black)[0][[0, -1]]
    visible_top, visible_bottom = np.where(visible_non_black)[0][[0, -1]]

    infrared_cropped = infrared_image[infrared_top:infrared_bottom + 1, :]
    visible_cropped = visible_image[visible_top:visible_bottom + 1, :]

    # Add a white border between infrared and visible image
    new_width = infrared_cropped.shape[1] + visible_cropped.shape[1] + padding

    # Create a new image with the appropriate width
    final_image = np.ones((max(infrared_cropped.shape[0], visible_cropped.shape[0]), new_width, 3), dtype=np.uint8)

    # Paste the infrared and visible images with padding in the middle
    final_image[:infrared_cropped.shape[0], :infrared_cropped.shape[1]] = infrared_cropped
    final_image[:visible_cropped.shape[0], infrared_cropped.shape[1] + padding:] = visible_cropped

    return final_image

def process_video(input_video_path, output_video_path, padding=10):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Processing each frame might result in a different size, get the new frame size
    # Adjust width and height if necessary
    test_frame = cap.read()[1]  # Read one frame to test
    processed_test_frame = remove_black_borders_and_add_padding(test_frame, padding)
    new_width, new_height = processed_test_frame.shape[1], processed_test_frame.shape[0]
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process each frame to remove black borders and add padding
        processed_frame = remove_black_borders_and_add_padding(frame, padding)

        # Write the frame to the output video
        out.write(processed_frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Video saved to {output_video_path}")

# Example usage:
input_video_path = '/home/admin1/datasets/4.5/DJI_20250405150202_0061_S.MP4'  # Path to your input video
output_video_path = '/home/admin1/datasets/4.5/DJI_20250405150202_0061_S-processed.MP4'  # Path to save the output video

process_video(input_video_path, output_video_path)
