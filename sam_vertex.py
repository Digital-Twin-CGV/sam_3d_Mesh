import os
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import numpy as np
import pandas as pd
import csv

# Settings
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

# Load the model
sam = sam_model_registry[MODEL_TYPE](checkpoint="../sam_vit_h_4b8939.pth")
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

# File paths
input_img_path = r"D:\Users\hoans\rist\sam\result_img"
output_img_path = r"D:\Users\hoans\rist\sam\sam_img"
csv_file_path = r"D:\Users\hoans\rist\sam\marker_coordinate.csv"
input_2d_path = r"D:\Users\hoans\rist\sam\vertex_2d"
output_txt_path = os.path.join(output_img_path, 'filtered_coordinates.csv')

# Ensure the output directory exists
os.makedirs(output_img_path, exist_ok=True)

# Global variables
scale_factor = 0.3  # Image scaling factor

filtered_coord = set()
camera_index_dict = {}

def get_color():
    """Returns a specific color for annotations."""
    return (255, 0, 255)  # Pink color in RGB format

def read_2d_coordinates(file_path):
    """Reads 2D coordinates from a text file."""
    coordinates = []
    for filename in os.listdir(file_path):
        if filename.lower().endswith(".csv"):
            with open(os.path.join(file_path, filename), 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) == 4:
                        camera_id, index, x_str, y_str = row
                        try:
                            x = float(x_str)
                            y = float(y_str)
                            coordinates.append((camera_id, index, x, y))
                        except ValueError:
                            print(f"Invalid coordinate in file {file_path}: {row}")
                    else:
                        print(f"Malformed line in file {file_path}: {row}")
    return coordinates

def read_csv_file(csv_path):
    """Reads coordinates from a CSV file."""
    coordinates = {}
    
    # if not os.path.isdir(csv_path):
    #     os.mkdir(csv_path)
    
    with open(csv_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                try:
                    camera_id = parts[0].strip()
                    x = float(parts[2])
                    y = float(parts[3])
                    if camera_id not in coordinates:
                        coordinates[camera_id] = []
                    coordinates[camera_id].append((x, y))
                except ValueError:
                    print(f"Coordinate conversion error: {line.strip()}")
            else:
                print(f"Malformed line: {line.strip()}")
    return coordinates

def process_image(image_path, points, input_2d_coordinates):
    """Processes an image to generate masks and annotate coordinates."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Cannot read image file: {image_path}")
        return
    else:
        print(f"{image_path}")

    # Extract camera ID from image filename
    camera = os.path.basename(image_path).split('.')[0]
    if camera not in points:
        print(f"No coordinates for image: {os.path.basename(image_path)}")
        return
    
    # Save original image dimensions
    original_height, original_width = image_bgr.shape[:2]
    # Resize the image
    image_resized = cv2.resize(image_bgr, (0, 0), fx=scale_factor, fy=scale_factor)
    # Convert to RGB
    image_rgb_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    # Generate masks

    annotated_image_resized = image_rgb_resized.copy()
    
    # Draw points from CSV (markers)
    marker_points = []
    for (x, y) in points[camera]:
        x_resized = int(round(x * scale_factor))
        y_resized = int(round(y * scale_factor))
        # Draw red circles
        cv2.circle(annotated_image_resized, (x_resized, y_resized), 5, (255, 0, 0), -1)  # Red in RGB
        marker_points.append((x_resized, y_resized))

    result = mask_generator.generate(image_rgb_resized) # 얘가 문제임
    # Prepare to store filtered coordinates
    # filtered_coordinates = []

    # Overlay masks only if they contain marker points
    mask_color = np.array(get_color(), dtype=np.uint8)
    for mask in result:
        mask_array = mask['segmentation']  # This is a boolean array
        # Find contours of the mask
        mask_binary = mask_array.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if any marker point is inside the mask
        contains_marker = False
        for point in marker_points:
            for contour in contours:
                if cv2.pointPolygonTest(contour, point, False) >= 0:
                    contains_marker = True
                    break
            if contains_marker:
                break

        if contains_marker:
            # Apply the mask overlay
            mask_image = np.zeros_like(annotated_image_resized)
            mask_image[mask_array] = mask_color
            annotated_image_resized = cv2.addWeighted(annotated_image_resized, 1.0, mask_image, 0.5, 0)

            # Filter input 2D coordinates that fall within this mask
            for (camera_id, index, x, y) in input_2d_coordinates:
                # Scale coordinates to resized image
                x_resized = x * scale_factor
                y_resized = y * scale_factor
                x_resized_int = int(round(x_resized))
                y_resized_int = int(round(y_resized))
                
                if (0 <= x_resized_int < mask_array.shape[1]) and (0 <= y_resized_int < mask_array.shape[0]):
                    if (mask_array[y_resized_int, x_resized_int]):
                        if camera_id == camera:
                            # filtered_coord.add((x, y))
                            cv2.circle(annotated_image_resized, (x_resized_int, y_resized_int), 1, (0, 0, 255), -1)  # Yellow in RGB
                            filtered_coord.add((index))
                            
                            if camera_id not in camera_index_dict:
                                camera_index_dict[camera_id] = set()
                            camera_index_dict[camera_id].add(index)
                        

    # Resize annotated image back to original size
    annotated_image = cv2.resize(annotated_image_resized, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    # Convert back to BGR for saving
    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    # Save the annotated image
    output_path = os.path.join(output_img_path, f"annotated_{os.path.basename(image_path)}")
    success = cv2.imwrite(output_path, annotated_image_bgr)
    if success:
        print(f"Annotated image saved successfully at {output_path}")
        
    else:
        print(f"Failed to save image at {output_path}")

    # Optional: Display the image (uncomment if needed)
    # cv2.imshow("Annotated Image", cv2.cvtColor(annotated_image_resized, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    

# n/2 이상 있을
def save_common_index(output_csv):
    if not camera_index_dict:
        print("No index data available to process.")
        return
    
    camera_count = len(camera_index_dict)
    
    index_count = {}
    
    # 각 카메라의 인덱스를 순회하며 등장 횟수 세기
    for indexes in camera_index_dict.values():
        for index in indexes:
            if index not in index_count:
                index_count[index] = 0
            index_count[index] += 1
    
     # 카메라 개수의 절반 이상인 인덱스 필터링
    filtered_indexes = [index for index, count in index_count.items() if count >= (camera_count * (3/5))]
    
    if filtered_indexes:
        with open(output_csv, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for index in filtered_indexes:
                writer.writerow([index])
    
# Read points from CSV and 2D coordinates from text file
points = read_csv_file(csv_file_path)
input_2d_coordinates = read_2d_coordinates(input_2d_path)

# Process the image(s)
if os.path.isdir(input_img_path):
    for filename in os.listdir(input_img_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_file = os.path.join(input_img_path, filename)

            process_image(image_file, points, input_2d_coordinates)
else:
    process_image(input_img_path, points, input_2d_coordinates)


save_common_index(output_txt_path)
# print(filtered_coord)
# with open(output_txt_path, mode='a', newline='') as file:
#             writer = csv.writer(file)
#             for coord in filtered_coord:
#                 writer.writerow([coord])