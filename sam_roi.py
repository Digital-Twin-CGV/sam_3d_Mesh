import os
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import numpy as np
import pandas as pd
import csv

import time

# Settings
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

# Load the model
# sam = sam_model_registry[MODEL_TYPE](checkpoint="../sam_vit_h_4b8939.pth")
sam = sam_model_registry[MODEL_TYPE](checkpoint="../sam_vit_h_4b8939.pth")
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

# File paths

input_img_path = r"C:\Users\yejim\Desktop\cgv\github\github_script_final\result\result_img"
output_img_path = r"C:\Users\yejim\Desktop\cgv\github\github_script_final\result\sam_img"
csv_file_path = r"C:\Users\yejim\Desktop\cgv\github\github_script_final\result\marker_coordinate.csv"
input_2d_path = r"C:\Users\yejim\Desktop\cgv\github\github_script_final\result\vertex_2d"

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
    """디렉토리에서 CSV 파일들의 2D 좌표를 읽어옵니다."""
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

def get_roi_bounds(input_2d_coordinates, camera_id):
    """특정 카메라의 2D 좌표들의 최소/최대 범위를 반환합니다."""
    coords = [(x, y) for (cam, _, x, y) in input_2d_coordinates if cam == camera_id]
    if not coords:
        return None
    
    x_coords, y_coords = zip(*coords)
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    # ROI 영역을 약간 확장 (10% 여유)
    padding_x = (max_x - min_x) * 0.1
    padding_y = (max_y - min_y) * 0.1
    
    return {
        'min_x': max(0, min_x - padding_x),
        'max_x': max_x + padding_x,
        'min_y': max(0, min_y - padding_y),
        'max_y': max_y + padding_y
    }

def process_image(image_path, points, input_2d_coordinates):
    """이미지를 처리하여 마스크를 생성하고 좌표를 표시합니다."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Cannot read image file: {image_path}")
        return
    
    # 카메라 ID 추출
    camera = os.path.basename(image_path).split('.')[0]
    if camera not in points:
        print(f"No coordinates for image: {os.path.basename(image_path)}")
        return

    # ROI 범위 계산
    roi_bounds = get_roi_bounds(input_2d_coordinates, camera)
    if roi_bounds is None:
        print(f"No ROI bounds for camera: {camera}")
        return

    # ROI 범위 출력
    # print(f"\nROI bounds for camera {camera}:")
    # print(f"X range: {roi_bounds['min_x']:.2f} to {roi_bounds['max_x']:.2f}")
    # print(f"Y range: {roi_bounds['min_y']:.2f} to {roi_bounds['max_y']:.2f}")

    original_height, original_width = image_bgr.shape[:2]
    image_resized = cv2.resize(image_bgr, (0, 0), fx=scale_factor, fy=scale_factor)
    image_rgb_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # ROI 좌표를 리사이즈된 이미지에 맞게 조정
    roi_x1 = int(roi_bounds['min_x'] * scale_factor)
    roi_y1 = int(roi_bounds['min_y'] * scale_factor)
    roi_x2 = int(roi_bounds['max_x'] * scale_factor)
    roi_y2 = int(roi_bounds['max_y'] * scale_factor)

    # ROI 영역 표시 (더 두껍고 눈에 잘 띄는 초록색 사각형)
    cv2.rectangle(image_rgb_resized, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 3)
    
    # ROI만 표시된 이미지 저장
    # roi_debug_path = os.path.join(output_img_path, f"roi_debug_{os.path.basename(image_path)}")
    # cv2.imwrite(roi_debug_path, cv2.cvtColor(image_rgb_resized, cv2.COLOR_RGB2BGR))
    # print(f"ROI debug image saved at: {roi_debug_path}")

    # ROI 영역만 추출
    roi_image = image_rgb_resized[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # ROI 영역에 대해서만 SAM 실행
    # 약 5.9초
    result = mask_generator.generate(roi_image)

    # ROI 크기의 이미지 생성
    annotated_roi = roi_image.copy()

    # Draw points from CSV (markers) in ROI coordinates
    # 0.0000 초
    
    marker_points = []
    for (x, y) in points[camera]:
        x_resized = int(round(x * scale_factor))
        y_resized = int(round(y * scale_factor))
        
        # 마커가 ROI 내부에 있는 경우만 처리
        if (roi_x1 <= x_resized < roi_x2 and roi_y1 <= y_resized < roi_y2):
            # ROI 좌표계로 변환
            x_roi = x_resized - roi_x1
            y_roi = y_resized - roi_y1
            cv2.circle(annotated_roi, (x_roi, y_roi), 5, (255, 0, 0), -1)  # Red in RGB
            marker_points.append((x_roi, y_roi))
    

    # Overlay masks only if they contain marker points

    mask_color = np.array(get_color(), dtype=np.uint8)
    
    
    # 약 0.1 ~ 0.3 초
    for mask in result:
        mask_array = mask['segmentation']  # ROI 크기의 마스크
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

        mask_image = np.zeros_like(annotated_roi)
        mask_image[mask_array] = mask_color
        annotated_roi = cv2.addWeighted(annotated_roi, 1.0, mask_image, 0.5, 0)
        if contains_marker:
            # Apply the mask overlay to ROI

            # Filter input 2D coordinates that fall within this mask
            for (camera_id, index, x, y) in input_2d_coordinates:
                if camera_id == camera:
                    # Scale coordinates to resized image
                    x_resized = int(round(x * scale_factor))
                    y_resized = int(round(y * scale_factor))
                    
                    # 좌표가 ROI 내부에 있는지 확인
                    if (roi_x1 <= x_resized < roi_x2 and roi_y1 <= y_resized < roi_y2):
                        # ROI 좌표계로 변환
                        x_roi = x_resized - roi_x1
                        y_roi = y_resized - roi_y1
                        
                        if (0 <= x_roi < mask_array.shape[1] and 
                            0 <= y_roi < mask_array.shape[0] and 
                            mask_array[y_roi, x_roi]):
                            cv2.circle(annotated_roi, (x_roi, y_roi), 1, (0, 0, 255), -1)
                            filtered_coord.add(index)
                            
                            if camera_id not in camera_index_dict:
                                camera_index_dict[camera_id] = set()
                            camera_index_dict[camera_id].add(index)
    
    # ROI 영역을 원본 이미지에 복사
    annotated_image_resized = image_rgb_resized.copy()
    annotated_image_resized[roi_y1:roi_y2, roi_x1:roi_x2] = annotated_roi
    
    # ROI 경계 표시
    cv2.rectangle(annotated_image_resized, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)

    # Resize annotated image back to original size
    annotated_image = cv2.resize(annotated_image_resized, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    
    # Convert back to BGR for saving
    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    # Save the annotated image
    # 0.0000 초
    output_path = os.path.join(output_img_path, f"annotated_{os.path.basename(image_path)}")
    
    # 0.6 ~ 0.75 초
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
        with open(output_csv, 'a') as csvfile:
            writer = csv.writer(csvfile)
            for index in filtered_indexes:
                writer.writerow([index])
    
# Read points from CSV and 2D coordinates from text file
points = read_csv_file(csv_file_path)
input_2d_coordinates = read_2d_coordinates(input_2d_path)
# 4초

# Process the image(s)
# 351.5 초
start_time = time.time()
if os.path.isdir(input_img_path):
    for filename in os.listdir(input_img_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_file = os.path.join(input_img_path, filename)

            process_image(image_file, points, input_2d_coordinates)
            # 약 7.5초
else:
    process_image(input_img_path, points, input_2d_coordinates)
end_time = time.time()
print(f"SAM time: {end_time - start_time:.4f} seconds")

start_time = time.time()
# 0.2 초
save_common_index(output_txt_path)
end_time = time.time()
print(f"index time: {end_time - start_time:.4f} seconds")
# print(filtered_coord)
# with open(output_txt_path, mode='a', newline='') as file:
#             writer = csv.writer(file)
#             for coord in filtered_coord:
#                 writer.writerow([coord])