import os
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint="../sam_vit_h_4b8939.pth")
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

# File paths
input_img_path = r"D:\Users\hoans\rist\sam\result_img"
output_img_path = r"D:\Users\hoans\rist\sam\sam_img"
csv_file_path = r"D:\Users\hoans\rist\sam\marker_coordinate.csv"
input_2d_path = r"D:\Users\hoans\rist\sam\vertex_2d"
output_txt_path = os.path.join(output_img_path, 'filtered_coordinates.csv')

os.makedirs(output_img_path, exist_ok=True)
scale_factor = 0.3  # Image scaling

filtered_coord = set()

def get_color():
    return (255, 0, 255)  # SAM 영역 표시 색 (분홍색)

def read_2d_coordinates(folder_path):
    coordinates = []
    if not os.path.exists(folder_path):
        print(f"폴더를 찾을 수 없습니다: {folder_path}")
        return coordinates

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r') as file:
                    for line in file:
                        parts = line.strip().split(',')
                        camera_id = parts[0]
                        index = int(parts[1])
                        x = float(parts[2])
                        y = float(parts[3])
                        coordinates.append((camera_id, index, x, y))
            except Exception as e:
                print(f"파일 처리 중 오류 발생 ({filename}): {str(e)}")
    
    return coordinates

def read_csv_file(csv_path):
    coordinates = {}
    with open(csv_path, 'r') as file:
        for line in file:
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
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Cannot read image file: {image_path}")
        return

    camera = os.path.basename(image_path).split('.')[0]
    if camera not in points:
        print(f"No coordinates for image: {os.path.basename(image_path)}")
        return

    original_height, original_width = image_bgr.shape[:2]
    image_resized = cv2.resize(image_bgr, (0, 0), fx=scale_factor, fy=scale_factor)
    image_rgb_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    result = mask_generator.generate(image_rgb_resized)
    annotated_image_resized = image_rgb_resized.copy()
    
    marker_points = []
    for (x, y) in points[camera]:
        x_resized = int(round(x * scale_factor))
        y_resized = int(round(y * scale_factor))
        cv2.circle(annotated_image_resized, (x_resized, y_resized), 5, (255, 0, 0), -1)
        marker_points.append((x_resized, y_resized))

    if not result:
        print(f"No mask generated for {image_path}")
        return

    contains_marker_count = 0
    mask_color = np.array(get_color(), dtype=np.uint8)

    for i, mask in enumerate(result):
        mask_array = mask['segmentation']
        mask_overlay = np.zeros_like(image_rgb_resized)
        mask_overlay[mask_array] = get_color()
        masked_image = cv2.addWeighted(annotated_image_resized, 0.5, mask_overlay, 0.5, 0)

        contours, _ = cv2.findContours(mask_array.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contains_marker = False

        for point in marker_points:
            for contour in contours:
                if cv2.pointPolygonTest(contour, point, True) >= 0:
                    contains_marker = True
                    break
            if contains_marker:
                contains_marker_count += 1
                break

        if contains_marker:
            mask_image = np.zeros_like(annotated_image_resized)
            mask_image[mask_array] = mask_color
            annotated_image_resized = cv2.addWeighted(annotated_image_resized, 1.0, mask_image, 0.5, 0)

            filtered_coord = set()

            for (camera_id, index, x, y) in input_2d_coordinates:
                if camera_id == camera:
                    x_resized = x * scale_factor
                    y_resized = y * scale_factor
                    x_resized_int = int(round(x_resized))
                    y_resized_int = int(round(y_resized))
                    print(f"Resized coordinates: {x_resized_int}, {y_resized_int}")

                    if 0 <= x_resized_int < mask_array.shape[1] and 0 <= y_resized_int < mask_array.shape[0]:
                        if mask_array[y_resized_int, x_resized_int]:
                            filtered_coord.add((camera_id, x, y))
                            cv2.circle(annotated_image_resized, (x_resized_int, y_resized_int), 1, (0, 0, 255), -1)
                            print(f"Filtered coordinate: {camera_id}, {x}, {y}")

            if filtered_coord:
                with open(output_txt_path, 'a') as f:
                    for (camera_id, x, y) in filtered_coord:
                        f.write(f"{camera_id},{x},{y}\n")

    print(f"{camera} image - Masks containing markers: {contains_marker_count}")

    annotated_image = cv2.resize(annotated_image_resized, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    output_path = os.path.join(output_img_path, f"annotated_{os.path.basename(image_path)}")

    if contains_marker_count == 1:
        success = cv2.imwrite(output_path, annotated_image_bgr)
        if success:
            print(f"Annotated image saved successfully at {output_path}")
        else:
            print(f"Failed to save image at {output_path}")
    else:
        print("No SAM No Save")


# Read coordinates
points = read_csv_file(csv_file_path)
input_2d_coordinates = read_2d_coordinates(input_2d_path)

# Process images with threading
if os.path.isdir(input_img_path):
    image_files = [
        os.path.join(input_img_path, filename)
        for filename in os.listdir(input_img_path)
        if filename.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers as needed
        executor.map(lambda img: process_image(img, points, input_2d_coordinates), image_files)
else:
    process_image(input_img_path, points, input_2d_coordinates)

# 프로그램 시작 시 output_txt_path 파일 초기화
with open(output_txt_path, 'w') as f:
    f.write("")  # 빈 파일 생성
