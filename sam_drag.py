import os
import torch
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np

# 초기 설정
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint=r"C:\Users\yejim\Desktop\cgv\sam\python\checkpoint\sam_vit_h_4b8939.pth")
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

input_img_path = r"C:\Users\yejim\Desktop\sam\result_img"
output_img_path = r"C:\Users\yejim\Desktop\sam\sam_img"
csv_file_path = r"C:\Users\yejim\Desktop\sam\marker_coordinate.csv"
input_2d_path = r"C:\Users\yejim\Desktop\sam\vertex_2d.csv"
output_txt_path = os.path.join(output_img_path, 'filtered_coordinates.csv')

os.makedirs(output_img_path, exist_ok=True)
scale_factor = 0.3  # 이미지 축소 비율

filtered_coord = set()
drawing = False
start_point = None
end_point = None
current_image = None
current_points = None
current_input_2d_coordinates = None
current_image_path = None

# 유틸리티 함수
def get_color():
    return (255, 0, 255)  # 마스크 영역 색상 (분홍색)

def read_2d_coordinates(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        for line in file:
            camera_id, index, x_str, y_str = line.strip().split(',')
            x = float(x_str)
            y = float(y_str)
            coordinates.append((camera_id, int(index), x, y))
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
    return coordinates

# 마우스 콜백 함수
def draw_box(event, x, y, flags, param):
    global drawing, start_point, end_point, current_image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = current_image.copy()
            cv2.rectangle(img_copy, start_point, (x, y), (0, 255, 0), 2)
            cv2.imshow('image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        x1, y1 = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
        x2, y2 = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])
        process_with_box(np.array([x1, y1, x2, y2]))

# 박스 처리 함수
def process_with_box(box):
    global current_image, current_points, current_input_2d_coordinates, current_image_path

    original_image = cv2.imread(current_image_path)
    height_ratio = original_image.shape[0] / current_image.shape[0]
    width_ratio = original_image.shape[1] / current_image.shape[1]

    x1, y1, x2, y2 = box
    x1, x2 = int(x1 * width_ratio), int(x2 * width_ratio)
    y1, y2 = int(y1 * height_ratio), int(y2 * height_ratio)

    input_box = np.array([x1, y1, x2, y2])
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    roi_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    roi_mask[y1:y2, x1:x2] = 255

    predictor.set_image(image_rgb)
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False
    )

    if masks is None or len(masks) == 0:
        print("No mask generated.")
        return

    mask = masks[0].astype(bool) & (roi_mask == 255)
    mask_color = np.array(get_color(), dtype=np.uint8)
    mask_image = np.zeros_like(image_rgb)
    mask_image[mask] = mask_color

    result_image = cv2.addWeighted(image_rgb, 1.0, mask_image, 0.5, 0)

    camera = os.path.basename(current_image_path).split('.')[0]
    if camera in current_points:
        for (x, y) in current_points[camera]:
            cv2.circle(result_image, (int(x), int(y)), 5, (255, 0, 0), -1)

    for (camera_id, index, x, y) in current_input_2d_coordinates:
        if camera_id == camera:
            x_int = int(round(x))
            y_int = int(round(y))
            if (x1 <= x_int <= x2 and y1 <= y_int <= y2) and mask[y_int, x_int]:
                filtered_coord.add((x, y))
                cv2.circle(result_image, (x_int, y_int), 1, (0, 0, 255), -1)

    result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    output_path = os.path.join(output_img_path, f"annotated_{os.path.basename(current_image_path)}")
    cv2.imwrite(output_path, result_bgr)

    with open(output_txt_path, 'w') as f:
        for coord in filtered_coord:
            f.write(f"{coord[0]}, {coord[1]}\n")

# 이미지 처리 함수
def process_image(image_path, points, input_2d_coordinates):
    global current_image, current_points, current_input_2d_coordinates, current_image_path

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Cannot read image file: {image_path}")
        return

    new_width = int(original_image.shape[1] * scale_factor)
    new_height = int(original_image.shape[0] * scale_factor)
    current_image = cv2.resize(original_image, (new_width, new_height))

    current_points = points
    current_input_2d_coordinates = input_2d_coordinates
    current_image_path = image_path

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_box)

    while True:
        cv2.imshow('image', current_image)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 키
            break

    cv2.destroyAllWindows()

points = read_csv_file(csv_file_path)
input_2d_coordinates = read_2d_coordinates(input_2d_path)

if os.path.isdir(input_img_path):
    for filename in os.listdir(input_img_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_file = os.path.join(input_img_path, filename)
            process_image(image_file, points, input_2d_coordinates)
else:
    process_image(input_img_path, points, input_2d_coordinates)
