# 선택한 vertex의 3d, 2d 저장. 사진 저장
import Metashape 
import random, os, shutil, csv, cv2
import pandas as pd

chunk = Metashape.app.document.chunk
doc = Metashape.app.document

img_dir = r"D:\Users\hoans\rist\sam\result_img"
marker_file = r"D:\Users\hoans\rist\sam\marker_coordinate.csv"
vertex_3d_file = r"D:\Users\hoans\rist\sam\vertex_3d.csv"
vertex_2d_file = r"D:\Users\hoans\rist\sam\vertex_2d"

# 픽셀 좌표를 3D 좌표로 변환하는 함수
def pixel_to_point3D(imX, imY, camera, chunk):
    point2D = Metashape.Vector([imX, imY])
    sensor = camera.sensor
    unprojected = sensor.calibration.unproject(point2D)
    picked_point = chunk.model.pickPoint(camera.center, camera.transform.mulp(unprojected))
    if picked_point is None:
        return None
    transformed_point = chunk.transform.matrix.mulp(picked_point)
    transformed_point.size = 3
    world_point = chunk.crs.project(transformed_point)
    return world_point

# 마커 좌표를 통해 3D 변환을 테스트하는 함수
def marker_coordinates(doc, img_dir):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    
    # if not os.path.exists(marker_file):
    #     os.makedirs(marker_file)
    
    for chunk in doc.chunks:
        for camera in chunk.cameras:
            visible_marker = 0
            for marker in chunk.markers:
                if marker.position and marker.projections[camera]:
                    # 마커의 이미지 좌표 얻기
                    marker_proj = marker.projections[camera].coord
                    imX, imY = marker_proj[0], marker_proj[1]
                    
                    # 3D 좌표 변환
                    result = pixel_to_point3D(imX, imY, camera, chunk)
                    if result is not None:
                        if visible_marker is not 2:
                            visible_marker = 1
                        # print(f"Camera: {camera.label}, Marker: {marker.label}, Image Coords: ({imX}, {imY})")
                        image_path = camera.photo.path
                        image_name = os.path.basename(image_path)
                        output_path = os.path.join(img_dir, image_name)
                        if not os.path.exists(output_path):
                            shutil.copy(image_path, output_path)
                            print(f"Image saved: {output_path}")    

                        # 2D 마커 좌표를 CSV 파일에 저장
                        with open(marker_file, mode='a', newline='') as file:  # 'a' 모드로 수정
                            writer = csv.writer(file)
                            writer.writerow([camera.label, marker.label, imX, imY])
            if visible_marker is 1:
                vertex_3d_to_2d(camera)
                visible_marker = 2

def vertex_3d_to_2d(camera):
    if not camera.transform:
        print(f"카메라 {camera.label}의 변환 행렬이 없습니다.")
        return
    
    camera_vertex_2d_file = os.path.join(vertex_2d_file, f"{camera.label}.csv")
    
    if not os.path.exists(vertex_2d_file):
        os.makedirs(vertex_2d_file)
        
    with open(camera_vertex_2d_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow(["Camera", "Vertex Index", "2D X", "2D Y"])  # 헤더 작성

        for vertex_index, vertex_coord in vertex_index_to_coord.items():
            vertex_3D = Metashape.Vector(vertex_coord)
            vertex_2D = camera.project(vertex_3D)
            writer.writerow([camera.label, vertex_index, vertex_2D.x, vertex_2D.y])

    print(f"2D vertex coordinates saved for camera {camera.label}: {camera_vertex_2d_file}")

# 선택된 vertex 인덱스와 좌표를 저장할 딕셔너리
vertex_index_to_coord = {}
# 결과 사진

for face in chunk.model.faces:
    if face.selected:
        with open(vertex_3d_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            for v in face.vertices:
                vertex_coord = tuple(chunk.model.vertices[v].coord)
                if v not in vertex_index_to_coord:
                    writer.writerow([v] + list(vertex_coord))
                    vertex_index_to_coord[v] = vertex_coord

# print(vertex_index_to_coord)

random_coords = random.sample(list(vertex_index_to_coord.items()), 1)  # list로 변환
for index, coord in random_coords:
    # print(coord)
    chunk.addMarker(coord, visibility=True)
    
    
# SAM을 하기 위한 마커 좌표
marker_coordinates(doc, img_dir)