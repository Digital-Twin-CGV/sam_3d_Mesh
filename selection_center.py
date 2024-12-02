# 선택한 vertex의 3d, 2d 저장. 사진 저장
import Metashape 
import os, shutil, csv


chunk = Metashape.app.document.chunk
doc = Metashape.app.document

img_dir = r"C:\Users\yejim\Desktop\cgv\github\github_script_final\result\result_img"
marker_file = r"C:\Users\yejim\Desktop\cgv\github\github_script_final\result\marker_coordinate.csv"
vertex_3d_file = r"C:\Users\yejim\Desktop\cgv\github\github_script_final\result\vertex_3d.csv"
vertex_2d_file = r"C:\Users\yejim\Desktop\cgv\github\github_script_final\result\vertex_2d"

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

# 선택된 vertex들의 좌표 리스트 추출
selected_coords = list(vertex_index_to_coord.values())

# 선택된 vertex들의 중간 좌표 계산
if selected_coords:
    avg_coord = [sum(coord[i] for coord in selected_coords) / len(selected_coords) for i in range(3)]
    
    # 선택된 vertex들의 중심점과 가장 가까운 카메라 찾기
    closest_camera = min(chunk.cameras, 
                        key=lambda cam: ((cam.transform.translation()[0] - avg_coord[0])**2 + 
                                       (cam.transform.translation()[1] - avg_coord[1])**2 + 
                                       (cam.transform.translation()[2] - avg_coord[2])**2)
                        if cam.transform else float('inf'))
    
    if closest_camera.transform:  # 카메라의 위치 정보가 있는지 확인
        camera_position = closest_camera.transform.translation()
        
        # 드래그한 vertex들 중에서 카메라와 가장 가까운 vertex 찾기
        closest_vertex = min(vertex_index_to_coord.items(), 
                           key=lambda item: ((item[1][0] - camera_position[0])**2 + 
                                           (item[1][1] - camera_position[1])**2 + 
                                           (item[1][2] - camera_position[2])**2))
        
        # 가장 가까운 vertex에 마커 추가
        chunk.addMarker(closest_vertex[1], visibility=True)
        print(f"표면의 vertex {closest_vertex[0]}에 마커 추가: {closest_vertex[1]}")
        print(f"기준 카메라: {closest_camera.label}")
    else:
        print("카메라 위치 정보를 찾을 수 없습니다.")
else:
    print("선택된 vertex가 없습니다.")

# SAM을 하기 위한 마커 좌표
marker_coordinates(doc, img_dir)