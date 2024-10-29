import csv
import time
import Metashape

# 파일 경로
input_file = r"D:\Users\hoans\rist\sam\output.csv"
output_file = r"D:\Users\hoans\rist\sam\localcoord.csv"
coordinate_file = output_file  # 변환된 로컬 좌표를 다시 읽을 파일

tolerance = 0.001  # 좌표 매칭 시 허용 오차
highlight_color = (255, 0, 0)  # RGB 색상 (빨간색)

# 모델의 변환 행렬 및 역행렬 가져오기
chunk = Metashape.app.document.chunk
model_transform = chunk.model.transform
model_transform_inv = model_transform.inv()  # 글로벌 좌표에서 로컬 좌표로 변환할 역행렬

# 글로벌 좌표를 로컬 좌표로 변환 후 CSV 파일로 저장
with open(input_file, mode='r', newline='') as infile, open(output_file, mode='w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        vertex_index = int(row[0])  # 첫 번째 값은 vertex 인덱스
        global_coord = Metashape.Vector([float(row[1]), float(row[2]), float(row[3])])

        # 글로벌 좌표를 로컬 좌표로 변환
        local_coord = model_transform_inv.mulp(global_coord)

        # 변환된 로컬 좌표를 output CSV 파일에 작성
        writer.writerow([vertex_index] + list(local_coord))

# Function to read coordinates from a CSV file using csv module
def read_coordinates(file_path):
    coords = []
    with open(file_path, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 4:  # 최소 4개의 열이 있는지 확인
                x_str, y_str, z_str = row[1], row[2], row[3]  # 첫번째 열은 무시하고 2,3,4열 사용
                x = float(x_str)
                y = float(y_str)
                z = float(z_str)
                coords.append(Metashape.Vector([x, y, z]))
    return coords

# Function to select vertices by coordinates and change their color
def select_and_color_vertices_by_coords(chunk, coords, tolerance, highlight_color):
    coords_set = set((round(coord.x / tolerance), round(coord.y / tolerance), round(coord.z / tolerance)) for coord in coords)

    for vertex in chunk.model.vertices:
        v = vertex.coord
        key = (round(v.x / tolerance), round(v.y / tolerance), round(v.z / tolerance))
        
        if key in coords_set:
            # Change the color of the selected vertex
            vertex.color = highlight_color  # Setting the vertex color
        else:
            vertex.color = (255, 255, 255)  # Default color (white)

start_time = time.time()

# 로컬 좌표로 변환된 좌표를 읽어들이기
coordinates = read_coordinates(coordinate_file)

# 선택한 좌표에 맞는 정점 선택 및 색상 변경
select_and_color_vertices_by_coords(chunk, coordinates, tolerance, highlight_color)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Vertex coloring completed in: {elapsed_time:.6f} seconds")

Metashape.app.messageBox("Vertex coloring completed successfully.")