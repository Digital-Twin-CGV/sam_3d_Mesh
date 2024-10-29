import pandas as pd

# Step 1: Read the filtered coordinates CSV file into a DataFrame
filtered_df = pd.read_csv(r'D:\Users\hoans\rist\sam\sam_img\filtered_coordinates.csv', header=None, names=['index'])

# Step 2: Read the vertex 3D CSV file into a DataFrame
vertex_3d_df = pd.read_csv(r'D:\Users\hoans\rist\sam\vertex_3d.csv', header=None, names=['index', 'x', 'y', 'z'])

# Step 3: Filter the vertex_3d_df to include only those indices present in filtered_df
filtered_vertex_3d_df = vertex_3d_df[vertex_3d_df['index'].isin(filtered_df['index'])]

# Step 4: Keep only the columns 'index', 'x', 'y', and 'z' from the filtered DataFrame
final_output_df = filtered_vertex_3d_df[['index', 'x', 'y', 'z']]

# Step 5: Write the final DataFrame to 'output.csv'
final_output_df.to_csv(r'D:\Users\hoans\rist\sam\output.csv', index=False, header=False)
