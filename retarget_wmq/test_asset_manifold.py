import open3d as o3d
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

obj_folder = r"thirdparty\dex-retargeting\assets\robots\hands\shadow_hand\meshes\visual"

def mesh_to_plotly(mesh: o3d.geometry.TriangleMesh, non_manifold_vertices=None, non_manifold_faces_indices=None):
    """
    Open3D mesh -> [Plotly Mesh3d, Scatter3d(edges)]
    返回 list 可直接 unpack 进 go.Figure(data=[...])
    """
    # --------- 面片 ---------
    v = pd.DataFrame(np.asarray(mesh.vertices), columns=['x', 'y', 'z'])
    f = np.asarray(mesh.triangles)
    face_trace = go.Mesh3d(
        x=v['x'], y=v['y'], z=v['z'],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color="#86B8EA",
        flatshading=True,
        opacity=0.9,
        name='faces'
    )
    yield face_trace

    # --------- 边 ---------
    edges = set()
    for tri in f:
        edges.add(tuple(sorted([tri[0], tri[1]])))
        edges.add(tuple(sorted([tri[1], tri[2]])))
        edges.add(tuple(sorted([tri[2], tri[0]])))
    edges = np.array(list(edges))  # (E,2)

    # 顶点坐标序列：每条线段两个端点
    x_lines = []
    y_lines = []
    z_lines = []
    for e in edges:
        x_lines += [v.iloc[e[0]]['x'], v.iloc[e[1]]['x'], None]
        y_lines += [v.iloc[e[0]]['y'], v.iloc[e[1]]['y'], None]
        z_lines += [v.iloc[e[0]]['z'], v.iloc[e[1]]['z'], None]

    edge_trace = go.Scatter3d(
        x=x_lines, y=y_lines, z=z_lines,
        mode='lines',
        line=dict(color="#2E0101", width=2),
        name='edges',
        hoverinfo='skip'
    )
    yield edge_trace

    # --------- 非流形顶点 ---------
    if non_manifold_vertices is not None:
        if len(non_manifold_vertices) == 0:
            nm_v = pd.DataFrame([[np.nan, np.nan, np.nan]], columns=['x', 'y', 'z'])
        else:
            nm_v = pd.DataFrame(np.asarray(non_manifold_vertices), columns=['x', 'y', 'z'])
        non_manifold_trace = go.Scatter3d(
            x=nm_v['x'], y=nm_v['y'], z=nm_v['z'],
            mode='markers',
            marker=dict(color='red', size=3),
            name='non-manifold vertices',
            hoverinfo='skip'
        )
        yield non_manifold_trace
    # else: yield go.Scatter3d(x=[], y=[], z=[], mode='markers', marker=dict(size=0), name='non-manifold vertices')
        
    if non_manifold_faces_indices is not None:
        edges_non = set()
        for tri in f[non_manifold_faces_indices]:
            edges_non.add(tuple(sorted([tri[0], tri[1]])))
            edges_non.add(tuple(sorted([tri[1], tri[2]])))
            edges_non.add(tuple(sorted([tri[2], tri[0]])))
        edges_non = np.array(list(edges_non))  # (E,2)
        # 顶点坐标序列：每条线段两个端点
        x_lines = []
        y_lines = []
        z_lines = []
        if len(edges_non) == 0:
            x_lines = y_lines = z_lines = [np.nan, np.nan, None]
        else:
            for e in edges_non:
                x_lines += [v.iloc[e[0]]['x'], v.iloc[e[1]]['x'], None]
                y_lines += [v.iloc[e[0]]['y'], v.iloc[e[1]]['y'], None]
                z_lines += [v.iloc[e[0]]['z'], v.iloc[e[1]]['z'], None]

        edge_trace_non = go.Scatter3d(
            x=x_lines, y=y_lines, z=z_lines,
            mode='lines',
            line=dict(color="#DD1AEF", width=2),
            name='non-manifold edges',
            hoverinfo='skip'
        )
        yield edge_trace_non
    # else: yield go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(width=0), name='non-manifold edges')

# 寻找obj_folder下的所有obj文件
obj_files = [f for f in os.listdir(obj_folder) if f.endswith('.obj')]
fig = go.Figure()

print(f"{'link_name':<30} │ edge_manifold │ vertex_manifold(num) │ self_intersect │  before (V/F)")
print("─" * 105)
# 为每个obj文件生成plotly data
frames = []
for obj_file in obj_files:
    obj_path = os.path.join(obj_folder, obj_file)
    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.compute_vertex_normals()

    non_manifold_vertices_index = mesh.get_non_manifold_vertices()  # 计算非流形顶点
    non_manifold_vertices = np.asarray(mesh.vertices)[non_manifold_vertices_index]

    # 计算每个非流形顶点包含的面的索引
    non_manifold_faces_indices = []
    for index in non_manifold_vertices_index:
        for idx, faces in enumerate(np.asarray(mesh.triangles).tolist()):
            if index in faces:
                non_manifold_faces_indices.append(idx)

    is_edge_manifold = mesh.is_edge_manifold()
    is_self_intersecting = mesh.is_self_intersecting()
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    print(f"{obj_file.split('.')[0]:<30} │ {is_edge_manifold!s:<13} │ {len(non_manifold_vertices_index)!s:<20} │ "
      f"{is_self_intersecting!s:<14} │ {len(vertices):>5} / {len(faces):>5}")

    data = mesh_to_plotly(mesh, non_manifold_vertices, non_manifold_faces_indices)
    frames.append(go.Frame(data=list(data), name=obj_file))

# 初始化显示第一个mesh
initial_data = frames[0].data
fig = go.Figure(
    data=list(initial_data),
    frames=frames
)
slider_steps = []
for i, obj_file in enumerate(obj_files):
    slider_steps.append(dict(
        method="animate",
        args=[[obj_file], {"frame": {"duration": 500, "redraw": True},
                           "mode": "immediate",
                           "transition": {"duration": 300}}],
        label=obj_file
    ))
fig.update_layout(
    title="3D Mesh Viewer",
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='data'
    ),
    updatemenus=[dict(
        type="buttons",
        buttons=[dict(label="Play",
                      method="animate",
                      args=[None, {"frame": {"duration": 500, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 300}}]),
                 dict(label="Pause",
                      method="animate",
                      args=[[None], {"frame": {"duration": 0, "redraw": False},
                                     "mode": "immediate",
                                     "transition": {"duration": 0}}])]
    )],
    sliders=[dict(
        active=0,
        currentvalue={"prefix": "Current Mesh: ", "visible": True, "xanchor": "right"},
        pad={"b": 10, "t": 50},
        steps=slider_steps
    )]
)

fig.write_html("mesh_viewer.html", auto_open=True)