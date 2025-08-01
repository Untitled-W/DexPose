import os
import torch
import pytorch_kinematics as pk
import numpy as np
import open3d as o3d


class HandModelURDF:
    
    def __init__(self, robot_name:str, urdf_path=None, mesh_path=None, device=None):
        
        """
        Parameters
        ----------
        mjcf_path: str
            path to mjcf file
        mesh_path: str
            path to mesh directory
        device: str | torch.Device
            device for torch tensors
        """
        # if device is None:
        #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # else:
        #     self.device = device
        self.device = 'cpu'

        if urdf_path is None:
            urdf_path = f'robot_models/{robot_name}.urdf'
            
        self.chain = pk.build_chain_from_urdf(open(urdf_path, "rb").read()).to(dtype=torch.float, device=self.device)
        print("n-DoF:", len(self.chain.get_joint_parameter_names()))
        self.n_dofs = len(self.chain.get_joint_parameter_names())
        
        self.mesh = {}

        def build_mesh_recurse(body):
            """hand-made a obj style mesh for each body

            Args:
                body (a chain object): 
            """
            # body is a pytorch.kinematics.frame.Frame object
            # attributes: name, link, joint, children

            if (len(body.link.visuals) > 0):
                link_name = body.link.name
                link_vertices = []
                link_faces = []
                n_link_vertices = 0
                for visual in body.link.visuals:
                    
                    if visual.geom_type == None:
                        continue

                    elif visual.geom_type == "box":

                        if isinstance(visual.geom_param, torch.Tensor):
                            scale = visual.geom_param.to(dtype=torch.float, device=device)
                        elif isinstance(visual.geom_param, list):
                            scale = torch.tensor(visual.geom_param, dtype=torch.float, device=device)
                        # Create an Open3D box mesh
                        box_mesh = o3d.geometry.TriangleMesh.create_box(
                            width=scale[0].item(), height=scale[1].item(), depth=scale[2].item()
                        )
                        box_mesh.compute_vertex_normals()
                        link_mesh = box_mesh

                    elif visual.geom_type == "capsule":

                        # Create an Open3D capsule mesh
                        radius = visual.geom_param[0]
                        half_height = visual.geom_param[1]
                        # Open3D does not have a direct capsule primitive, so we combine a cylinder and two spheres
                        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=half_height*2)
                        cylinder.compute_vertex_normals()
                        cylinder.translate((0, 0, -half_height))

                        sphere_top = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                        sphere_top.compute_vertex_normals()
                        sphere_top.translate((0, 0, half_height))

                        sphere_bottom = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                        sphere_bottom.compute_vertex_normals()
                        sphere_bottom.translate((0, 0, -half_height))

                        # Combine meshes
                        link_mesh = cylinder + sphere_top + sphere_bottom

                    else:

                        link_mesh = o3d.io.read_triangle_mesh(
                            os.path.join(mesh_path, visual.geom_param[0]))
                        
                        if len(visual.geom_param) > 1 and visual.geom_param[1] is not None:
                            scale = torch.tensor(visual.geom_param[1], dtype=torch.float, device=device)
                            # Scale mesh along x, y, z axes
                            vertices = np.asarray(link_mesh.vertices)
                            vertices *= scale.detach().cpu().numpy()
                            link_mesh.vertices = o3d.utility.Vector3dVector(vertices)


                    vertices = torch.from_numpy(np.asarray(link_mesh.vertices)).to(dtype=torch.float, device=device)
                    faces = torch.from_numpy(np.asarray(link_mesh.triangles)).to(dtype=torch.long, device=device)
                    
                    pos = visual.offset.to(dtype=torch.float, device=device)
                    
                    vertices = pos.transform_points(vertices)
                    link_vertices.append(vertices)
                    link_faces.append(faces + n_link_vertices)
                    n_link_vertices += len(vertices)

                if len(link_vertices) != 0: 

                    link_vertices = torch.cat(link_vertices, dim=0)
                    link_faces = torch.cat(link_faces, dim=0)

                    self.mesh[link_name] = {
                        'vertices': link_vertices,
                        'faces': link_faces,
                    }

                    self.mesh[link_name]['geom_param'] = body.link.visuals[0].geom_param
                    
                
            for children in body.children:
                build_mesh_recurse(children)
        
        build_mesh_recurse(self.chain._root)
        self.joints_names = []
        self.joints_lower = []
        self.joints_upper = []
        
        def set_joint_range_recurse(body):
            if body.joint.joint_type != "fixed":
                self.joints_names.append(body.joint.name)
                self.joints_lower.append(torch.tensor(body.joint.limits[0], dtype=torch.float, device=self.device) if body.joint.limits is not None else torch.tensor(-3.14, dtype=torch.float, device=self.device))
                self.joints_upper.append(torch.tensor(body.joint.limits[1], dtype=torch.float, device=self.device) if body.joint.limits is not None else torch.tensor(3.14, dtype=torch.float, device=self.device))
            for children in body.children:
                set_joint_range_recurse(children)
        
        set_joint_range_recurse(self.chain._root)
        self.joints_lower = torch.stack(
            self.joints_lower).float().to(self.device)
        self.joints_upper = torch.stack(
            self.joints_upper).float().to(self.device)
        
        self.link_name_to_link_index = dict(zip([link_name for link_name in self.mesh], range(len(self.mesh))))


    def set_qpos(self, qpos):
        """
        Set translation, rotation, and joint angles of grasps
        
        Parameters
        ----------
        hand_pose: (3+3+`n_dofs`), not in batch
            translation, rotation in axis_angle, and joint angles
        """
        self.qpos = qpos
        
        self.state = self.chain.forward_kinematics(qpos)
        

    def get_hand_mesh(self):
        
        data = o3d.geometry.TriangleMesh()
        for link_name in self.mesh:
            
            # a tensor with shape (N,3)
            v = self.mesh[link_name]['vertices']
            
            # a Transform3d object
            transform = self.state[link_name]

            v = transform.transform_points(v).detach().cpu()
            f = self.mesh[link_name]['faces'].detach().cpu()

            data += o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(v.numpy()),
                triangles=o3d.utility.Vector3iVector(f.numpy())
            )

        # data = data.simplify_quadric_decimation(10000)
        data = data.simplify_vertex_clustering(voxel_size=0.005)
        data.compute_vertex_normals()
            
        return data
    

def load_robot(robot_name_str: str, side):

    def urdf_name_map(robot_name, side):
        name = {
            'shadow': 'shadow_',
            'leap': 'leap_',
            'svh': 'schunk_svh_',
            'allegro': 'allegro_',
            'barrett': 'barrett_',
            'inspire': 'inspire_',
            'panda': 'panda_gripper',
        }
        if robot_name == 'panda': return name[robot_name]
        return name[robot_name] + 'hand_' + side

    asset_name_map = {
        'shadow': 'shadow_hand',
        'leap': 'leap_hand',
        'svh': 'schunk_hand',
        'allegro': 'allegro_hand',
        'barrett': 'barrett_hand',
        'inspire': 'inspire_hand',
        'panda': 'panda_gripper',
    }

    robot = HandModelURDF(robot_name_str,
                          f'/home/qianxu/Desktop/Project/DexPose/retarget/urdf/{urdf_name_map(robot_name_str, side)}_glb.urdf',
                          f'/home/qianxu/Desktop/Project/DexPose/thirdparty/dex-retargeting/assets/robots/hands/{asset_name_map[robot_name_str]}/meshes',
                        )

    return robot