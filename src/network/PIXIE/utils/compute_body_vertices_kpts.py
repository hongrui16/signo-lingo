import os, sys
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from tqdm import tqdm
import argparse
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import torch.nn as nn
import platform  # Import platform module to detect the operating system


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append('../')
from pixie_zoo.pixie import PIXIE
# from pixielib.pixie_parallel import PIXIE
from pixie_zoo.visualizer import Visualizer
from pixie_zoo.datasets.body_datasets import TestData
from pixie_zoo.utils import util
from pixie_zoo.utils.config import cfg as pixie_cfg
from pixie_zoo.utils.tensor_cropper import transform_points


class VisMeshPoints():
    def __init__(self, height=1000, width=1000, face_filepath = None):
        self.vis = o3d.visualization.Visualizer()
        #give the code, if on windows, set visible to True
        if platform.system() == 'Windows':
            visible = True
            print("Windows")
        elif platform.system() == 'Linux':
            visible = False
            print("Linux")

        self.vis.create_window(width=width, height=height, visible = visible)
        render_option = self.vis.get_render_option()
        if render_option is not None:
            render_option.light_on = True
            print("Render option is available.")
        else:
            print("Render option is not available.")


        self.vis.get_render_option().light_on = True
        if face_filepath is None:
            faces_filepath = 'data/SMPLX_NEUTRAL_2020.npz'
        all_data = np.load(faces_filepath, allow_pickle=True)
        self.faces = all_data['f']
        self.pcd = o3d.geometry.PointCloud()
        


    def vis_mesh(self, vertices, output_dir = None, name = None):
        mesh = o3d.geometry.TriangleMesh()

        # 设置网格的顶点
        mesh.vertices = o3d.utility.Vector3dVector(vertices)

        # 设置网格的三角形面
        mesh.triangles = o3d.utility.Vector3iVector(self.faces)

        # 计算顶点的法线
        mesh.compute_vertex_normals()

        # 重置视图
        self.vis.clear_geometries()
        self.vis.add_geometry(mesh)

        # 光照和相机视角设置
        ctr = self.vis.get_view_control()
        ctr.set_front([0, 0, 1])
        ctr.set_lookat([0, -0.65, 0])
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.6)

        transformation = o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi, np.pi))
        mesh.rotate(transformation, center=mesh.get_center())

        self.vis.update_geometry(mesh)
        self.vis.poll_events()
        self.vis.update_renderer()


        if not output_dir is None and not name is None:
            os.makedirs(output_dir, exist_ok=True)
            # 保存当前视角下的图像
            output_filepath = os.path.join(output_dir, f'{name}_mesh.jpg')
            self.vis.capture_screen_image(output_filepath)

    def vis_points(self, points, i, output_dir = None, name = None):
        self.pcd.points = o3d.utility.Vector3dVector(points)
        
        # 如果是第一次迭代，需要添加点云到可视化窗口
        if i == 0:
            self.vis.add_geometry(self.pcd)
        
        # 更新点云
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

        if not output_dir is None and not name is None:
            os.makedirs(output_dir, exist_ok=True)
            # 保存当前视角下的图像
            output_filepath = os.path.join(output_dir, f'{name}_points.jpg')
            self.vis.capture_screen_image(output_filepath)

    def destroy(self):
        self.vis.destroy_window()


class ComputeBodyVerticesKpts(nn.Module):
    def __init__(self, input_img_dir, save_img_dir = None, iscrop = True, args = None):
        self.input_img_dir = input_img_dir
        self.save_img_dir = save_img_dir
        self.iscrop = iscrop
        self.args = args

        ###        # 创建可视化窗口
        if platform.system() == 'Windows':
            self.mesh_point_visualizer = VisMeshPoints()
        else:
            pass
        
        

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
        else:
            self.device = torch.device('cpu')

        if save_img_dir is not None:
            os.makedirs(save_img_dir, exist_ok=True)



        # load test images 
        self.testdata = TestData(input_img_dir, iscrop=iscrop, body_detector='rcnn')

        #-- run PIXIE
        pixie_cfg.model.use_tex = True
        self.pixie = PIXIE(config = pixie_cfg, device=self.device)

        
        # point_cloud = o3d.geometry.PointCloud()
        
        # print('faces.shape', faces.shape)  
        # return



    def forward(self, debug = False):
        for i, batch in enumerate(tqdm(self.testdata, dynamic_ncols=True)):
            util.move_dict_to_device(batch, self.device)
            batch['image'] = batch['image'].unsqueeze(0)
            batch['image_hd'] = batch['image_hd'].unsqueeze(0)
            name = batch['name']
            name = os.path.basename(name)
            name = name.split('.')[0]
            # print(name)
            # frame_id = int(name.split('frame')[-1])
            # name = f'{frame_id:05}'

            data = {
                'body': batch
            }
            try:
                param_dict = self.pixie.encode(data, threthold=True, keep_local=True, copy_and_paste=False)
                
                # param_dict = pixie.encode(data, threthold=True, keep_local=True, copy_and_paste=True)
                # only use body params to get smplx output. TODO: we can also show the results of cropped head/hands
                moderator_weight = param_dict['moderator_weight']
                codedict = param_dict['body']

                opdict = self.pixie.decode(codedict, param_type='body')
                '''
                prediction = {
                            'vertices': verts,
                            'transformed_vertices': trans_verts,
                            'face_kpt': predicted_landmarks,
                            'smplx_kpt': predicted_joints,
                            'smplx_kpt3d': smplx_kpt3d,
                            'joints': joints,
                            'cam': cam,
                            }
                '''
                points = opdict['joints'].cpu().numpy().squeeze()            
                vertices = opdict['vertices'].cpu().numpy().squeeze()
                # print('points.shape', points.shape) #  (145, 3)
                # print('vertices.shape', vertices.shape) #(10475, 3)

                self.mesh_point_visualizer.vis_mesh(vertices, self.save_img_dir, name)
                self.mesh_point_visualizer.vis_points(points, i, self.save_img_dir, name)

                # 结束可视化
                
            except Exception as e:
                continue



        self.mesh_point_visualizer.destroy()
        print(f'-- please check the results in {self.save_img_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PIXIE')
    parser.add_argument('--input_img_dir', type=str, default='../TestSamples/body', help='input image directory')
    parser.add_argument('--save_img_dir', type=str, default='output_images', help='output image directory')
    parser.add_argument('--iscrop', type=bool, default=True, help='whether crop the image')
    args = parser.parse_args()

    compute_body_vertices_kpts = ComputeBodyVerticesKpts(args.input_img_dir, args.save_img_dir, args.iscrop, args)
    compute_body_vertices_kpts.forward()