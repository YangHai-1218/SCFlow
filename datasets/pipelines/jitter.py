from typing import Optional
import numpy as np
from scipy.spatial.transform import Rotation

from .builder import PIPELINES
from ..pose import load_mesh, eval_rot_error

@PIPELINES.register_module()
class PoseJitter:
    def __init__(self, 
                jitter_angle_dis:list,
                jitter_x_dis: list,
                jitter_y_dis: list,
                jitter_z_dis: list,
                jitter_pose_field: list,
                jittered_pose_field: list,
                add_limit: Optional[float]=None,
                translation_limit: Optional[float]=None,
                angle_limit: Optional[float]=None,
                mesh_dir: Optional[str]=None,
                mesh_diameter: Optional[list]=None):
        assert isinstance(jitter_angle_dis, (list, tuple))
        assert isinstance(jitter_x_dis, (list, tuple))
        assert isinstance(jitter_y_dis, (list, tuple))
        assert isinstance(jitter_z_dis, (list, tuple))
        assert len(jitter_angle_dis) == 2
        assert len(jitter_z_dis) == 2 and len(jitter_x_dis) == 2 and len(jitter_y_dis) == 2
        self.jitter_angle_dis = jitter_angle_dis
        self.jitter_x_dis = jitter_x_dis
        self.jitter_y_dis = jitter_y_dis
        self.jitter_z_dis = jitter_z_dis
        assert isinstance(jitter_pose_field, (list, tuple))
        assert isinstance(jittered_pose_field, (list, tuple))
        assert len(jittered_pose_field) == len(jitter_pose_field)
        assert 'rotation' in jitter_pose_field[0]
        assert 'translation' in jitter_pose_field[1]
        assert 'rotation' in jittered_pose_field[0]
        assert 'translation' in jittered_pose_field[1]
        self.jitter_pose_field = jitter_pose_field
        self.jittered_pose_field = jittered_pose_field
        self.angle_limit = angle_limit
        self.translation_limit = translation_limit
        self.add_limit = add_limit
        if add_limit is not None:
            assert mesh_dir is not None and mesh_vertices is not None
            self.meshes = load_mesh(mesh_dir)
            mesh_vertices = [mesh.vertices.view(np.ndarray).astype(np.float32) for mesh in self.meshes]
            self.mesh_vertices = [vertices[np.random.choice(vertices.shape[0], 1000)] for vertices in mesh_vertices]
            self.mesh_diameters = mesh_diameter
        
    def jitter(self, rotation, translation, label):
        found_proper_jitter_flag = False
        while not found_proper_jitter_flag:
            angle = [np.random.normal(self.jitter_angle_dis[0], self.jitter_angle_dis[1]) for _ in range(3)]
            delta_rotation = Rotation.from_euler('zyx', angle, degrees=True).as_matrix().astype(np.float32)
            jittered_rotation = np.matmul(delta_rotation, rotation)
            rotation_error = eval_rot_error(rotation[None], jittered_rotation[None])[0]
            if self.angle_limit is not None and rotation_error > self.angle_limit:
                continue

            # translation jitter
            x_noise = np.random.normal(loc=self.jitter_x_dis[0], scale=self.jitter_x_dis[1])
            y_noise = np.random.normal(loc=self.jitter_y_dis[0], scale=self.jitter_y_dis[1])
            z_noise = np.random.normal(loc=self.jitter_z_dis[0], scale=self.jitter_z_dis[1])
            translation_noise = np.array([x_noise, y_noise, z_noise], dtype=np.float32)
            translation_error = np.linalg.norm(translation_noise)
            if self.translation_limit is not None and translation_error > self.translation_limit:
                continue
            jittered_translation = translation + translation_noise
            if self.add_limit is not None:
                verts = self.mesh_vertices[label]
                gt_points = (np.matmul(rotation, verts.T) + translation[:, None]).T
                ref_points = (np.matmul(jittered_rotation, verts.T) + jittered_translation[:, None]).T
                add_error = np.linalg.norm(gt_points - ref_points, axis=-1).mean() / self.mesh_diameters[label]
                if add_error > self.add_limit:
                    continue
            else:
                add_error = None
            return jittered_rotation, jittered_translation, add_error, translation_error, rotation_error

    
    def __call__(self, results):
        rotations, translations = results[self.jitter_pose_field[0]], results[self.jitter_pose_field[1]]
        labels = results['labels']
        k = results['k']
        num_obj = len(rotations)
        if k.ndim == 2:
            k = np.repeat(k[None], num_obj, axis=0)

        jittered_rotations, jittered_translations = [], []
        add_error_list, trans_error_list, rot_error_list = [], [], []
        for i in range(num_obj):
            jittered_rotation, jittered_translation, add_error, rotation_error, translation_error = self.jitter(rotations[i], translations[i], labels[i])
            jittered_translations.append(jittered_translation)
            jittered_rotations.append(jittered_rotation)
            add_error_list.append(add_error)
            rot_error_list.append(rotation_error)
            trans_error_list.append(translation_error)
        add_error = np.array(add_error_list).reshape(-1)
        trans_error = np.array(trans_error_list).reshape(-1)
        rot_error = np.array(rot_error_list).reshape(-1)
        jittered_translations = np.stack(jittered_translations, axis=0)
        jittered_rotations = np.stack(jittered_rotations, axis=0)
        results[self.jittered_pose_field[0]] = jittered_rotations
        results[self.jittered_pose_field[1]] = jittered_translations
        results['init_add_error'] = add_error
        results['init_trans_error'] = trans_error
        results['init_rot_error'] = rot_error
        return results