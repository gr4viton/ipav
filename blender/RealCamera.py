
# http://blender.stackexchange.com/questions/41533/how-to-remotely-run-a-python-script-in-an-existing-blender-instance

# Script to run from blender:
#   blender --python blender_server.py

import sys
import pickle

import bpy
import os
import datetime as dt
import numpy as np


sys.path.append("D:/DEV/PYTHON/pyCV/kivyCV_start/blender")
import render


# HOST = "127.0.0.1"
# HOST = "192.168.1.100"
PATH_MAX = 4096

def printl():
    print('_'*42)

import bpy, mathutils, math
from mathutils import Vector, Euler
from math import pi

import time


class RealCamera():
    pos = Vector((0,0,0))
    rot = Vector((0,0,0))
    # height, width, distance of the plane == aspect ratios, distance
    size = Vector((0,0,0))

    proj = None
    rcam = None
    hull = None

    source_name_delimiter = '~'
    focal = 400
    id = 0
    resolution = [320,240]

    def __init__(self, name, pos, rot, eul, size):
        self.name = name
        self.pos = Vector(pos)
        self.rot = Vector(rot)
        self.eul = Vector(eul)
        self.size = Vector(size)

        self.init_source_name()

    def __init__(self, name, pos, rot, eul, size, rcam_obj):
        self.name = name
        self.pos = Vector(pos)
        self.rot = Vector(rot)
        self.eul = eul
        self.size = Vector(size)
        self.rcam = rcam_obj

        self.init_source_name()

    def init_source_name(self):
        self.source_name = self.name.split(self.source_name_delimiter )[1]
        print(self.source_name)

    def init_rot(self, real_point, pixel):
        """
        initializes rotation from real space point prjs to camera plane
        """
    def create_iso(self):
        cubeobject = bpy.ops.mesh.primitive_ico_sphere_add
        #    cursor = context.scene.cursor_location
        #   x = cursor.x
        #  y = cursor.y
        # z = cursor.z
        cubeobject(location=(1, 1, 1))

    def create_cone(self):
        new_cone = bpy.ops.mesh.primitive_cone_add(
            vertices=4, radius1=2, depth=10.0,
            location=self.pos, rotation=self.rot)
        print("created new_cone =", new_cone)
        # scn = bpy.context.scene.GetCurrent()
        # scn.objects.new(new_cone, 'cone')

    def delete_object(self, object):
    #     bpy.ops.object.select_all(action='DESELECT')
    #     #rcam_d = bpy.data.objects['prj.001']
    #     object.select = True
    #     bpy.ops.object.delete()
    #
    #     scene = bpy.context.scene
    # #    scene.objects.link(rcam_d)
    #     scene.update()

        if object == None:
            print('Object could not be deleted')
            return
        scene = bpy.context.scene
        # [print(obj) for obj in scene.objects]
        # if object in scene.objects:
        scene.objects.unlink(object)
        bpy.data.objects.remove(object)
        scene.update()

        # object.hide
        # when it is unlinked from scene, it won't be saved
        object = None

        # object.delete()

    def delete_projection(self):
        self.delete_object(self.proj)
        self.proj = None


    def setMaterial(ob, mat):
        me = ob.data

        me.materials.append(mat)

    def makeMaterial(name, diffuse, specular, alpha):
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = diffuse
        mat.diffuse_shader = 'LAMBERT'
        mat.diffuse_intensity = 1.0
        mat.specular_color = specular
        mat.specular_shader = 'COOKTORR'
        mat.specular_intensity = 0.5
        mat.alpha = alpha
        mat.ambient = 1
        mat.use_transparency = True
        return mat


    def duplicate(self, obj, duplicate_name, alpha=1):

        obj_duplicate = obj.copy()
        obj_duplicate.name = duplicate_name
        if alpha < 1:

            if self.blue_mat is None:
                self.blue_mat = self.makeMaterial('BlueSemi', (0,0,1), (0.5,0.5,0), 0.5)
                print('Created material blue')
            # set transparency
            self.setMaterial(obj, self.blue_mat)
            pass

        scene = bpy.context.scene
        scene.objects.link(obj_duplicate)
        scene.update()

        return obj_duplicate

    def get_contour_projection_object(self):
            rcam_eul = self.eul
            rcam_pos = self.pos
            # print('%'*42, rcam_pos)
            rcam_rot = self.rot

            rcam_sca = self.size
            # rcam_sca = (640,480,5)
            # rcam_sca = (320,240, self.focal)

            # rcam_rot = np.deg2rad(rcam_rot)

            rcam_rot_euler = Euler(rcam_rot, rcam_eul)

            # x = 200
            # h, v = (640/x, 480/x)
            # focal = 300/x
            # h, v, focal = rcam_sca
            # eye_center = [(h/2, v/2, focal*80)]
            h, v = self.resolution
            focal = self.focal
            eye_center = [(h/2, v/2, focal)]

            print(eye_center)

            hull = self.hull

            # map to 3d coordinates
            hull_xyz = [(x,y,0) for (x,y) in hull]

            # shift origin to eye_center
            verts = eye_center + hull_xyz
        #    print('verts=\n', '\n'.join([str(v) for v in verts]) )

            # generate faces from vertexes
            start_index = 1
            hull_max_index = len(hull_xyz) + start_index -1

            faces_eye = [ (x, x+1, 0) for x in range(start_index, hull_max_index)]
            faces_eye += [(hull_max_index, start_index, 0)]

            faces_canvas = [ (x+1, x, start_index) for x in range(start_index+1, hull_max_index)]
        #    print('faces_eye[',len(faces_eye),']=',faces_eye)
        #    print('faces_canvas[',len(faces_canvas),']=',faces_canvas)

            faces = faces_eye + faces_canvas

            # create mesh object from faces and vertices
            mesh_data = bpy.data.meshes.new("hull_mesh")
            mesh_data.from_pydata(verts, [], faces)
            mesh_data.update()

            hull_name = "hull_" + self.name
            hull = bpy.data.objects.new(hull_name, mesh_data)

            scene = bpy.context.scene
            scene.objects.link(hull)
            scene.update()

            new_origin = eye_center[0]
            neg_new_origin = [-x for x in new_origin]
            hull.data.transform(mathutils.Matrix.Translation(neg_new_origin))
        #    hull.matrix_world.translation += new_origin

            hull.location = rcam_pos
            hull.rotation_euler = rcam_rot_euler

            scene.update()

            return hull

    def create_projection(self):
        if self.rcam:
            if self.proj:
                print('Deleting old projection object')
                self.delete_projection()

            if self.hull == None:
                # if no projection data -> use scaled camera pyramid
                self.proj = self.rcam.copy()
                self.proj.scale = Vector((1, 1, 3))
            else:
                print('Going to [get_contour_projection_object]')
                proj = self.get_contour_projection_object()

                name = 'prj_hull_' + self.rcam.name[-4:]

                self.proj = proj

                # # keep the projections visible
                keep_prjs = True
                if keep_prjs == True:
                    alpha = 0.5
                    alpha = 1
                    proj_original = self.duplicate(proj, name, alpha)
                    self.proj = proj_original

            # self.proj.name = 'prj' + self.rcam.name[-4:]
            #    rcam_d.location += Vector((1,1,1))
            #    rcam_d.rotation_euler = Vector((pi,0,0))

            scene = bpy.context.scene
            # scene.objects.link(self.proj)
            scene.update()



# import bpy, bmesh
#
# verts = ((0, 3),(2.5, 0.5),(5, 1),(4.5, 3.5),(10.5, 2),(8, 10),(7, 4.5),(2, 6))
#
# bm = bmesh.new()
# for v in verts:
#     bm.verts.new((v[0], v[1], 0))
# bm.faces.new(bm.verts)
#
# bm.normal_update()
#
# me = bpy.data.meshes.new("")
# bm.to_mesh(me)
#
# ob = bpy.data.objects.new("", me)
# bpy.context.scene.objects.link(ob)
# bpy.context.scene.update()