import bpy, mathutils, math
from mathutils import Vector
from math import pi
import os
import time
from datetime import datetime

def addTrackToConstraint(ob, name, target):
    cns = ob.constraints.new('TRACK_TO')
    cns.name = name
    cns.target = target
    cns.track_axis = 'TRACK_NEGATIVE_Z'
    cns.up_axis = 'UP_Y'
    cns.owner_space = 'WORLD'
    cns.target_space = 'WORLD'
    return

def createLamp(name, lamptype, loc):
    bpy.ops.object.add(
        type='LAMP',
        location=loc)        
    ob = bpy.context.object
    ob.name = name
    lamp = ob.data
    lamp.name = 'Lamp'+name
    lamp.type = lamptype
    return ob


def createLamps(origin, target):
    deg2rad = 2*pi/360
 
    sun = createLamp('sun', 'SUN', origin+Vector((0,10,5)))
    lamp = sun.data
    lamp.type = 'SUN'
    addTrackToConstraint(sun, 'TrackMiddle', target)
 
    for ob in bpy.context.scene.objects:
        if ob.type == 'MESH':
            spot = createLamp(ob.name+'Spot', 'SPOT', ob.location+Vector((0,2,1)))
            bpy.ops.transform.resize(value=(0.5,0.5,0.5))
            lamp = spot.data
 
            # Lamp
            lamp.type = 'SPOT'
            lamp.color = (0.5,0.5,0)
            lamp.energy = 0.9
            lamp.falloff_type = 'INVERSE_LINEAR'
            lamp.distance = 7.5
 
            # Spot shape
            lamp.spot_size = 30*deg2rad
            lamp.spot_blend = 0.3
 
            # Shadows
            lamp.shadow_method = 'BUFFER_SHADOW'
            lamp.use_shadow_layer = True
            lamp.shadow_buffer_type = 'REGULAR'
            lamp.shadow_color = (0,0,1)
 
            addTrackToConstraint(spot, 'Track'+ob.name, ob)
    return
 

def createCamera(origin, target):
    # Create object and camera
    bpy.ops.object.add(
        type='CAMERA',
        location=origin,
        rotation=(pi/2,0,pi))        
    ob = bpy.context.object
    ob.name = 'MyCamOb'
    cam = ob.data
    cam.name = 'MyCam'
    addTrackToConstraint(ob, 'TrackMiddle', target)
 
 
    # Lens
    cam.type = 'PERSP'
    cam.lens = 75
    cam.lens_unit = 'MILLIMETERS'
    cam.shift_x = -0.05
    cam.shift_y = 0.1
    cam.clip_start = 10.0
    cam.clip_end = 250.0
 
    empty = bpy.data.objects.new('DofEmpty', None)
    empty.location = origin+Vector((0,10,0))
    cam.dof_object = empty
 
    # Display
#    cam.show_title_safe = True
    cam.show_name = True
 
    # Make this the current camera
    scn = bpy.context.scene
    scn.camera = ob
    return ob



def run(origin):
    # Delete all old cameras and lamps
    scn = bpy.context.scene
    for ob in scn.objects:
        if ob.type == 'CAMERA' or ob.type == 'LAMP':
            scn.objects.unlink(ob)
 
    # Add an empty at the middle of all render objects
    midpoint = Vector((1,1,1))
    bpy.ops.object.add(
        type='EMPTY',
        location=midpoint),
    target = bpy.context.object
    target.name = 'Target'
 
    createCamera(origin+Vector((10,18,10)), target)
    createLamps(origin, target)
    
    add_isosphere()
    render()

 
 
def add_isosphere():
    cubeobject = bpy.ops.mesh.primitive_ico_sphere_add

#    cursor = context.scene.cursor_location
 #   x = cursor.x 
  #  y = cursor.y 
   # z = cursor.z
    cubeobject(location=(1, 1, 1))

def render():
    # path = os.path.abspath('D:/DEV/PYTHON/pyCV/blender_out/image.jpg')
    # i = time.strftime("%Y-%m-%d %H_%M_%S", time.gmtime)
    i = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    path = os.path.abspath(''.join(['D:/DEV/PYTHON/pyCV/kivyCV_start/blender/pic/',str(i),'image.jpg']))
    bpy.data.scenes['Scene'].render.filepath = path
    bpy.ops.render.render( write_still=True ) 

if __name__ == "__main__":
    run(Vector((0,0,0)))

