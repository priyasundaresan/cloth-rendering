import bpy
import bpy, bpy_extras
from math import *
from mathutils import *
import random
import numpy as np
from random import sample
import bmesh

'''Usage: blender -b -P cloth-blender.py'''

def clear_scene():
    '''Clear existing objects in scene'''
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def make_table():
    # Generate table surface
    bpy.ops.mesh.primitive_plane_add(size=3, location=(0,0,0))
    bpy.ops.object.modifier_add(type='COLLISION')
    return bpy.context.object

def make_cloth():
    '''Create cloth and generate new state'''
    # Generate a rigid cloth, add cloth and collision physics
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0,0,0))
    bpy.ops.object.modifier_add(type='COLLISION')
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.subdivide(number_cuts=25) # Tune this number for cloth detail
    bpy.ops.object.editmode_toggle()
    bpy.ops.object.modifier_add(type='CLOTH')
    bpy.ops.object.modifier_add(type='SUBSURF')
    bpy.context.object.modifiers["Subdivision"].levels=3 # Smooths the cloth so it doesn't look blocky
    bpy.context.object.modifiers["Cloth"].collision_settings.use_self_collision = True
    return bpy.context.object

def generate_cloth_state(cloth):
    # Move cloth slightly above the table and simulate a drop
    # Pinned group is the vertices that should not fall
    if cloth is None:
        cloth = make_cloth()
    dx = np.random.uniform(0,0.7,1)*random.choice((-1,1))
    dy = np.random.uniform(0,0.7,1)*random.choice((-1,1))
    dz = np.random.uniform(0.4,0.8,1)
    cloth.location = (dx,dy,dz)
    cloth.rotation_euler = (0, 0, random.uniform(-np.pi, np.pi)) # fixed z, rotate only about x/y axis slightly
    if 'Pinned' in cloth.vertex_groups:
        cloth.vertex_groups.remove(cloth.vertex_groups['Pinned'])
    pinned_group = bpy.context.object.vertex_groups.new(name='Pinned')
    n = random.choice(range(1,4)) # Number of vertices to pin
    subsample = sample(range(len(cloth.data.vertices)), n)
    pinned_group.add(subsample, 1.0, 'ADD')
    cloth.modifiers["Cloth"].settings.vertex_group_mass = 'Pinned'
    bpy.context.scene.frame_end = 30 # Roughly when the cloth settles
    return cloth

def reset_cloth(cloth):
    cloth.modifiers["Cloth"].settings.vertex_group_mass = ''
    cloth.location = (0,0,0)
    bpy.context.scene.frame_set(0)

def set_viewport_shading(mode):
    '''Makes color/texture viewable in viewport'''
    areas = bpy.context.workspace.screens[0].areas
    for area in areas:
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = mode

def pattern(obj, texture_filename):
    '''Add image texture to object'''
    mat = bpy.data.materials.new(name="ImageTexture")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
    texImage.image = bpy.data.images.load(texture_filename)
    mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
    obj.data.materials.append(mat)

def colorize(obj, color):
    '''Add color to object'''
    mat = bpy.data.materials.new(name="Color")
    mat.use_nodes = False
    mat.diffuse_color = color
    obj.data.materials.append(mat)
    set_viewport_shading('MATERIAL')

def add_camera_light():
    bpy.ops.object.light_add(type='SUN', radius=1, location=(0,0,0))
    bpy.ops.object.camera_add(location=(0,0,8), rotation=(0,0,0))
    bpy.context.scene.camera = bpy.context.object

def render(filename, engine, episode):
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.filepath = "./images/{}".format(filename)
    scene.view_settings.exposure = 1.4
    if engine == 'BLENDER_WORKBENCH':
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.display_mode
        scene.display_settings.display_device = 'None'
        scene.sequencer_colorspace_settings.name = 'XYZ'
        scene.render.image_settings.file_format='PNG'
    elif engine == "BLENDER_EEVEE":
        scene.eevee.taa_samples = 1
        scene.eevee.taa_render_samples = 1
    for frame in range(scene.frame_start, scene.frame_end+1):
        scene.render.filepath = filename % ((scene.frame_end - scene.frame_start)*episode + frame)
        scene.frame_set(frame)
        bpy.ops.render.render(write_still=True)

def render_dataset(num_episodes, filename, texture_filepath='', color=None):
    clear_scene()
    add_camera_light()
    table = make_table()
    cloth = make_cloth()
    if texture_filepath:
        engine = 'BLENDER_EEVEE'
        pattern(cloth, texture_filepath)
    elif color:
        engine = 'BLENDER_WORKBENCH'
        colorize(cloth, color)
    for i in range(num_episodes):
        reset_cloth(cloth)
        cloth = generate_cloth_state(cloth)
        render(filename, engine, i)
    
if __name__ == '__main__':
    texture_filepath = 'textures/cloth.jpg'
    green = (0,0.5,0.5,1)
    filename = "rgb_%06d.png"
    episodes = 2
    render_dataset(episodes, filename, color=green)
