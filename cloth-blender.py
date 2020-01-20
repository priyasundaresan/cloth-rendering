import bpy
import os
import cv2
import json
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
    bpy.ops.mesh.primitive_plane_add(size=5, location=(0,0,0))
    bpy.ops.object.modifier_add(type='COLLISION')
    return bpy.context.object

def make_polygon(subdivisions):
    verts = []
    for x in np.linspace(-1,1,subdivisions):
        for y in np.linspace(-1,1,subdivisions):
            verts.append((x,y,0))
    edges = []
    for i in range(len(verts)):
        # For each vertex, add an edge right & down
        right = i+1
        down = i+subdivisions
        if i%subdivisions != (subdivisions-1) and right < len(verts):
            edges.append((i, right))
        if down < len(verts):
            edges.append((i, down))
    return verts, edges

def make_cloth(subdivisions):
    '''Create cloth and generate new state'''
    # Make cloth with ordered vertices (up-down and left-right)
    verts, edges = make_polygon(subdivisions) # This makes cloth vertices ordered nicely (top-bottom, left-right)
    faces = []
    mesh = bpy.data.meshes.new("Plane")
    obj = bpy.data.objects.new("Cloth", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    mesh = obj.data
    mesh.from_pydata(verts, edges, faces)
    mesh.update()
    bpy.ops.object.editmode_toggle()
    bm = bmesh.from_edit_mesh(mesh)
    bmesh.ops.edgenet_fill(bm, edges=bm.edges)
    bmesh.update_edit_mesh(mesh, True)
    bpy.ops.object.editmode_toggle()

    cloth = bpy.context.object
    # Add cloth and collision physics
    bpy.ops.object.modifier_add(type='CLOTH')
    cloth.modifiers["Cloth"].collision_settings.use_self_collision = True
    bpy.ops.object.modifier_add(type='SUBSURF')
    cloth.modifiers["Subdivision"].levels=3 # Smooths the cloth so it doesn't look blocky
    cloth.modifiers["Subdivision"].show_on_cage = True
    return cloth

def render_action_sequence(cloth, v_idx, actions):
    scene = bpy.context.scene
    render_path = set_render_settings('BLENDER_WORKBENCH', 'actions', '%06d_rgb.png')

    for frame in range(0, 30):
        if frame%3==0:
            index = frame//3 
            scene.render.filepath = render_path % index
            bpy.ops.render.render(write_still=True)
        scene.frame_set(frame)

    unpin(cloth)

    for frame in range(30, 75):
        if frame%3==0:
            index = frame//3 
            scene.render.filepath = render_path % index
            bpy.ops.render.render(write_still=True)
        scene.frame_set(frame)

    cloth_deformed = update(cloth)
    pick = cloth_deformed.matrix_world @ cloth_deformed.data.vertices[v_idx].co

    bpy.ops.object.armature_add(location=pick)
    arma = bpy.data.objects['Armature']
    bpy.ops.object.select_all(action='DESELECT')
    arma.select_set(state=True)
    bpy.context.view_layer.objects.active = arma
    bpy.ops.object.mode_set(mode='EDIT')
    parent_bone = 'Bone' # choose the bone name which you want to be the parent
    arma.data.edit_bones.active = arma.data.edit_bones[parent_bone]
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT') #deselect all objects
    cloth.select_set(state=True)
    arma.select_set(state=True)
    bpy.context.view_layer.objects.active = arma
    bpy.ops.object.parent_set(type='ARMATURE_NAME')
    bpy.context.view_layer.objects.active = cloth
    bpy.ops.object.modifier_move_up(modifier="Armature")
    bpy.ops.object.modifier_move_up(modifier="Armature")
    cloth.vertex_groups['Bone'].name = 'Pinned'

    bpy.ops.object.select_all(action='DESELECT')
    pinned_group = cloth.vertex_groups['Pinned']
    pinned_group.add([v_idx], 1.0, 'ADD')
    cloth.modifiers["Cloth"].settings.vertex_group_mass = 'Pinned'

    bpy.context.view_layer.objects.active = arma
    bpy.ops.object.mode_set(mode='POSE')
    bone=bpy.context.object.pose.bones['Bone']

    bone.keyframe_insert(data_path='location',frame=frame)
    frame_offset = (scene.frame_end - 75)//len(actions)
    for i, (dx,dy,dz) in enumerate(actions):
        bone.location[0] += dx
        bone.location[1] += dy
        bone.location[2] += dz
        bone.keyframe_insert(data_path='location',frame=frame + i*frame_offset)

    for frame in range(75, scene.frame_end):
        if frame%3==0:
            index = frame//3 
            scene.render.filepath = render_path % index
            bpy.ops.render.render(write_still=True)
        scene.frame_set(frame)

def unpin(cloth):
    if 'Pinned' in cloth.vertex_groups:
        cloth.vertex_groups.remove(cloth.vertex_groups['Pinned'])
    cloth.modifiers["Cloth"].settings.vertex_group_mass = ''

def generate_cloth_state(cloth, subdivisions):
    # Move cloth slightly above the table and simulate a drop
    # Pinned group is the vertices that should not fall
    if cloth is None:
        cloth = make_cloth(subdivisions)
    dx = np.random.uniform(0,0.7,1)*random.choice((-1,1))
    dy = np.random.uniform(0,0.7,1)*random.choice((-1,1))
    dz = np.random.uniform(0.4,0.8,1)
    cloth.location = (dx,dy,dz)
    cloth.rotation_euler = (0, 0, random.uniform(0, np.pi)) # fixed z, rotate only about x/y axis slightly
    pinned_group = bpy.context.object.vertex_groups.new(name='Pinned')
    n = random.choice(range(1,4)) # Number of vertices to pin
    subsample = sample(range(len(cloth.data.vertices)), n)
    pinned_group.add(subsample, 1.0, 'ADD')
    cloth.modifiers["Cloth"].settings.vertex_group_mass = 'Pinned'
    # Episode length = 30 frames
    bpy.context.scene.frame_start = 0 
    bpy.context.scene.frame_end = 180 # Roughly when the cloth settles (mostly still after this until 250 frames)
    return cloth

def reset_cloth(cloth):
    unpin(cloth)
    cloth.location = (0,0,0)
    cloth.rotation_euler = (0,0,0)
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

def render(folder, filename, engine, episode, cloth, annotations=None, num_annotations=0):
    scene = bpy.context.scene
    render_path = set_render_settings(engine, folder, filename)
    for frame in range(0, 30):
        # Render 10 images per episode (episode is really 30 frames)
        if frame%3==0:
            index = ((30 - scene.frame_start)*episode + frame)//3 
            #render_mask("image_masks/%06d_visible_mask.png", index)
            scene.render.filepath = render_path % index
            bpy.ops.render.render(write_still=True)
            if annotations is not None:
                annotations = annotate(cloth, index, annotations, num_annotations)
        scene.frame_set(frame)
    return annotations

def render_mask(filename, index):
    # NOTE: this method is still in progress
    scene = bpy.context.scene
    saved = scene.render.engine
    set_render_settings('BLENDER_EEVEE', folder, filename)
    scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    render_node = tree.nodes["Render Layers"]
    norm_node = tree.nodes.new(type="CompositorNodeNormalize")
    inv_node = tree.nodes.new(type="CompositorNodeInvert")
    math_node = tree.nodes.new(type="CompositorNodeMath")
    math_node.operation = 'CEIL' # Threshold the depth image
    composite = tree.nodes.new(type = "CompositorNodeComposite")
    links.new(render_node.outputs["Depth"], inv_node.inputs["Color"])
    links.new(inv_node.outputs[0], norm_node.inputs[0])
    links.new(norm_node.outputs[0], math_node.inputs[0])
    links.new(math_node.outputs[0], composite.inputs["Image"])
    scene.render.filepath = filename % index
    bpy.ops.render.render(write_still=True)
    # Clean up 
    scene.render.engine = saved
    for node in tree.nodes:
        if node.name != "Render Layers":
            tree.nodes.remove(node)
    scene.use_nodes = False

def update(cloth):
    # Call this method whenever you want the updated coordinates of the cloth after it has been deformed
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cloth_deformed = cloth.evaluated_get(depsgraph)
    return cloth_deformed

def set_render_settings(engine, folder, filename, render_width=640, render_height=480):
    scene = bpy.context.scene
    scene.render.resolution_percentage = 100
    render_scale = scene.render.resolution_percentage / 100
    scene.render.resolution_x = render_width
    scene.render.resolution_y = render_height
    scene.render.engine = engine
    filename = "./{}/{}".format(folder, filename)
    scene.view_settings.exposure = 1.3
    if engine == 'BLENDER_WORKBENCH':
        scene.render.display_mode
        scene.render.image_settings.color_mode = 'RGB'
        scene.display_settings.display_device = 'None'
        scene.sequencer_colorspace_settings.name = 'XYZ'
        scene.render.image_settings.file_format='PNG'
    elif engine == "BLENDER_EEVEE":
        scene.eevee.taa_samples = 1
        scene.eevee.taa_render_samples = 1
    scene.render.resolution_percentage = 100
    render_scale = scene.render.resolution_percentage / 100
    scene.render.resolution_x = render_width
    scene.render.resolution_y = render_height
    return filename

def annotate(cloth, frame, mapping, num_annotations):
    scene = bpy.context.scene
    '''Gets num_annotations annotations of cloth image at provided frame #, adds to mapping'''
    cloth_deformed = update(cloth)
    #vertices = [cloth_deformed.matrix_world @ v.co for v in list(cloth_deformed.data.vertices)[::len(list(cloth_deformed.data.vertices))//num_annotations]] 
    vertices = [cloth_deformed.matrix_world @ v.co for v in list(cloth_deformed.data.vertices)[:num_annotations]] 
    render_size = (scene.render.resolution_x, scene.render.resolution_y)
    pixels = []
    for i in range(len(vertices)):
        v = vertices[i]
        camera_coord = bpy_extras.object_utils.world_to_camera_view(scene, bpy.context.scene.camera, v)
        pixel = [round(camera_coord.x * render_size[0]), round(render_size[1] - camera_coord.y * render_size[1])]
        pixels.append([pixel])
    mapping[frame] = pixels
    return mapping

def render_dataset(num_episodes, folder, filename, num_annotations, subdivisions, texture_filepath='', color=None):
    if not os.path.exists("./images"):
        os.makedirs('./images')
    else:
        os.system('rm -r ./images')
        os.makedirs('./images')
    clear_scene()
    add_camera_light()
    table = make_table()
    cloth = make_cloth(subdivisions)
    if texture_filepath != '':
        engine = 'BLENDER_EEVEE'
        pattern(cloth, texture_filepath)
    elif color:
        engine = 'BLENDER_WORKBENCH'
        colorize(cloth, color)
    annot = {}
    for episode in range(num_episodes):
        reset_cloth(cloth) # Restores cloth to flat state
        cloth = generate_cloth_state(cloth, subdivisions) # Creates a new deformed state
        annot = render(folder, filename, engine, episode, cloth, annotations=annot, num_annotations=num_annotations) # Render, save ground truth
    with open("./images/knots_info.json", 'w') as outfile:
        json.dump(annot, outfile, sort_keys=True, indent=2)
    
if __name__ == '__main__':
    texture_filepath = 'textures/asymm.jpg'
    green = (0,0.5,0.5,1)
    folder = "images"
    filename = "%06d_rgb.png"
    episodes = 2 # Notes each episode has 10 rendered frames 
    subdivisions = 25
    num_annotations = subdivisions**2 # Pixelwise annotations per image
    #render_dataset(episodes, folder, filename, num_annotations, subdivisions, color=green)
    #render_dataset(episodes, filename, num_annotations, texture_filepath=texture_filepath)

    clear_scene()
    add_camera_light()
    table = make_table()
    cloth = make_cloth(subdivisions)
    colorize(cloth, green)
    cloth = generate_cloth_state(cloth, subdivisions) # Creates a new deformed state
    actions = [(0,0,0.5), (0.5,-0.5,0), (0,0,1.0), (-0.5,0.5,0)]
    render_action_sequence(cloth, 0, actions)
