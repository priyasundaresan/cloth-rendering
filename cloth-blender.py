#from sklearn.neighbors import NearestNeighbors
import bpy
import cv2
import time
import json
import bpy, bpy_extras
from math import *
from mathutils import *
import random
import numpy as np
from random import sample
import bmesh


#Adi: Descriptor related imports
import os
import sys

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

    #Adi: Increase the friction of the plane
    #bpy.context.object.collision.friction_factor = 4
    bpy.context.object.collision.cloth_friction = 20
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

    #Adi: Add Vertex Weight Edit modifier to change pinned vertices over time
    bpy.ops.object.modifier_add(type='VERTEX_WEIGHT_EDIT')
    bpy.ops.object.modifier_move_up(modifier="VertexWeightEdit")
    bpy.ops.object.modifier_move_up(modifier="VertexWeightEdit")
    bpy.ops.object.modifier_move_up(modifier="VertexWeightEdit")
    bpy.context.object.modifiers["VertexWeightEdit"].remove_threshold = 1
    bpy.context.object.modifiers["VertexWeightEdit"].use_remove = False

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
    cloth.rotation_euler = (0, 0, random.uniform(0, np.pi)) # fixed z, rotate only about x/y axis slightly
    if 'Pinned' in cloth.vertex_groups:
        cloth.vertex_groups.remove(cloth.vertex_groups['Pinned'])

    #Initial Pinning
    #pinned_group = bpy.context.object.vertex_groups.new(name='Pinned')
    #n = random.choice(range(1,4)) # Number of vertices to pin
    #subsample = sample(range(len(cloth.data.vertices)), n)
    #pinned_group.add(subsample, 0.99, 'ADD') #Adi: Adding with 0.99 weight so that we can remove pinned vertices after settling
    #cloth.modifiers["Cloth"].settings.vertex_group_mass = 'Pinned'
    #Adi: Can only assign to "Pinned" after "Pinned" is created
    #cloth.modifiers["VertexWeightEdit"].vertex_group = "Pinned"



    # Episode length = 30 frames
    bpy.context.scene.frame_start = 0 
    bpy.context.scene.frame_end = 90 # Roughly when the cloth settles
    return cloth

def action(cloth, v_index=0, frame_num=0):
    #v_index is the index of the vertex you want to grab

    #Grab Pinning
    grab_pinned_group = bpy.context.object.vertex_groups.new(name='Grab')
    n = 1 # Number of vertices to pin
    seq = []
    seq.append(v_index)
    grab_pinned_group.add(seq, 0.99, 'ADD')
    cloth.modifiers["VertexWeightEdit"].vertex_group = "Grab"

    bpy.ops.object.mode_set(mode = 'OBJECT')
    obj = bpy.context.active_object
    bpy.ops.object.mode_set(mode = 'EDIT') 
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action = 'DESELECT')
    bpy.ops.object.mode_set(mode = 'OBJECT')
    obj.data.vertices[v_index].select = True
    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.object.hook_add_newob()

    bpy.ops.object.mode_set(mode = 'OBJECT')

    hook = bpy.data.objects['Empty']


    cloth.modifiers["Cloth"].settings.vertex_group_mass = 'Grab'
    bpy.ops.object.modifier_move_up(modifier="Hook-Empty")
    bpy.ops.object.modifier_move_up(modifier="Hook-Empty")

    bpy.context.scene.frame_set(frame_num)
    hook.keyframe_insert(data_path='location')

    #bpy.context.scene.frame_set(frame_num+10)
    cloth.keyframe_insert(data_path='modifiers["VertexWeightEdit"].use_remove')
    bpy.context.scene.frame_set(60)
    bpy.ops.transform.translate(value=(1, 1, 0), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
    
    cloth.modifiers["VertexWeightEdit"].use_remove = True
    cloth.keyframe_insert(data_path='modifiers["VertexWeightEdit"].use_remove')
    hook.keyframe_insert(data_path='location')

def fold_action(cloth, v_index_grab=0, v_index_release=624, frame_num=0):
    #v_index is the index of the vertex you want to grab

    #Grab Pinning
    grab_pinned_group = bpy.context.object.vertex_groups.new(name='Grab')
    n = 1 # Number of vertices to pin
    seq = []
    seq.append(v_index_grab)
    grab_pinned_group.add(seq, 0.99, 'ADD')
    cloth.modifiers["VertexWeightEdit"].vertex_group = "Grab"

    bpy.ops.object.mode_set(mode = 'OBJECT')
    obj = bpy.context.active_object
    bpy.ops.object.mode_set(mode = 'EDIT') 
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action = 'DESELECT')
    bpy.ops.object.mode_set(mode = 'OBJECT')
    obj.data.vertices[v_index_grab].select = True
    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.object.hook_add_newob()

    bpy.ops.object.mode_set(mode = 'OBJECT')

    hook = bpy.data.objects['Empty']


    cloth.modifiers["Cloth"].settings.vertex_group_mass = 'Grab'
    bpy.ops.object.modifier_move_up(modifier="Hook-Empty")
    bpy.ops.object.modifier_move_up(modifier="Hook-Empty")

    bpy.context.scene.frame_set(frame_num)
    hook.keyframe_insert(data_path='location')

    #bpy.context.scene.frame_set(frame_num+10)
    cloth.keyframe_insert(data_path='modifiers["VertexWeightEdit"].use_remove')
    bpy.context.scene.frame_set(60)

    vertices = [cloth.matrix_world @ v.co for v in list(cloth.data.vertices)] 
    v_grab = vertices[v_index_grab]
    v_release = vertices[v_index_release]
    dx = v_release[0] - v_grab[0]
    dy = v_release[1] - v_grab[1]
    dz = v_release[2] - v_grab[2]
    
    
    bpy.ops.transform.translate(value=(dx, dy, dz), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
    
    cloth.modifiers["VertexWeightEdit"].use_remove = True
    cloth.keyframe_insert(data_path='modifiers["VertexWeightEdit"].use_remove')
    hook.keyframe_insert(data_path='location')
    

    

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

def render_old(filename, engine, episode, cloth, annotations=None, num_annotations=0):
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.filepath = "./images/{}".format(filename)
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
    for frame in range(0, scene.frame_end):
        # Render 10 images per episode (episode is really 30 frames)
        if frame%3==0:
            index = ((scene.frame_end - scene.frame_start)*episode + frame)//3 
            render_mask("image_masks/%06d_visible_mask.png", index)
            scene.render.filepath = filename % index
            bpy.ops.render.render(write_still=True)
            if annotations is not None:
                annotations = annotate(cloth, index, annotations, num_annotations)
        # TODO: this is kind of a hack for now, must increment frame by one or cloth looks weird
        # Baking the simulation seems too time-consuming..., so for now just stepping through the frames
        scene.frame_set(frame)
    return annotations

def render(frame_num, filename, engine, episode, cloth, annotations=None, num_annotations=0):
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.filepath = "./images/{}".format(filename)
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

    for frame in range(0, scene.frame_end):
        # Render 10 images per episode (episode is really 30 frames)
        if frame==frame_num or frame==3:
            index = ((scene.frame_end - scene.frame_start)*episode + frame)//3 
            render_mask("image_masks/%06d_visible_mask.png", index)
            scene.render.filepath = filename % index
            bpy.ops.render.render(write_still=True)
            if annotations is not None:
                annotations = annotate(cloth, index, annotations, num_annotations)
        # TODO: this is kind of a hack for now, must increment frame by one or cloth looks weird
        # Baking the simulation seems too time-consuming..., so for now just stepping through the frames
        scene.frame_set(frame)
    return annotations


def render_mask(filename, index):
    # NOTE: this method is still in progress
    scene = bpy.context.scene
    saved = scene.render.engine
    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.taa_samples = 1
    scene.eevee.taa_render_samples = 1
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

def annotate(cloth, frame, mapping, num_annotations, render_width=640, render_height=480):
    '''Gets num_annotations annotations of cloth image at provided frame #, adds to mapping'''
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cloth_deformed = cloth.evaluated_get(depsgraph)
    vertices = [cloth_deformed.matrix_world @ v.co for v in list(cloth_deformed.data.vertices)[::len(list(cloth_deformed.data.vertices))//num_annotations]] 
    scene.render.resolution_percentage = 100
    render_scale = scene.render.resolution_percentage / 100
    scene.render.resolution_x = render_width
    scene.render.resolution_y = render_height
    render_size = (
            int(scene.render.resolution_x * render_scale),
            int(scene.render.resolution_y * render_scale),
            )
    pixels = []
    for i in range(len(vertices)):
        v = vertices[i]
        v_lyst = [v[0], v[1], v[2]]
        camera_coord = bpy_extras.object_utils.world_to_camera_view(scene, bpy.context.scene.camera, v)
        #pixel = [round(camera_coord.x * render_size[0]), round(render_size[1] - camera_coord.y * render_size[1])]
        #Adi: Pixel need to be converted to int for key
        pixel = [round(camera_coord.x * render_size[0]), round(render_size[1] - camera_coord.y * render_size[1])]
        #Adi: Going from pixel to vertex now
        flattened = pixel[0] * 480 + pixel[1] #Should this be 480 or 640?
        mapping[flattened] = v_lyst
        pixels.append([pixel])
    #mapping[frame] = pixels
    return mapping

#def knn(self, points, error_margin, k, inputs, model=None):
#    if model is None:
#        model = NearestNeighbors(k, error_margin)
#        model.fit(points)
#    match_indices = model.kneighbors(inputs, k, return_distance=False).squeeze()
#    k_matches = points[match_indices]
#    return model, k_matches

def render_dataset_old(num_episodes, filename, num_annotations, texture_filepath='', color=None):
    # Remove anything in scene 
    clear_scene()
    # Make the camera, lights, table, and cloth only ONCE
    add_camera_light()
    table = make_table()
    cloth = make_cloth()
    if texture_filepath != '':
        engine = 'BLENDER_EEVEE'
        pattern(cloth, texture_filepath)
    elif color:
        engine = 'BLENDER_WORKBENCH'
        colorize(cloth, color)
    annot = {}
    for episode in range(num_episodes):
        reset_cloth(cloth) # Restores cloth to flat state
        cloth = generate_cloth_state(cloth) # Creates a new deformed state
        annot = render(29, filename, engine, episode, cloth, annotations=annot, num_annotations=num_annotations) # Render, save ground truth
    with open("./images/knots_info.json", 'w') as outfile:
        json.dump(annot, outfile, sort_keys=True, indent=2)

def render_dataset(num_episodes, filename, num_annotations, texture_filepath='', color=None):
    # Remove anything in scene 
    clear_scene()
    # Make the camera, lights, table, and cloth only ONCE
    add_camera_light()
    table = make_table()
    cloth = make_cloth()
    if texture_filepath != '':
        engine = 'BLENDER_EEVEE'
        pattern(cloth, texture_filepath)
    elif color:
        engine = 'BLENDER_WORKBENCH'
        colorize(cloth, color)
    annot = {}
    ep_len = 1
    for episode in range(num_episodes):
        reset_cloth(cloth) # Restores cloth to flat state
        cloth = generate_cloth_state(cloth) # Creates a new deformed state
        for i in range(ep_len): #ep_len should be 10 actions, but right now it can only be one
            #Adi: Take an action on the cloth
            #r_index = sample(range(len(cloth.data.vertices)), 1)
            action(cloth, v_index=0, frame_num=30*(i)) #Should be i+1 if we drop first
            annot = render(30*(i+3)-1, filename, engine, episode, cloth, annotations=annot, num_annotations=num_annotations) # Render, save ground truth, should be i+2 if we drop first
    with open("./images/knots_info.json", 'w') as outfile:
        json.dump(annot, outfile, sort_keys=True, indent=2)
    
def test(num_episodes=1):
    # Remove anything in scene 
    clear_scene()
    # Make the camera, lights, table, and cloth only ONCE
    add_camera_light()
    table = make_table()
    cloth = make_cloth()
    for episode in range(num_episodes):
        reset_cloth(cloth) # Restores cloth to flat state
        cloth = generate_cloth_state(cloth) # Creates a new deformed state
        #Adi: Take an action on the cloth
        #index = sample(range(len(cloth.data.vertices)), 1)
        #action(cloth, v_index=0)
        fold_action(cloth)

    
if __name__ == '__main__':
    #Adi: So that we can import python files in the same directory
    dir = os.path.dirname(bpy.data.filepath)
    if not dir in sys.path:
        sys.path.append(dir )
        print(sys.path)
    from descriptors import main
    from descriptors.dense_correspondence_network import DenseCorrespondenceNetwork
    

    #texture_filepath = 'textures/cloth.jpg'
    #texture_filepath = 'textures/qr.png'
    goal_img_path = 'cloth_images/flat_goal_rgb.png'
    green = (0,0.5,0.5,1)
    filename = "images/%06d_rgb.png"
    episodes = 1 # Note each episode has 10 rendered frames 
    num_annotations = 300 # Pixelwise annotations per image

    goal_img = cv2.imread(goal_img_path)
    #render_dataset_old(episodes, filename, num_annotations, color=green)
    #render_dataset_old(episodes, filename, num_annotations, texture_filepath=texture_filepath)
    test()
    #render_dataset(episodes, filename, num_annotations, color=green)

    #Adi: This is location of the models on nfs
    #base_dir = '/nfs/diskstation/adi/models/dense_descriptor_models'
    #Adi: This is location of the models on MacOS local
    base_dir = '/Users/adivganapathi/Documents/UC Berkeley/Current Projects/dense_descriptor_models'
    network_dir = 'tier1_oracle_1811_consecutive_3'
    dcn = DenseCorrespondenceNetwork.from_model_folder(os.path.join(base_dir, network_dir), model_param_file=os.path.join(base_dir, network_dir, '003501.pth'))
    dcn.eval()
    image_dir = "./cloth_images"
    #with open(image_dir + '/knots_info.json', 'r') as f:
    #    knots_info = json.load(f)
    #print(knots_info['0'])


    #with open('../cfg/dataset_info.json', 'r') as f:
    with open('./cfg/dataset_info.json', 'r') as f:
        dataset_stats = json.load(f)
    dataset_mean, dataset_std_dev = dataset_stats["mean"], dataset_stats["std_dev"]

    descriptors = Descriptors(dcn, dataset_mean, dataset_std_dev, image_dir)
    print("starting correspondence finder")
    #descriptors.run()

    print("finished correspondence finder")
    cv2.destroyAllWindows()
