import bpy
import os
import numpy as np
import mathutils
import sys

try:
    import xmltodict  # type: ignore
except ImportError:
    xmltodict = None
def parse_render_image_list(path):
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            
            data_line=line.split(' ')
            w, h,fx,fy,cx,cy = list(map(float,data_line[2:8]))[:]  

            K = [w, h,fx,fy,cx,cy]
            break
  
    return K

def parse_image_list(path):
    images = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            name, *data = line.split()
            model, width, height, *params = data
            params = np.array(params, float)
            images[os.path.basename(name).split('.')[0]] = (model, int(width), int(height), params)
  
    assert len(images) > 0
    return images

def parse_pose_list(path, origin_coord):
    poses = {}
    with open(path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = (data[0].split('/')[-1]).split('.')[0]
            q, t = np.split(np.array(data[1:], float), [4])
            
            # Interpret q,t as a standard COLMAP-style w2c.
            # We convert to c2w without applying any extra axis flips,
            # assuming the OBJ and poses already share the same ECEF / world frame.
            R = np.asmatrix(qvec2rotmat(q)).transpose()  # c2w (world)

            T = np.identity(4)
            T[0:3, 0:3] = R
            T[0:3, 3] = -R.dot(t)
            if origin_coord is not None:
                origin_coord = np.array(origin_coord)
                T[0:3, 3] -= origin_coord

            poses[name] = T
            
    
    assert len(poses) > 0
    return poses

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def _make_materials_emission():
    for mat in bpy.data.materials:
        # Try to read an existing base color
        base_col = None
        if mat.use_nodes and mat.node_tree:
            nodes = mat.node_tree.nodes
            bsdf = None
            for n in nodes:
                if n.type == "BSDF_PRINCIPLED":
                    bsdf = n
                    break
            if bsdf is not None:
                base_col = list(bsdf.inputs["Base Color"].default_value)

        if base_col is None:
            # Fallback to diffuse_color if no principled BSDF exists
            base_col = list(mat.diffuse_color) if mat.diffuse_color else [1.0, 1.0, 1.0, 1.0]

        # Build a minimal emission-only material
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        out_node = nodes.new("ShaderNodeOutputMaterial")
        emis = nodes.new("ShaderNodeEmission")
        emis.inputs["Color"].default_value = base_col
        emis.inputs["Strength"].default_value = 1.0
        links.new(emis.outputs["Emission"], out_node.inputs["Surface"])

def prepare_scene(resolution=[760, 760], device='GPU'):
    bpy.data.scenes["Scene"].render.resolution_x = resolution[0]
    bpy.data.scenes["Scene"].render.resolution_y = resolution[1]
#    bpy.data.scenes["Scene"].cycles.device = device
    try:
        delete_object('Cube')
        delete_object('Light')
    except:
        pass

def add_camera(xyz=(0, 0, 0),
               rot_vec_degree=(0, 0, 0),
               name=None,
               proj_model='PERSP',
               f=35,
               sensor_fit='HORIZONTAL',
               sensor_width=32,
               sensor_height=18,
               clip_start=0.1,
               clip_end=10000):
    bpy.ops.object.camera_add(location=xyz, rotation=rot_vec_degree)
    cam = bpy.context.active_object

    if name is not None:
        cam.name = name
    cam.data.type = proj_model
    cam.data.lens = f
    cam.data.sensor_fit = sensor_fit
    cam.data.sensor_width = sensor_width
    cam.data.sensor_height = sensor_height
    cam.data.clip_start = clip_start
    cam.data.clip_end = clip_end
    return cam
        
def prepare_world(image_save_path, name):
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    bpy.context.scene.render.image_settings.color_depth = '16'
    bpy.context.scene.render.image_settings.color_mode = 'RGB'


    bpy.context.scene.render.image_settings.file_format = "OPEN_EXR"

    # Clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    
    scene = bpy.context.scene
    cam = scene.objects['Camera']
    for output_node in [depth_file_output]: #image_file_output
        output_node.base_path = ''


    scene.render.filepath = image_save_path
    depth_file_output.file_slots[0].path = scene.render.filepath + name
    bpy.ops.render.render(write_still=True)



def delete_object(label):
    """
    Definition to delete an object from the scene.
    Parameters
    ----------
    label          : str
                     String that identifies the object to be deleted.
    """
    bpy.data.objects.remove(bpy.context.scene.objects[label], do_unlink=True)


# Command-line arguments
image_save_path = str(sys.argv[-1])
input_pose = str(sys.argv[-2])
input_intrin = str(sys.argv[-3])

# Get sensor parameters
f_mm = float(sys.argv[-4])
sensor_width = float(sys.argv[-5])
sensor_height = float(sys.argv[-6])

# Get origin coordinates from XML file
xml_path = str(sys.argv[-7])
if xmltodict is not None:
    # Preferred path: use xmltodict if available (as in original Render2Loc)
    with open(xml_path, encoding="utf-8") as file_object:
        all_the_xml_str = file_object.read()
        dictdata = dict(xmltodict.parse(all_the_xml_str))
        origin = dictdata["ModelMetadata"]["SRSOrigin"]
else:
    # Fallback: use standard library XML parser if xmltodict is not installed
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Try to find <SRSOrigin> under <ModelMetadata> or at top level
    origin_node = None
    if root.tag == "ModelMetadata":
        origin_node = root.find("SRSOrigin")
    if origin_node is None:
        origin_node = root.find(".//SRSOrigin")
    if origin_node is None or origin_node.text is None:
        raise RuntimeError(f"Cannot find <SRSOrigin> in {xml_path}")
    origin = origin_node.text.strip()

# Split the origin string and convert to floats
x, y, z = origin.split(",")[0], origin.split(",")[1], origin.split(",")[2]
origin_coord = [float(x), float(y), float(z)]

# Parse poses and intrinsics
poses = parse_pose_list(input_pose, origin_coord)
intrinsics = parse_render_image_list(input_intrin) 

# Iterate over the poses and render images
for name in list(poses.keys()):
    # Get the pose frame matrix
    pose_frame = poses[name]
    R = pose_frame[:3, :3]
    t = list(pose_frame[:3, 3])
    
    # Convert rotation matrix to Euler angles
    rot_mat = mathutils.Matrix(list(R))
    rot_eulr = rot_mat.to_euler()

    # Get intrinsic parameters
    w = int(intrinsics[0])
    h = int(intrinsics[1])
    f_x = np.float32(intrinsics[2])
    f_y = np.float32(intrinsics[3])
    
    print("euler:", rot_eulr, t)
    # Add camera based on focal length format or pixel format
    if f_mm > 0:
        # Focal length in millimeters
        camera_obj = add_camera(
            xyz=t,
            rot_vec_degree=rot_eulr,
            f=f_mm,
            sensor_fit='HORIZONTAL',
            sensor_width=sensor_width,
            sensor_height=sensor_height,
            clip_end=10000
        )
    else:
        # Focal length in pixels
        f_mm = 35  # Default focal length in millimeters
        intrin_factor_fx = f_x / f_mm
        intrin_factor_fy = f_y / f_mm
        camera_obj = add_camera(
            xyz=t,
            rot_vec_degree=rot_eulr,
            f=f_mm,
            sensor_fit='HORIZONTAL',
            sensor_width=w / intrin_factor_fx,
            sensor_height=h / intrin_factor_fy,
            clip_end=10000
        )
    # Set the camera and render settings
    scene = bpy.context.scene
    scene.camera = camera_obj

    # Use Workbench for fast, simple rendering with original colors
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.filepath = f"{image_save_path}/{name}"

    # Disable all shadows
    for light in bpy.data.lights:
        light.use_shadow = False

    # Set background to pure black
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    world = scene.world
    world.use_nodes = False
    world.color = (0.0, 0.0, 0.0)

    # Render the image
    bpy.ops.render.render(write_still=True)
    