bl_info = {
    "name": "Distool: Displacement & Normal Generator",
    "author": "wsmnb12",
    "version": (1, 0),
    "blender": (4, 4, 0),
    "location": "Shader Editor > Sidebar > Distool",
    "description": "Generate grayscale displacement and tangent-space normal maps from texture detail.",
    "category": "Node",
}

import bpy
import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

def convert_image_to_grayscale(image_path, scene):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    contrast = scene.distool_disp_contrast
    gray = (gray - 127.5) * (1 + contrast) + 127.5
    gray = np.clip(gray, 0, 255)
    
    blur_strength = scene.distool_disp_blur
    if blur_strength != 0:
        sigma = abs(blur_strength)
        if blur_strength > 0:
            gray = gaussian_filter(gray, sigma=sigma)
        else:
            blurred = gaussian_filter(gray, sigma=sigma)
            gray = np.clip(2 * gray - blurred, 0, 255)
    
    if scene.distool_invert_disp:
        gray = 255 - gray
        
    return gray.astype(np.uint8)
def generate_normal_map_from_texture(image_path, scene):
    
    gray = convert_image_to_grayscale(image_path, scene).astype(np.float32) / 255.0


    sigma = max(0.1, abs(scene.distool_normal_blur) or 0.1)
    height = gaussian_filter(gray, sigma=sigma)

    
    height = np.clip(height * 2, 0, 1)  

   
    grad_x = np.gradient(height, axis=1)
    grad_y = np.gradient(height, axis=0)

  
    scale = max(0.1, scene.distool_normal_strength)
    dx = grad_x * scale
    dy = grad_y * scale


    dz = -np.ones_like(dx)


    normal = np.stack((-dx, -dy, dz), axis=-1)


    length = np.linalg.norm(normal, axis=2, keepdims=True) + 1e-8
    normal = normal / length


    normal_rgb = np.zeros_like(normal, dtype=np.float32)
    normal_rgb[..., 0] = (normal[..., 0] + 1.0) * 127.5  
    normal_rgb[..., 1] = (normal[..., 1] + 1.0) * 127.5 
    normal_rgb[..., 2] = (normal[..., 2] + 1.0) * 127.5 + 127.5  

  
    if scene.distool_invert_r:
        normal_rgb[..., 0] = 255 - normal_rgb[..., 0]
    if scene.distool_invert_g:
        normal_rgb[..., 1] = 255 - normal_rgb[..., 1]
    if scene.distool_invert_height:
        normal_rgb[..., 2] = 255 - normal_rgb[..., 2]

    return np.clip(normal_rgb, 0, 255).astype(np.uint8)


def process_image(image_path, scene):
    base_name = os.path.splitext(os.path.basename(image_path))[0]

   
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(addon_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    normal_path = os.path.join(output_dir, base_name + "_normal.png")
    disp_path = os.path.join(output_dir, base_name + "_disp.png")

    if scene.distool_generate_normal:
        normal_img = generate_normal_map_from_texture(image_path, scene)
        cv2.imwrite(normal_path, normal_img)

    if scene.distool_generate_displacement:
        disp_img = convert_image_to_grayscale(image_path, scene)
        cv2.imwrite(disp_path, disp_img)

    return normal_path if scene.distool_generate_normal else "", disp_path if scene.distool_generate_displacement else ""


def apply_maps_to_material(context, normal_path, disp_path, strength):
    mat = context.object.active_material
    if not mat or not mat.use_nodes:
        return

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    if normal_path:
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(normal_path)
        tex.image.colorspace_settings.name = 'Non-Color'
        tex.label = "Distool Normal Map"
        norm = nodes.new("ShaderNodeNormalMap")
        norm.inputs["Strength"].default_value = strength
        links.new(tex.outputs["Color"], norm.inputs["Color"])
        bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf:
            links.new(norm.outputs["Normal"], bsdf.inputs["Normal"])

    if disp_path:
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(disp_path)
        tex.image.colorspace_settings.name = 'Non-Color'
        tex.label = "Distool Displacement Map"
        disp = nodes.new("ShaderNodeDisplacement")
        disp.inputs["Scale"].default_value = 0.1
        links.new(tex.outputs["Color"], disp.inputs["Height"])
        out = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
        if out:
            links.new(disp.outputs["Displacement"], out.inputs["Displacement"])

    context.scene.distool_preview_normal = None
    context.scene.distool_preview_disp = None

class DISTOOL_OT_GenerateSingle(bpy.types.Operator):
    bl_idname = "distool.generate_single"
    bl_label = "Generate Maps from Selected Node"

    def execute(self, context):
        node = context.active_node
        scene = context.scene
        if node and node.type == 'TEX_IMAGE' and node.image:
            img_path = bpy.path.abspath(node.image.filepath_raw)
            normal_path, disp_path = process_image(img_path, scene)
            scene.distool_generated_normal = normal_path or ""
            scene.distool_generated_disp = disp_path or ""

            if normal_path:
                scene.distool_preview_normal = bpy.data.images.load(normal_path)
            if disp_path:
                scene.distool_preview_disp = bpy.data.images.load(disp_path)

            scene.distool_applied = False

            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Select an image texture node with a valid image.")
            return {'CANCELLED'}

class DISTOOL_OT_ApplyMaps(bpy.types.Operator):
    bl_idname = "distool.apply_maps"
    bl_label = "Apply Maps to Material"

    def execute(self, context):
        scene = context.scene
        apply_maps_to_material(context, scene.distool_generated_normal, scene.distool_generated_disp, scene.distool_normal_strength)
        scene.distool_applied = True
        return {'FINISHED'}

class DISTOOL_PT_Panel(bpy.types.Panel):
    bl_label = "Distool"
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Distool"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.prop(scene, "distool_generate_normal")
        layout.prop(scene, "distool_generate_displacement")
        
        if scene.distool_generate_displacement:  
            box = layout.box()
            box.label(text="Displacement Map Settings:")
            box.prop(scene, "distool_disp_contrast")
            box.prop(scene, "distool_disp_blur")
            box.prop(scene, "distool_invert_disp")

        layout.prop(scene, "distool_use_subfolder")

        if scene.distool_generate_normal:
            box = layout.box()
            box.label(text="Normal Map Settings:")
            box.prop(scene, "distool_normal_strength")
            box.prop(scene, "distool_normal_level")
            box.prop(scene, "distool_normal_blur")
            box.prop(scene, "distool_invert_r")
            box.prop(scene, "distool_invert_g")
            box.prop(scene, "distool_invert_height")
            box.prop(scene, "distool_zrange")

        layout.separator()

        node = context.active_node
        if(scene.distool_generate_normal or scene.distool_generate_displacement):
            if node and node.type == 'TEX_IMAGE' and node.image:
                layout.operator("distool.generate_single")
            else:
                layout.label(text="(Select an Image Texture Node)", icon='INFO')
            
            layout.operator("distool.reset_defaults", icon='FILE_REFRESH')

        layout.separator()

        if scene.distool_generate_normal and scene.distool_preview_normal:
            layout.label(text="Normal Map Preview:")
            layout.template_ID_preview(scene, "distool_preview_normal", new="image.new", open="image.open")
        if scene.distool_generate_displacement and scene.distool_preview_disp:
            layout.label(text="Displacement Map Preview:")
            layout.template_ID_preview(scene, "distool_preview_disp", new="image.new", open="image.open")

        if (scene.distool_preview_normal or scene.distool_preview_disp) and not scene.get("distool_applied", False):
            layout.operator("distool.apply_maps", icon='NODE_MATERIAL')
        
        layout.separator()
        

def auto_update_maps(self, context):
    node = context.active_node
    scene = context.scene
    
    if not scene.distool_generated_normal and not scene.distool_generated_disp:
        return

    if node and node.type == 'TEX_IMAGE' and node.image:
        img_path = bpy.path.abspath(node.image.filepath_raw)
        normal_path, disp_path = process_image(img_path, scene)
        scene.distool_generated_normal = normal_path or ""
        scene.distool_generated_disp = disp_path or ""

        if normal_path:
            scene.distool_preview_normal = bpy.data.images.load(normal_path)
        if disp_path:
            scene.distool_preview_disp = bpy.data.images.load(disp_path)

        scene.distool_applied = False

        
class DISTOOL_OT_ResetDefaults(bpy.types.Operator):
    bl_idname = "distool.reset_defaults"
    bl_label = "Reset to Default Settings"

    def execute(self, context):
        scene = context.scene

  
        scene.distool_normal_strength = 2.5
        scene.distool_normal_level = 7.0
        scene.distool_normal_blur = 0
        scene.distool_invert_r = False
        scene.distool_invert_g = False
        scene.distool_invert_height = False
        scene.distool_zrange = True

   
        scene.distool_disp_contrast = -0.5
        scene.distool_disp_blur = 0
        scene.distool_invert_disp = False

        return {'FINISHED'}

def register():
    bpy.utils.register_class(DISTOOL_OT_GenerateSingle)
    bpy.utils.register_class(DISTOOL_OT_ApplyMaps)
    bpy.utils.register_class(DISTOOL_PT_Panel)
    bpy.utils.register_class(DISTOOL_OT_ResetDefaults)
    
    # Distool Settings
    bpy.types.Scene.distool_generate_normal = bpy.props.BoolProperty(name="Generate Normal Map", default=False)
    bpy.types.Scene.distool_generate_displacement = bpy.props.BoolProperty(name="Generate Displacement Map", default=False)
    bpy.types.Scene.distool_use_subfolder = bpy.props.BoolProperty(name="Save in Subfolder", default=True)
    
    # Normal Map Settings
    bpy.types.Scene.distool_normal_strength = bpy.props.FloatProperty(name="Strength", min=0.01, max=5.0, default=2.5, update=auto_update_maps)
    bpy.types.Scene.distool_normal_level = bpy.props.FloatProperty(name="Detail Level", min=4.0, max=10.0, default=7.0, update=auto_update_maps)
    bpy.types.Scene.distool_normal_blur = bpy.props.IntProperty(name="Blur/Sharpen", min=-32, max=32, default=0, update=auto_update_maps)
    bpy.types.Scene.distool_invert_r = bpy.props.BoolProperty(name="Invert R", default=False, update=auto_update_maps)
    bpy.types.Scene.distool_invert_g = bpy.props.BoolProperty(name="Invert G", default=False, update=auto_update_maps)
    bpy.types.Scene.distool_invert_height = bpy.props.BoolProperty(name="Invert Height", default=False, update=auto_update_maps)
    bpy.types.Scene.distool_zrange = bpy.props.BoolProperty(name="Z-Range (-1 to +1)", default=True, update=auto_update_maps)
    
    # Image Previews
    bpy.types.Scene.distool_generated_normal = bpy.props.StringProperty()
    bpy.types.Scene.distool_generated_disp = bpy.props.StringProperty()
    bpy.types.Scene.distool_preview_normal = bpy.props.PointerProperty(type=bpy.types.Image)
    bpy.types.Scene.distool_preview_disp = bpy.props.PointerProperty(type=bpy.types.Image)
    bpy.types.Scene.distool_applied = bpy.props.BoolProperty(default=False)
    
    # Displacement Map Settings
    bpy.types.Scene.distool_disp_contrast = bpy.props.FloatProperty(name="Contrast", min=-1.0, max=1.0, default=-0.5, update=auto_update_maps)
    bpy.types.Scene.distool_disp_blur = bpy.props.IntProperty(name="Blur/Sharpen", min=-32, max=32, default=0, update=auto_update_maps)
    bpy.types.Scene.distool_invert_disp = bpy.props.BoolProperty(name="Invert", default=False, update=auto_update_maps)
    
    
def unregister():
    bpy.utils.unregister_class(DISTOOL_OT_GenerateSingle)
    bpy.utils.unregister_class(DISTOOL_OT_ApplyMaps)
    bpy.utils.unregister_class(DISTOOL_PT_Panel)
    bpy.utils.unregister_class(DISTOOL_OT_ResetDefaults)

    del bpy.types.Scene.distool_generate_normal
    del bpy.types.Scene.distool_generate_displacement
    del bpy.types.Scene.distool_use_subfolder
    del bpy.types.Scene.distool_normal_strength
    del bpy.types.Scene.distool_normal_level
    del bpy.types.Scene.distool_normal_blur
    del bpy.types.Scene.distool_invert_r
    del bpy.types.Scene.distool_invert_g
    del bpy.types.Scene.distool_invert_height
    del bpy.types.Scene.distool_zrange
    del bpy.types.Scene.distool_generated_normal
    del bpy.types.Scene.distool_generated_disp
    del bpy.types.Scene.distool_preview_normal
    del bpy.types.Scene.distool_preview_disp
    del bpy.types.Scene.distool_applied
    del bpy.types.Scene.distool_disp_contrast
    del bpy.types.Scene.distool_disp_blur
    del bpy.types.Scene.distool_invert_disp

if __name__ == "__main__":
    register()