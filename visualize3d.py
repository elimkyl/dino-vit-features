import open3d as o3d
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_text_3d(text, pos, font_size=32, depth=0.001, font='/System/Library/Fonts/Supplemental/Arial.ttf'):
    """Create a simple 3D text label as a thin point cloud."""
    font_obj = ImageFont.truetype(font, font_size)

    # Compatible with all Pillow versions
    try:
        w, h = font_obj.getsize(text)
    except AttributeError:
        bbox = font_obj.getbbox(text)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Create image and draw text
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, fill=255, font=font_obj)
    img = np.asarray(img)

    # Convert white pixels to 3D points
    ys, xs = np.nonzero(img)
    pts = np.stack([xs, -ys, np.zeros_like(xs)], axis=1).astype(np.float32)
    pts -= pts.mean(0)  # center text
    pts *= 0.0015        # scale down to reasonable size
    pts += np.array(pos)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color([0, 0, 0])
    return pcd


def print_3d_frame(R=None, t= None, save_path=None):
    """
    Visualizes a 3D coordinate frame with Open3D, including world and camera frames.
    """

    if R is None:
        R = np.array([[ 0.79194903,  0.61058724,  0.        ],
                    [ 0.61058724, -0.79194903,  0.        ],
                    [ 0.        ,  0.        ,  1.        ]])
    
    if t is None:
        t = np.array([0.06536154, -0.21937567, 0.0])

    # ---- Frames ----
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
    camera_frame.rotate(R, center=(0,0,0))
    camera_frame.translate(t)

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])

    # ---- Labels ----
    world_label = create_text_3d("World Frame", pos=[0, 0.05, 0])
    camera_label = create_text_3d("Camera Frame", pos=t + np.array([0, 0.05, 0]))

    # ---- Visualize ----
   # ---- Visualize ----
    if save_path is None:
        # Just show
        o3d.visualization.draw_geometries([world_frame, camera_frame, world_label, camera_label])
    else:
        # Use Visualizer to save image
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)
        for g in [world_frame, camera_frame, world_label, camera_label]:
            vis.add_geometry(g)

        vis.poll_events()
        vis.update_renderer()

        vis.capture_screen_image(save_path, do_render=True)
        vis.destroy_window()
        print(f"Saved 3D frame as {save_path}")

