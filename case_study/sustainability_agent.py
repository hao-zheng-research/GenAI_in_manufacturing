import numpy as np
import trimesh
from skimage import measure
from scipy import ndimage
import os
import time

def make_printable(mesh):
    """
    The 'Mesh Doctor'. Fixes non-manifold edges for Bambu/Prusa slicers.
    """
    print("[6.5/7] Repairing Mesh for 3D Printing...")
    mesh.process(validate=True)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fill_holes(mesh)
    
    try:
        if hasattr(mesh, 'non_degenerate_faces'):
            mesh.update_faces(mesh.non_degenerate_faces)
    except Exception:
        pass
        
    return mesh

def optimize_heavy_duty(input_stl, output_stl, wall_thickness_mm=3.0, infill_density=0.0):
    """
    Heavy Duty Optimization.
    - infill_density=0.0: Keeps 50% of material (Thick Struts).
    - pitch=0.2: High resolution for accurate walls.
    """
    start_time = time.time()
    print(f"[1/7] Loading {input_stl}...")
    
    if not os.path.exists(input_stl):
        print(f"Error: {input_stl} not found.")
        return

    mesh = trimesh.load(input_stl)
    
    # --- STEP 1: ULTRA-FINE VOXELIZATION ---
    # Pitch 0.2mm is required for robust walls.
    # 0.3mm (your previous code) was too coarse for the lattice details.
    pitch = 0.2
    print(f"[2/7] Voxelizing at High Res (Pitch: {pitch}mm)...")
    
    voxel_grid = mesh.voxelized(pitch=pitch).fill()
    voxels = voxel_grid.matrix
    
    # --- STEP 2: SOLID SKIN PROTECTION ---
    print(f"[3/7] Calculating {wall_thickness_mm}mm Solid Skin...")
    
    distance_map = ndimage.distance_transform_edt(voxels)
    skin_voxels = wall_thickness_mm / pitch
    
    core_mask = distance_map > skin_voxels
    skin_mask = voxels & ~core_mask

    # --- STEP 3: GYROID GENERATION ---
    print("[4/7] Generating Heavy-Duty Lattice...")
    shape = voxels.shape
    
    # Scale: 0.06 keeps the pores reasonable (~5-8mm)
    scale = 0.06 * pitch 
    
    x = np.linspace(0, shape[0] * scale, shape[0])
    y = np.linspace(0, shape[1] * scale, shape[1])
    z = np.linspace(0, shape[2] * scale, shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    gyroid_field = np.sin(X)*np.cos(Y) + np.sin(Y)*np.cos(Z) + np.sin(Z)*np.cos(X)
    
    # Gaussian Blur: Rounds sharp edges to prevent printing errors
    gyroid_field = ndimage.gaussian_filter(gyroid_field, sigma=1.0)
    
    # --- THE FIX: THRESHOLD 0.0 ---
    # 0.4 = Thin Wires (Weak)
    # 0.0 = Thick Channels (Strong) -> This fixes the "Missing Walls"
    lattice_mask = gyroid_field > infill_density
    
    # --- STEP 4: FUSION ---
    print("[5/7] Fusing Geometry...")
    final_voxels = skin_mask | (core_mask & lattice_mask)
    
    # --- STEP 5: RECONSTRUCTION ---
    print("[6/7] Reconstructing Mesh...")
    verts, faces, normals, values = measure.marching_cubes(
        final_voxels, 
        level=0.5, 
        spacing=(pitch, pitch, pitch)
    )
    
    eco_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    
    # --- STEP 6: REPAIR ---
    # Apply the repair function to fix non-manifold edges
    eco_mesh = make_printable(eco_mesh)

    # --- STEP 7: SMOOTHING & EXPORT ---
    print(f"[7/7] Saving to {output_stl}...")
    trimesh.smoothing.filter_laplacian(eco_mesh, iterations=2)
    eco_mesh.export(output_stl)
    
    duration = time.time() - start_time
    vol_reduction = 100 * (1 - (eco_mesh.volume / mesh.volume))
    print("-" * 30)
    print(f"DONE in {duration:.1f} seconds.")
    print(f"Material Saved: {vol_reduction:.1f}%")
    print("-" * 30)

if __name__ == "__main__":
    # Settings
    INPUT_FILE = "phone_stand_charging.stl" 
    OUTPUT_FILE = "phone_stand_HEAVY_DUTY_v3.stl"
    
    # RUN
    # wall_thickness_mm=3.0 -> Gives a thick outer shell
    # infill_density=0.0    -> Keeps 50% internal volume (Crucial for strength)
    try:
        optimize_heavy_duty(INPUT_FILE, OUTPUT_FILE, wall_thickness_mm=3.0, infill_density=0.0)
    except Exception as e:
        print(f"Failed: {e}")