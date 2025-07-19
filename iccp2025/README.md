


(1) First point the SPAD at a planar wall (ensure that the entire FOV is looking at a planar surface). 

(2) Plug SPAD into computer. 

(3) Create conda environment

(4) pip install -e .

(5) cd cc-demos/iccp2025

(6) python calibrate_point_cloud.py. Ensure that the start_bin is set accordingly to ensure that the 1-bounce returns fall within the timing gate of the SPAD. Visualize point cloud to ensure reasonable results (script does this automatically). 

(7) python nlos_bp.py. 