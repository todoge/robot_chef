import pybullet as p
import pybullet_data
import os

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

stl_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bowl.stl")
import trimesh

# Load STL and save as OBJ
mesh = trimesh.load(stl_file_path)
mesh.export('bowl.obj')

obj_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bowl.obj")

# Decompose your bowl STL file
name_in = obj_file_path
name_out = "bowl_vhcad.obj"
name_log = "log.txt"
p.vhacd(name_in, name_out, name_log, alpha=0.04, resolution=50000)
