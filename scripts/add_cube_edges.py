"""
Add black wireframe edges to Cube_Yellow, Cube_Green, Cube_Red in sorting.usd.

This creates BasisCurves geometry as children of each cube prim,
with a black unlit material, so edges are always visible and move with the cube.
"""

import shutil
from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf, Vt

USD_PATH = "public/assets/usd/sorting.usd"
BACKUP_PATH = "public/assets/usd/sorting.usd.bak"

CUBE_NAMES = ["Cube_Yellow", "Cube_Green", "Cube_Red"]

# Slightly outside 0.5 to avoid z-fighting
E = 0.502

# 8 vertices of the cube, slightly outside the face
VERTS = [
    (-E, -E, -E), ( E, -E, -E), ( E,  E, -E), (-E,  E, -E),  # bottom
    (-E, -E,  E), ( E, -E,  E), ( E,  E,  E), (-E,  E,  E),  # top
]

# 12 edges as pairs of vertex indices
EDGES = [
    (0,1), (1,2), (2,3), (3,0),  # bottom
    (4,5), (5,6), (6,7), (7,4),  # top
    (0,4), (1,5), (2,6), (3,7),  # verticals
]


def create_black_material(stage, mat_path):
    """Create a simple black unlit material."""
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    # Make it emissive black (fully dark, unlit appearance)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def add_edges_to_cube(stage, cube_path, material):
    """Add BasisCurves wireframe edges as a child of the cube prim."""
    edges_path = f"{cube_path}/edges"

    # Remove existing edges prim if present (for idempotency)
    existing = stage.GetPrimAtPath(edges_path)
    if existing:
        stage.RemovePrim(edges_path)

    curves = UsdGeom.BasisCurves.Define(stage, edges_path)

    # Build points and vertex counts for all 12 edges
    points = []
    vertex_counts = []
    for (i, j) in EDGES:
        points.append(Gf.Vec3f(*VERTS[i]))
        points.append(Gf.Vec3f(*VERTS[j]))
        vertex_counts.append(2)

    curves.GetPointsAttr().Set(Vt.Vec3fArray(points))
    curves.GetCurveVertexCountsAttr().Set(Vt.IntArray(vertex_counts))
    curves.GetTypeAttr().Set("linear")
    curves.GetWidthsAttr().Set(Vt.FloatArray([0.04] * len(points)))  # line width in local space
    curves.SetWidthsInterpolation(UsdGeom.Tokens.vertex)

    # Bind black material
    UsdShade.MaterialBindingAPI.Apply(curves.GetPrim())
    UsdShade.MaterialBindingAPI(curves.GetPrim()).Bind(material)

    # Don't add physics to the edges - they're just visual
    curves.GetPrim().SetMetadata("kind", "component")

    print(f"  Added edges to {cube_path}")


def main():
    # Backup
    shutil.copy2(USD_PATH, BACKUP_PATH)
    print(f"Backed up to {BACKUP_PATH}")

    stage = Usd.Stage.Open(USD_PATH)

    # Create black material
    mat_path = "/World/Looks/Black_Edge"
    mat = create_black_material(stage, mat_path)
    print(f"Created material at {mat_path}")

    # Add edges to each cube
    for name in CUBE_NAMES:
        cube_path = f"/World/{name}"
        prim = stage.GetPrimAtPath(cube_path)
        if not prim:
            print(f"  WARNING: {cube_path} not found, skipping")
            continue
        add_edges_to_cube(stage, cube_path, mat)

    stage.GetRootLayer().Save()
    print(f"\nSaved {USD_PATH}")


if __name__ == "__main__":
    main()
