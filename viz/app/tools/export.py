"""Dump the ScoreSaber watch object meshes as raw triangulated JSON.

Run with headless Blender (the FBX/OBJ importers do the parsing):
    ~/Blender/blender-5.0.1-linux-x64/blender -b --python export.py
then `node compose.mjs` to bake the Unity transforms into
../src/lib/models.data.js. Raw mesh coordinates are used (NOT the object
transform Blender puts the FBX unit scale on), times Unity's per-file
globalScale — exactly what Unity's importer does with useFileScale 0.
"""
import bpy, bmesh, json, os
A = "/home/evolheartrise/git projects/scoresaber-watch/Assets/Models"
# (file, object name, unity globalScale, output key)
jobs = [("Objects/Cube.fbx","Cube",1.0,"cube"),("Objects/Note.fbx","Arrow",0.25,"arrow"),("Objects/Note.fbx","Dot",0.25,"dot"),
        ("Objects/Bomb.obj","Bomb",1.0,"bomb"),("Objects/ChainHead.fbx","ChainHead",1.0,"chainHead"),("Objects/ChainLink.fbx","ChainLink",1.0,"chainLink"),
        ("Player/Saber.fbx","Blade",0.5,"blade"),("Player/Saber.fbx","Handle",0.5,"handle"),("Player/Headset.fbx","Headset",1.0,"headset")]
out = {}
loaded = None
for rel, name, gs, key in jobs:
    if loaded != rel:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        p = os.path.join(A, rel)
        if rel.endswith(".obj"): bpy.ops.wm.obj_import(filepath=p)
        else: bpy.ops.import_scene.fbx(filepath=p)
        loaded = rel
    o = bpy.data.objects[name]
    me = o.data.copy()
    bm = bmesh.new(); bm.from_mesh(me); bmesh.ops.triangulate(bm, faces=bm.faces[:]); bm.to_mesh(me); bm.free()
    me.calc_loop_triangles()
    # split normals per loop -> expand to per-corner vertices, then weld identical (pos,normal)
    verts = {}; pos=[]; nrm=[]; idx=[]
    for tri in me.loop_triangles:
        for li in tri.loops:
            l = me.loops[li]
            v = me.vertices[l.vertex_index].co
            n = l.normal
            k = (round(v.x*gs,4), round(v.y*gs,4), round(v.z*gs,4), round(n.x,3), round(n.y,3), round(n.z,3))
            j = verts.get(k)
            if j is None:
                j = len(pos)//3; verts[k]=j
                pos += k[:3]; nrm += k[3:]
            idx.append(j)
    mn=[min(pos[i::3]) for i in range(3)]; mx=[max(pos[i::3]) for i in range(3)]
    out[key] = {"p":pos,"n":nrm,"i":idx,"bbox":[mn,mx],"src":rel+":"+name,"scale":gs}
    print(key, "verts", len(pos)//3, "tris", len(idx)//3, "bbox", [round(x,3) for x in mn], [round(x,3) for x in mx])
json.dump(out, open("meshes_raw.json","w"), separators=(",",":"))
print("bytes", os.path.getsize("meshes_raw.json"))
