import os, sys, bpy, glob

scene = bpy.context.scene
args = sys.argv[sys.argv.index("--") + 1 :]
path = os.path.abspath(args[0])
if path[-1] != "/":
    path += "/"
print("output path:", path)
scene.render.filepath = path
bpy.context.scene.render.image_settings.file_format = "FFMPEG"
preset_script = os.path.join(sys.prefix, "../scripts/presets/ffmpeg/h264_in_MP4.py")
exec(open(preset_script).read())

files = [f.split("/")[-1] for f in sorted(glob.glob(os.path.join(path, "*.png")))]
scene.sequence_editor_create()

for seq in scene.sequence_editor.sequences:
    if seq["created_by_bsb"]:
        scene.sequence_editor.sequences.remove(seq)

start = int(files[0].split("/")[-1].split(".")[0].split("_")[-1])
seq = scene.sequence_editor.sequences.new_image(
    name="FullStrip", filepath=os.path.join(path, files[0]), channel=1, frame_start=start
)
seq["created_by_bsb"] = True
for f in files:
    seq.elements.append(f)


bpy.context.scene.frame_start = start
bpy.context.scene.frame_end = start + len(seq.elements) - 1
bpy.context.scene.render.use_compositing = False
bpy.context.scene.render.use_sequencer = True

print(bpy.ops.render.render(animation=True))
