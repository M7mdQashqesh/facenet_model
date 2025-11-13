import tensorflow as tf

pb_path = r"C:\facenet_model\20180402-114759.pb"
tflite_path = r"C:\facenet_model\facenet.tflite"

print("📦 Loading frozen graph...")
with tf.io.gfile.GFile(pb_path, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

print("🔧 Importing graph into new Graph...")
tf.compat.v1.reset_default_graph()

def wrap_frozen_graph(graph_def, inputs, outputs):
    def _imports():
        tf.compat.v1.import_graph_def(graph_def, name="")
    wrapped = tf.compat.v1.wrap_function(_imports, [])
    graph = wrapped.graph
    return wrapped.prune(
        tf.nest.map_structure(graph.as_graph_element, inputs),
        tf.nest.map_structure(graph.as_graph_element, outputs),
    )

input_tensor = "input:0"
phase_tensor = "phase_train:0"
batch_tensor = "batch_size:0"
embedding_tensor = "embeddings:0"

print("⚡ Wrapping frozen graph...")
frozen = wrap_frozen_graph(
    graph_def,
    inputs=[input_tensor, phase_tensor, batch_tensor],
    outputs=[embedding_tensor],
)

@tf.function
def facenet_func(img):
    phase = tf.constant(False, dtype=tf.bool)
    batch = tf.constant(1, dtype=tf.int32)
    # كل الـ arguments لازم يكونوا Tensors
    return frozen(img, phase, batch)

concrete_func = facenet_func.get_concrete_function(
    tf.TensorSpec([1, 160, 160, 3], tf.float32)
)

print("🔁 Building converter...")
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("🔨 Converting...")
tflite_model = converter.convert()

print("💾 Saving TFLite model...")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("🎉 DONE — saved at:", tflite_path)
