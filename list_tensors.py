import tensorflow as tf

pb_path = r"C:\facenet_model\20180402-114759.pb"

with tf.io.gfile.GFile(pb_path, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.compat.v1.Session() as sess:
    tf.import_graph_def(graph_def, name="")
    print("\n===== All Ops in the Model =====")
    for op in sess.graph.get_operations():
        print(op.name)
