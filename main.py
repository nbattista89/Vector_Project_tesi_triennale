import mxnet as mx
import numpy as np
from collections import namedtuple
from mxnet.gluon.data.vision import transforms
from mxnet.contrib.onnx.onnx2mx.import_model import import_model
import array
import oracledb
from timeit import default_timer as timer
from datetime import timedelta

with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

Batch = namedtuple('Batch', ['data'])


def get_image(path, show=False):
    img = mx.image.imread(path)
    if img is None:
        return None
    return img


def preprocess(img):
    transform_fn = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform_fn(img)
    img = img.expand_dims(axis=0)
    return img


def predict(path):
    img = get_image(path, show=True)
    img = preprocess(img)
    mod.forward(Batch([img]))
    # Take softmax to generate probabilities
    scores = mx.ndarray.softmax(mod.get_outputs()[0]).asnumpy()
    # print the top-5 inferences class
    scores = np.squeeze(scores)
    a = np.argsort(scores)[::-1]
    results = {}
    for i in a[0:5]:
        results[labels[i]] = float(scores[i])
    return results


def gen_embeddings(path):
    img = get_image(path, show=True)
    img = preprocess(img)
    mod.forward(Batch([img]))
    scores = mx.ndarray.softmax(mod.get_outputs()[0]).asnumpy()
    # return it ready for db insertion
    return array.array("f", scores.squeeze())


sym, arg_params, aux_params = import_model('resnet152-v2-7.onnx')

# Determine and set context
if len(mx.test_utils.list_gpus()) == 0:
    ctx = mx.cpu()
else:
    ctx = mx.gpu(0)

# Load module
mod = mx.mod.Module(symbol=sym, context=ctx, data_names=['data'], label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))], label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

# Linux Path
#vector1_data_32=gen_embeddings('/home/test/example.jpg')
# Windows Path
vector1_data_32=gen_embeddings('C:/Users/test/Downloads/labrador.jpeg')

connection = oracledb.connect(
    user="test",
    password="welcome1",
    dsn="XXX.XXX.XXX.XXX/freepdb1"
)

start = timer()
with connection.cursor() as cursor:
    sql = """SELECT id, image_path FROM image_vector3
        ORDER BY VECTOR_DISTANCE(vector_32, :vector_new) FETCH FIRST 5 ROWS ONLY"""
    try:
            cursor.execute(sql, vector_new=vector1_data_32)
            for row in cursor:
                print(row)
    except oracledb.Error as err:
        print(f"Error executing query: {err}")
connection.close()
end = timer()
print(timedelta(seconds=end-start))