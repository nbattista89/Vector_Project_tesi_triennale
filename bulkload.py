import mxnet as mx
import numpy as np
from collections import namedtuple
from mxnet.gluon.data.vision import transforms
from mxnet.contrib.onnx.onnx2mx.import_model import import_model
import os
import array
import oracledb
from timeit import default_timer as timer
from datetime import timedelta

start = timer()
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


# Load module before defining functions that use it
sym, arg_params, aux_params = import_model('resnet152-v2-7.onnx')

# Determine and set context
if len(mx.test_utils.list_gpus()) == 0:
    ctx = mx.cpu()
else:
    ctx = mx.gpu(0)

# Load module
mod = mx.mod.Module(symbol=sym, context=ctx,
 data_names=['data'], label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))], label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

def generate_all_embeddings(folder_path):
  embeddings = []
  for root, dirs, files in os.walk(folder_path):
    for filename in files:
      # Skip non-image files
      if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
      image_path = os.path.join(root, filename)
      embedding = gen_embeddings(image_path)
      embeddings.append((image_path,filename, embedding))
  return embeddings


# Folder path images
# Linux 
#folder_path = "/home/test/vectorai/images/"  # Replace with your folder path
# Windows
folder_path = "C:/Users/test/vectorai/images/"
embeddings = generate_all_embeddings(folder_path)

# Database connection
connection = oracledb.connect(
    user="test",
    password="welcome1",
    dsn="XXX.XXX.XXX.XXX/freepdb1"
)

with connection.cursor() as cursor:
    sql = """insert into image_vector (filename, embedding)
                     values (:filename,:vector_32)"""
    for image_path,filename, embedding in embeddings:
    #    print(f"Embedding for {filename}: {embedding}")
       cursor.execute(sql, (filename, embedding))
connection.commit()
connection.close()
end = timer()
print(timedelta(seconds=end-start))