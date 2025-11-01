from gqcnn.model import get_fc_gqcnn_model
import json
import tensorflow.compat.v1 as tf

def get_gqcnn():
  
  tf.disable_v2_behavior()
  '''
  reader = tf.train.load_checkpoint("FC-GQCNN-4.0-PJ/FC-GQCNN-4.0-PJ/model.ckpt")
  for k in reader.get_variable_to_shape_map():
      print(k)
  '''
  with open("./robot_chef/FC-GQCNN-4.0-PJ/FC-GQCNN-4.0-PJ/config.json", "r") as f:
    config = json.load(f)
  config = config["gqcnn"]
  gqcnn = get_fc_gqcnn_model()
  gqcnn = gqcnn.load("./robot_chef/FC-GQCNN-4.0-PJ/FC-GQCNN-4.0-PJ", config, "gqcnn_log.txt")
  gqcnn.open_session()
  return gqcnn