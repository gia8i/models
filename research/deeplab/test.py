import os
import cv2
import argparse
import numpy as np
import os.path as osp
from PIL import Image
import tensorflow as tf


class GenericTFModel(object):
    def __init__(self, pb_path, height, width, input_name,
                 output_name_list, config=None):
        tf.reset_default_graph()
        self.graph = tf.Graph()
        self.pb_path = pb_path
        self.width = width
        self.height = height

        self.graph_def = None
        with tf.gfile.GFile(pb_path, 'rb') as f:
            self.graph_def = tf.GraphDef.FromString(f.read())

        if self.graph_def is None:
            raise RuntimeError('Cannot open pb file provided')

        with self.graph.as_default():
            tf.import_graph_def(self.graph_def, name='')

        self.sess = tf.Session(graph=self.graph, config=config)

        self.output_name_list = output_name_list
        self.input_name = input_name

    def run(self, image):
        rimage = cv2.resize(image, (self.width, self.height))
        output = self.sess.run(self.output_name_list,
                               feed_dict={self.input_name: [rimage]})

        return output


def fix_transparency(image):
    if (not Image.isImageType(image)) and (isinstance(image, np.ndarray)):
        image = Image.fromarray(image)
    new_image = Image.new("RGBA", image.size, "WHITE")  # Create a white rgba background
    new_image.paste(image, (0, 0), image)  # Paste the image on the background. Go to the links given below for details.
    return np.array(new_image.convert('RGB'))


def fix_grayscale(image):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    if len(image.shape) > 2 and image.shape[2] > 1:
        return image
    return np.dstack((image, image, image))


def mkpath(path1, path2):
    return os.path.join(path1, path2)


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
WIDTH = 513
HEIGHT = 513


def main():
    parser = argparse.ArgumentParser("Semantic Segmentation.")
    parser.add_argument("--input_image", type=str, help="Path to input image.", default=None)
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint", default=None)
    parser.add_argument("--output_dir", type=str, help="Path to output dir", default=None)

    args = parser.parse_args()

    if args.input_image is None:
        print("No input image path passed to application. Exiting...")
        return -1
    if args.checkpoint is None:
        print("No checkpoint path is passed to application. Exitting...")
        return -2
    if args.output_dir is None:
        print("No output dir is passed to application. Exitting...")
        return -3

    if not osp.exists(args.input_image) or not osp.isfile(args.input_image):
        print("Input image does not exist. Exitting...")
        return -4

    if not osp.exists(args.checkpoint) or not osp.isfile(args.checkpoint):
        print("Checkpoint file does not exist. Exitting...")
        return -5

    if not osp.exists(args.output_dir) or not osp.isdir(args.output_dir):
        print("Output dir does not exist. Exitting...")
        return -6

    output_name_list = ['SemanticPredictions:0']
    input_name = 'ImageTensor:0'

    model = GenericTFModel(args.checkpoint, HEIGHT, WIDTH, input_name,
                           output_name_list)

    rgb = np.array(Image.open(args.input_image))
    if rgb.shape[2] == 4:
        rgb = fix_transparency(rgb)

    ori_h = rgb.shape[0]
    ori_w = rgb.shape[1]

    img_name = os.path.splitext(os.path.basename(args.input_image))[0]

    rgb_resized = cv2.resize(rgb, (WIDTH, HEIGHT))
    ret = model.run(rgb_resized)[0]

    ret = ret.astype(np.uint8)

    rgb_subtracted = (np.ones((rgb.shape[0], rgb.shape[1], 3), np.uint8) * 255).astype(np.uint8)
    ret = cv2.resize(ret.squeeze(0), (ori_w, ori_h))
    ret3 = np.dstack((ret, ret, ret))
    np.copyto(rgb_subtracted, rgb, where=ret3.astype(np.bool))

    concat = np.concatenate((rgb_subtracted, rgb), axis=1)
    out_path = os.path.join(args.output_dir, img_name + '.png')
    Image.fromarray(concat).save(out_path)

    return 0


if __name__ == "__main__":
    print(main())
