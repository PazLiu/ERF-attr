import pickle
import cv2
import openCVattr.build_features as bf
from openCVattr.trainer import ERFTrainer


class ImageTagExtractor(object):
    def __init__(self, model_file, codebook_file):
        with open(model_file, 'rb') as f:
            self.erf = pickle.load(f)

        with open(codebook_file, 'rb') as f:
            self.kmeans, self.centroids = pickle.load(f)

    def predict(self, img, scaling_size):
        img = bf.resize_image(img, scaling_size)
        feature_vector = bf.BagOfWords().construct_feature(img, self.kmeans, self.centroids)
        image_tag = self.erf.classify(feature_vector)[0]
        return image_tag

if __name__=='__main__':

    model_file = 'erf.pkl'
    codebook_file = 'codebook.pkl'
    input_image = cv2.imread('test.jpg' )

    scaling_size = 200

    print ("Output:", ImageTagExtractor(model_file, codebook_file).predict(input_image, scaling_size))
    
    
    
    
    
    
    