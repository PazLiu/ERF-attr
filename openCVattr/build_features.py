import os
import pickle as pickle
import cv2
import numpy as np
from sklearn.cluster import KMeans

class StarFeatureDetector(object):
    def __init__(self):
        self.detector = cv2.xfeatures2d.StarDetector_create()
        
    def detect(self,img):
        return self.detector.detect(img)

def load_training_data(input_folder):
    training_data = []

    if not os.path.isdir(input_folder):
        raise IOError("The folder " + input_folder + " doesn't exist")
        
    for root, dirs, files in os.walk(input_folder):
        for filename in (x for x in files if x.endswith('.jpg')):
            filepath = os.path.join(root, filename)
            print(filepath)
            object_class = filepath.split('\\')[-2]
            training_data.append({'object_class': object_class, 
                'image_path': filepath})
                    
    return training_data

class FeatureBuilder(object):
    def extract_features(self, img):
        keypoints = StarFeatureDetector().detect(img)
        keypoints, feature_vectors = compute_sift_features(img, keypoints)
        return feature_vectors

    def get_codewords(self, input_map, scaling_size, max_samples=12):
        keypoints_all = []
        
        count = 0
        cur_class = ''
        for item in input_map:
            if count >= max_samples:
                if cur_class != item['object_class']:
                    count = 0
                else:
                    continue

            count += 1

            if count == max_samples:
                print("Built centroids for", item['object_class'])

            cur_class = item['object_class']
            img = cv2.imread(item['image_path'])
            img = resize_image(img, scaling_size)

            num_dims = 128
            feature_vectors = self.extract_features(img)
            keypoints_all.extend(feature_vectors) 

        kmeans, centroids = BagOfWords().cluster(keypoints_all)
        return kmeans, centroids
    
    
    
    
class BagOfWords(object):
    def __init__(self, num_clusters=32):
        self.num_dims = 128
        self.num_clusters = num_clusters
        self.num_retries = 10

    def cluster(self, datapoints):
        kmeans = KMeans(self.num_clusters, 
                        n_init=max(self.num_retries, 1),
                        max_iter=10, tol=1.0)

        res = kmeans.fit(datapoints)
        centroids = res.cluster_centers_
        return kmeans, centroids

    def normalize(self, input_data):
        sum_input = np.sum(input_data)

        if sum_input > 0:
            return input_data / sum_input
        else:
            return input_data

    def construct_feature(self, img, kmeans, centroids):
        keypoints = StarFeatureDetector().detect(img)
        keypoints, feature_vectors = compute_sift_features(img, keypoints)
        labels = kmeans.predict(feature_vectors)
        feature_vector = np.zeros(self.num_clusters)

        for i, item in enumerate(feature_vectors):
            feature_vector[labels[i]] += 1

        feature_vector_img = np.reshape(feature_vector, 
                ((1, feature_vector.shape[0])))
        return self.normalize(feature_vector_img)    
    
 # Extract features from the input images and 
# map them to the corresponding object classes
def get_feature_map(input_map, kmeans, centroids, scaling_size):
    feature_map = []
     
    for item in input_map:
        temp_dict = {}
        temp_dict['object_class'] = item['object_class']
    
        print ("Extracting features for", item['image_path'])
        img = cv2.imread(item['image_path'])
        img = resize_image(img, scaling_size)

        temp_dict['feature_vector'] = BagOfWords().construct_feature(
                    img, kmeans, centroids)

        if temp_dict['feature_vector'] is not None:
            feature_map.append(temp_dict)

    return feature_map   

# Extract SIFT features
def compute_sift_features(img, keypoints):
    if img is None:
        raise TypeError('Invalid input image')

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = cv2.xfeatures2d.SIFT_create().compute(img_gray, keypoints)
    return keypoints, descriptors
    
# Resize the shorter dimension to 'new_size' 
# while maintaining the aspect ratio
def resize_image(input_img, new_size):
    h, w = input_img.shape[:2]
    scaling_factor = new_size / float(h)

    if w < h:
        scaling_factor = new_size / float(w)

    new_shape = (int(w * scaling_factor), int(h * scaling_factor))
    return cv2.resize(input_img, new_shape)     
    
if __name__=='__main__':
    data_folder = 'mit-train'
    scaling_size = 200
    codebook_file = 'codebook.pkl'
    feature_map_file = 'feature_map.pkl'

    training_data = load_training_data(data_folder)

    kmeans, centroids = FeatureBuilder().get_codewords(training_data, scaling_size)

    if codebook_file:
        with open(codebook_file, 'wb') as f:
            print(f.name)
            pickle.dump((kmeans, centroids), f)

    feature_map = get_feature_map(training_data, kmeans, centroids, scaling_size)
    if feature_map_file:
        with open(feature_map_file, 'wb') as f:
            pickle.dump(feature_map, f)

    
    