from deepface import DeepFace
import numpy as np
import os
import cv2
import dlib
import matplotlib.pyplot as plt
import mediapipe
from time import time


class detection_model():
    def __init__(self, backend, config = None, threshold = 0.8):
        assert backend in ['ssd', 'mediapipe', 'dlib']
        assert (threshold>0.) & (threshold<=1.)
        
        self.threshold = threshold
        self.backend = backend
        
        if config:
            self.config = config
        else:
            self.config = {}
        if self.backend=='ssd':
            
            prototxt = self.config['prototxt'] if 'prototxt' in self.config else"deploy.prototxt"
            model_path = self.config['model_path'] if 'model_path' in self.config else "res10_300x300_ssd_iter_140000_fp16.caffemodel"

            self.model = cv2.dnn.readNetFromCaffe(prototxt, model_path)
            
        elif self.backend=='mediapipe':
            
            model_selection = int(self.config['model_selection']) if 'model_selection' in self.config else 0
            min_detection_confidence = self.config['min_detection_confidence'] if 'min_detection_confidence' in self.config else 0.5
            
            self.model = mediapipe.solutions.face_detection.FaceDetection(model_selection=model_selection, min_detection_confidence=min_detection_confidence)
            
        elif self.backend=='dlib':
            raise NotImplementedError('dlib has not been implemented.')
            
            pass
    
    def predict(self, image):
        
        h, w = image.shape[:2]
        
        if self.backend=='ssd':
            self.model.setInput(cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0)))
            result = np.squeeze(self.model.forward())
            
            result = result[result[:,2]>0.]
            result = result[:,2:7] * np.array([1, w, h, w, h]) # confidence, xmin, ymin, xmax, ymax
            
        elif self.backend=='mediapipe':
            result_ = self.model.process(image)
            
            if not result_.detections:
                result = np.zeros(shape = (0,5))
            else:
                result = np.zeros(shape = (len(result_.detections), 5))

                for i in range(len(result_.detections)):
                    confidence = result_.detections[i].score[0]
                    xmin = result_.detections[i].location_data.relative_bounding_box.xmin
                    ymin = result_.detections[i].location_data.relative_bounding_box.ymin
                    width = result_.detections[i].location_data.relative_bounding_box.width
                    height = result_.detections[i].location_data.relative_bounding_box.height

                    result[i,:] = [confidence, xmin, ymin, xmin+width, ymin+height]

                result = result * np.array([1,w,h,w,h])
            
        elif self.backend=='dlib':
            raise NotImplementedError('dlib has not been implemented.')
            
            pass
        
        return result
    
    def pred_crop(self, image):
        
        result = self.predict(image)
        result = result[result[:,0]>self.threshold]
        
        # no face were detected
        if len(result)==0:
            return result, None
        
        face_ls = []
        for i in range(len(result)):
            
            conf = result[i, 0]
            _, xmin, ymin, xmax, ymax = result[i,:].astype('int')
            
            face_ls.append(image[ymin:ymax, xmin:xmax,:])
        
        return result, face_ls


class recognition_model():
    def __init__(self, model_name, db_path, threshold = 0.7):
        assert model_name in ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib']
        
        if model_name== 'VGG-Face' :
            self.target_shape = (224, 224)
        elif model_name == 'Facenet':
            self.target_shape = (160, 160)
        elif model_name == 'OpenFace':
            self.target_shape = (96, 96)
        elif model_name == 'DeepFace':
            self.target_shape = (152, 152)
        elif model_name == 'DeepID':
            self.target_shape = (47, 55)
        elif model_name == 'ArcFace':
            self.target_shape = (112, 112)
        elif model_name =='Dlib':
            self.target_shape = (150, 150)
            
        # 待修改 -> 調整成直接使用keras框架import model
        self.model = DeepFace.build_model(model_name)
        
        self.embedding_name = []
        self.embedding_ = []
        self.threshold = threshold
        
        # process image from database
        self.process_db(db_path)
        
    def process_db(self, db_path):
        
        file_list = [f for f in os.listdir(db_path) if f.split('.')[-1] in ['jpg', 'jpeg', 'png']]
        for filepath in file_list:
            
            # 待修改 -> 可以調整成只剩下人名
            name = filepath.split('.')[0]
            
            image = cv2.imread(os.path.join(db_path, filepath))[:,:,::-1]
            emb = self.get_embedding(image)
            
            self.embedding_name.append(name)
            self.embedding_.append(emb)
        
        print(f"{len(file_list)} images are processed in DataBase.")
    
    def get_embedding(self, image):
        image = cv2.resize(image, self.target_shape)
        
        # to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = np.stack([image, image, image], axis = -1)
        
        image = image[np.newaxis,:]
        
        embedding = np.squeeze(self.model.predict(image))
        
        return embedding
    
    def recognize(self, image_ls, sim = 'cosine'):
        
        assert sim in ['cosine', 'euclidean']
        
        if np.array(image_ls).ndim==3:
            image_ls = [image_ls]
        
        result = []
        for image in image_ls:
            emb = self.get_embedding(image)
            
            print(np.array(emb).shape)
            
            if sim=='cosine':

                # normalize
                emb = emb / np.linalg.norm(emb)
                
                db_embedding = np.array(self.embedding_) / np.linalg.norm(np.array(self.embedding_), axis = 1)[:,np.newaxis]

                cos_sim = np.dot(emb[np.newaxis,:], db_embedding.T)

                max_sim = cos_sim.max()
                max_id = self.embedding_name[cos_sim.argmax()]
            
            elif sim=='euclidean':
                
                dist = np.linalg.norm(self.embedding_ - emb, axis = 1)
                max_sim = dist.min() * -1.
                max_id = self.embedding_name[dist.argmin()]
            
            if max_sim> self.threshold:
                result.append((max_sim, max_id))
            else:
                result.append(None)
        
        return result
    
    def result_2_text(self, result):
        
        text_ls = []
        for r in result:
            if r==None:
                text = 'Unknown person.'
            else:
                text = f"{r[1]} : {r[0]:.3f}"
            
            text_ls.append(text)
        
        return text_ls

def draw_bbox(image, result, text = 'prob'):
    
    image_drawed = image.copy()
    font_scale = 1.0
    
    for i in range(0, result.shape[0]):
        start_x, start_y, end_x, end_y = result[i, 1:5].astype(np.int)
        cv2.rectangle(image_drawed, (start_x, start_y), (end_x, end_y), color=(255, 0, 0), thickness=2)

        if text=='prob':
            cv2.putText(image_drawed, f"{confidence*100:.2f}%", (start_x, start_y-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 2)
        else:
            cv2.putText(image_drawed, text[i], (start_x, start_y-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 2)
    return image_drawed



def main_loop():
    
    threshold = 0.6
    recognition_threshold = -120.
    fd_model = detection_model(backend = 'ssd',
                               config = {'prototxt':'models/opencv_face_detector/deploy.prototxt',
                                         'model_path':'models/opencv_face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel'},
                               threshold = threshold
                              )
    r_model = recognition_model(model_name='VGG-Face', db_path = 'img/database/', threshold = recognition_threshold)

    
    cap = cv2.VideoCapture(0)

    while True:
        
        now = time()
        _, image = cap.read()
        
        # detection
        result, face_ls = fd_model.pred_crop(image)
        
        if len(result)==0:
            image_drawed = image.copy()
        else:
            # recognition
            rec_result = r_model.recognize(face_ls, sim = 'euclidean')

            # draw bbox
            text_draw = r_model.result_2_text(rec_result)
            image_drawed = draw_bbox(image, result, text = text_draw)
        
        # fps calculation
        fps = 1/(time() - now)
        cv2.putText(image_drawed, f"fps : {fps:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow("image", image_drawed)
        if cv2.waitKey(1) == ord("q"):
            break
            
            
    cv2.destroyAllWindows()
    cap.release()