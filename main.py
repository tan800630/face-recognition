import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from module import *
import PIL
from PIL import ImageTk


class App():
    def __init__(self, root, window_title = '', video_source = 0, size = (1800, 500)):
        
        ## attributes for main window
        
        self.root = root
        self.root.title(window_title)
        self.window = tk.Frame(self.root, width = size[0], height = size[1])
        
        ##  attributes for streaming
        
        self.video_source = video_source
        self.vid = VideoCapture(video_source = self.video_source)
        self.delay = 15
        self.streaming_id = None
        
        ## attributes for face detection and recognition
        
        self.image = None
        self.face_id = None
        self.detection_faces = None
        self.detection_result = None
        self.recognition_result = None
        self.detection_model = detection_model(backend = 'ssd',
                               config = {'prototxt':'models/opencv_face_detector/deploy.prototxt',
                                         'model_path':'models/opencv_face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel'},
                               threshold = 0.8
                              )
        self.recognition_model = recognition_model(model_name='VGG-Face', db_path = 'img/database/', threshold = 0.7)
        
        ## string variables
        
        self.open_filename = tk.StringVar()
        self.save_filepath = tk.StringVar()
        self.prediction_text = tk.StringVar()
        self.prediction_text.set('Prediction.')
        
        
        ## widgets 
        
        # canvas for video streaming
        self.streaming_w = tk.Canvas(self.root, width = int(size[0] * 1/3), height = int(size[1] * 6/7))
        # streaming button
        self.streaming_button = tk.Button(self.root, text = 'Use Webcam', command = self.update)
        # SnapShot button
        self.snapshot_button = tk.Button(self.root, text = 'Snapshot', command = self.snapshot)
        # open image button
        self.open_image_button = tk.Button(self.root, text = 'Open file', command = self.openfile_and_readimg)
        # canvas for face detection
        self.face_w = tk.Canvas(self.root, width = int(size[0] * 1/3 * 1/3), height = int(size[1] * 1/3))
        # Prediction text
        self.pred_text = tk.Label(self.root, textvariable = self.prediction_text)
        # Detect Face button
        self.detect_button = tk.Button(self.root, text = 'Face Detection', command = self.face_detect_and_draw)
        # Previous & Next button
        self.prev_button = tk.Button(self.root, text = 'Prev', command = self.prev_face)
        self.next_button = tk.Button(self.root, text = 'Next', command = self.next_face)
        # Save file button
        self.save_button = tk.Button(self.root, text = 'Save', command = self.savefile)
        
        
        # Layout setting
        self.streaming_w.grid(row = 0, column = 0, columnspan = 3, rowspan = 6)
        self.streaming_button.grid(row = 6, column = 0, pady = 10)
        self.snapshot_button.grid(row = 6, column = 1, pady = 10)
        self.open_image_button.grid(row = 6, column = 2, pady = 10)
        self.vseparator = ttk.Separator(self.root, orient='vertical').grid(row = 0, column = 3, rowspan = 6, sticky = 'ns')
        self.face_w.grid(row = 0, column = 4, rowspan = 2, padx = 10, pady = 10)
        self.pred_text.grid(row = 2, column = 4, columnspan = 2, padx = 10, pady = 10)
        self.detect_button.grid(row = 3, column = 4, columnspan = 2, padx = 10)
        self.prev_button.grid(row = 0, column = 6, sticky = tk.S, padx = 10, pady = 5)
        self.next_button.grid(row = 1, column = 6, sticky = tk.N, padx = 10, pady = 5)
        self.prev_button['state'] = tk.DISABLED
        self.next_button['state'] = tk.DISABLED
        self.hseparator = ttk.Separator(self.root, orient='horizontal').grid(row = 4, column = 4, columnspan = 2, sticky = 'ew')
        self.save_button.grid(row = 5, column = 4)
        
        self.window.mainloop()

        
    def openfile_and_readimg(self):
        
        # cancel video streaming and enable streaming button
        if self.streaming_id:
            self.window.after_cancel(self.streaming_id)
            self.streaming_button['state'] = tk.NORMAL
        
        # reset detection_faces
        self.detection_faces = None
        self.prev_button['state'] = tk.DISABLED
        self.next_button['state'] = tk.DISABLED
        
        filepath = filedialog.askopenfilename(parent = self.root,
                                              initialdir = './')
        self.open_filename.set(filepath)
        
        img = cv2.imread(filepath)[:,:,::-1]
        self.image = img
        
        self.draw(self.image, self.streaming_w)

        
    def savefile(self):
        f = filedialog.asksaveasfile(parent = self.root,
                                     initialdir = './',
                                     initialfile = 'Untitle.png',
                                     defaultextension = '.png',
                                     confirmoverwrite = True)
        
        self.dirname = dirname
        self.save_filepath.set(dirname)
        
    def update(self):
        
        # reset snapshot photo
        self.image = None
        
        ret, frame = self.vid.get_frame()
        
        if ret:
            self.draw(frame, self.streaming_w)

        
        self.streaming_id = self.window.after(self.delay, self.update)
        self.streaming_button['state'] = tk.DISABLED
        
    def snapshot(self):
        
        # cancel video streaming and enable streaming_button
        if self.streaming_id:
            self.window.after_cancel(self.streaming_id)
            self.streaming_button['state'] = tk.NORMAL
        
        # reset detection_faces
        self.detection_faces = None
        self.prev_button['state'] = tk.DISABLED
        self.next_button['state'] = tk.DISABLED
        
        ret, frame = self.vid.get_frame()
        self.image = frame
        
        if ret:
            self.draw(frame, self.streaming_w)
            
    def face_detect_and_draw(self):
        
        if not (self.image is None):
        
            # detection
            result, face_ls = self.detection_model.pred_crop(self.image)
            
            if len(face_ls)!=0:

                self.detection_result = result
                self.detection_faces = face_ls
                 
                # recognition
                rec_result = self.recognition_model.recognize(face_ls)
                # draw bbox
                self.recognition_result = self.recognition_model.result_2_text(rec_result)         
                
                
                # draw
                self.face_id = 0
                self.face_img = self.detection_faces[self.face_id]
                self.draw(self.detection_faces[self.face_id], self.face_w)
                
                # draw prediction text
                self.prediction_text.set(self.recognition_result[self.face_id])
            
            if len(face_ls)>1:
                self.prev_button['state'] = tk.NORMAL
                self.next_button['state'] = tk.NORMAL
            else:
                self.prev_button['state'] = tk.DISABLED
                self.next_button['state'] = tk.DISABLED

    
    def next_face(self):
        
        if self.face_id<(len(self.detection_faces)-1):
            self.face_id+=1

            self.face_img = self.detection_faces[self.face_id]
            self.draw(self.detection_faces[self.face_id], self.face_w)
            
            # draw prediction text
            self.prediction_text.set(self.recognition_result[self.face_id])
    
    def prev_face(self):
        
        if self.face_id>0:
            self.face_id-=1

            self.face_img = self.detection_faces[self.face_id]
            self.draw(self.detection_faces[self.face_id], self.face_w)

            # draw prediction text
            self.prediction_text.set(self.recognition_result[self.face_id])
    
    def draw(self, image, canvas):
        
        image = cv2.resize(image, (int(canvas['width']), int(canvas['height'])))
        image = ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
        
        canvas.image = image
        canvas.create_image(0, 0, image = canvas.image, anchor = tk.NW)
        

            
class VideoCapture():
    def __init__(self, video_source = 0):
        
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        
#         self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
#         self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        
        else:
            return (ret, None)

        
App(tk.Tk(), "")