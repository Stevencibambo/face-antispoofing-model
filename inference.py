#codin:utf8
# import the necessary package

from imutils.video import VideoStream
from torchsummary import summary
from torchvision import transforms
from torch.autograd import Variable
from config import opt
import models
import face_alignment
from skimage import io
from torchnet import meter
from utils import Visualizer
from tqdm import tqdm
import torchvision
import imutils
import torch
import json
import numpy as np
import cv2
import os
import time

class DataHandle():
    def __init__(self, scale=2.7, image_size=224, use_gpu=False, transform=None, data_source = None):
        self.transform = transform
        self.scale = scale
        self.image_size = image_size
        dir_detector = "face_detector"
        protoPath = os.path.sep.join([dir_detector, "deploy.prototxt"])
        modelPath = os.path.sep.join([dir_detector, "res10_300x300_ssd_iter_140000.caffemodel"])
        self.net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        if use_gpu:
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        else:
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,device='cpu')

    def det_img(self, imgdir, image):
        input = image
        if(len(imgdir) > 5):
            input = io.imread(imgdir)

        preds = self.fa.get_landmarks(input)
        if 0:
            for pred in preds:
                img = cv2.imread(imgdir)
                print('ldmk num:', pred.shape[0])
                for i in range(pred.shape[0]):
                    x, y = pred[i]
                    print(x, y)
                    cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
                cv2.imshow('-', img)
                cv2.waitKey()
        return preds

    def crop_with_ldmk(self, image, landmark):
        ct_x, std_x = landmark[:, 0].mean(), landmark[:, 0].std()
        ct_y, std_y = landmark[:, 1].mean(), landmark[:, 1].std()

        std_x, std_y = self.scale * std_x, self.scale * std_y

        src = np.float32([(ct_x, ct_y), (ct_x + std_x, ct_y + std_y), (ct_x + std_x, ct_y)])
        dst = np.float32([((self.image_size -1 )/ 2.0, (self.image_size -1)/ 2.0),
                  ((self.image_size-1), (self.image_size -1 )),
                  ((self.image_size -1 ), (self.image_size - 1)/2.0)])
        retval = cv2.getAffineTransform(src, dst)
        result = cv2.warpAffine(image, retval, (self.image_size, self.image_size), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT)

        return result

    def get_data(self, image_path, image): #[img,label]

        img = image
        if(len(image_path) > 5):
            img = cv2.imread(image_path)

        ldmk = np.asarray(self.det_img(image_path, image), dtype=np.float32)
        if 0:
            for pred in ldmk:
                for i in range(pred.shape[0]):
                    x, y = pred[i]
                    cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        ldmk = ldmk[np.argsort(np.std(ldmk[:, :, 1], axis=1))[-1]]
        img = self.crop_with_ldmk(img, ldmk)
        if 0:
            cv2.imshow('crop face', img)
            cv2.waitKey()

        return np.transpose(np.array(img, dtype = np.float32), (2, 0, 1)), image_path

    def __len__(self):
        return len(self.img_label)

def inference(**kwargs):
    import glob
    images = glob.glob(kwargs['images'])
    assert len(images) > 0
    data_handle = DataHandle(scale = opt.cropscale, use_gpu = opt.use_gpu, transform = None, data_source='none')
    pths = glob.glob('checkpoints/%s/*.pth'%(opt.model))
    pths.sort(key= os.path.getmtime, reverse=False)
    print(pths)
    opt.parse(kwargs)
    opt.load_model_path = pths[0]
    model = getattr(models, opt.model)().eval()
    assert os.path.exists(opt.load_model_path)
    if opt.load_model_path:
       model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()
    model.train(False)
    fopen = open('result/inference.txt', 'w')
    tqbar = tqdm(enumerate(images), desc='Inference with %s'%(opt.model))
    for idx, imgdir in tqbar:
        data, _ = data_handle.get_data(imgdir, "")
        data = data[np.newaxis, :]
        data = torch.FloatTensor(data)
        with torch.no_grad():
            if opt.use_gpu:
                data = data.cuda()
            outputs = model(data)
            outputs = torch.softmax(outputs,dim=-1)
            preds = outputs.to('cpu').numpy()
            attack_prob = preds[:, opt.ATTACK]
            genuine_prob = preds[:, opt.GENUINE]
            tqbar.set_description(desc = 'Inference %s attack_prob=%f genuine_prob=%f with %s'%(imgdir, attack_prob,genuine_prob, opt.model))
            print('Inference %s attack_prob=%f genuine_prob=%f'%(imgdir, attack_prob, genuine_prob), file=fopen)
    fopen.close()
def help():
    '''
    python file.py help
    '''
    print('''
    usage : python {0} <function> [--args=value,]
    <function> := train | test | help
    example:
           python {0} train --env='env0701' --lr=0.01
           python {0} test --dataset='path/to/dataset/root/'
           python {0} inference --images='image dirs'
           python {0} help
    avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

'''inference with live camera'''
def demo():
    import glob
    # vs = VideoStream(src=2).start()
    vs = cv2.VideoCapture(0) # 'rtsp://admin:abcd@1234@10.0.0.88:554'
    time.sleep(2.0)
    data_handle = DataHandle(scale=opt.cropscale, use_gpu=opt.use_gpu, transform=None, data_source='none')
    
    while True:
        # frame = vs.read()
        ret, frame = vs.read()
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame,
                                                (300, 300)), 1.0, (300, 300), (104, 177.0, 123.0))
        #pass the blob through the network and obtain the detection and prediction
        data_handle.net.setInput(blob)
        detections = data_handle.net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e. probability) assorciate with the prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections
            if confidence > 0.5:
                # compute the (X,y) -coordinates of the bounding box for the face and extract the face ROI
                box = detections[0, 0, i, 3:7] * np.array(([w, h, w, h]))
                (startX, startY, endX, endY) = box.astype("int")
                # ensure the detected bounding box does fall outside the dimensions of the frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                image = cv2.resize(frame, (300, 300))
                data, _ = data_handle.get_data("", frame)

                data = data[np.newaxis, :]
                data = torch.FloatTensor(data)
                with torch.no_grad():
                    if opt.use_gpu:
                        data = data.cuda()
                    outputs = model(data)
                    print(data.shape)
                    outputs = torch.softmax(outputs, dim=-1)
                    preds = outputs.to('cpu').numpy()
                    index = np.argmax(preds[0])

                    attack_prob = preds[:, opt.ATTACK]
                    genuine_prob = preds[:, opt.GENUINE]
                    print('Inference %s attack_prob=%f genuine_prob=%f with %s' % ("img from cam", attack_prob, genuine_prob, opt.model))
                    box_coords = ((startX, startY - 22), (startX + 70, startY))
                    cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 250, 0), cv2.FILLED)
                    cv2.putText(frame, '{}'.format(opt.LABELS[index]), (startX, startY - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.imshow("Camera 01", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the q key was pressed, break from the loop
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    vs.stop()

if __name__=='__main__':
    import fire
    import glob

    pths = glob.glob('checkpoints/%s/*.pth' % (opt.model))
    pths.sort(key=os.path.getmtime, reverse=True)

    opt.load_model_path = pths[0]
    model = getattr(models, opt.model)().eval()
    assert os.path.exists(opt.load_model_path)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()
    model.train(False)

    fire.Fire()
