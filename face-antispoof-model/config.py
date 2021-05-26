# import the necessary packages

import torch
import warnings

class DefaultConfig(object):
   model = 'Myxception' # the name must be the same as the model used /__init__.py
   env = model #
   ATTACK = 1
   GENUINE = 0
   LABELS = ['Genuine', 'Attack']
   train_filelists=[
        ['/Users/steven/Public/Project/python/face-antispoofing-model/face-antispoof-model/raw/ClientRaw', '/Users/steven/Public/Project/python/face-antispoofing-model/face-antispoof-model/raw/client_train_raw.txt', GENUINE],
        ['/Users/steven/Public/Project/python/face-antispoofing-model/face-antispoof-model/raw/ImposterRaw', '/Users/steven/Public/Project/python/face-antispoofing-model/face-antispoof-model/raw/imposter_train_raw.txt', ATTACK]
    ]

   test_filelists=[
        ['/Users/steven/Public/Project/python/face-antispoofing-model/face-antispoof-model/raw/ClientRaw', '/Users/steven/Public/Project/python/face-antispoofing-model/face-antispoof-model/raw/client_test_raw.txt',GENUINE],
        ['/Users/steven/Public/Project/python/face-antispoofing-model/face-antispoof-model/raw/ImposterRaw','/Users/steven/Public/Project/python/face-antispoofing-model/face-antispoof-model/raw/imposter_test_raw.txt',ATTACK]
    ]
   #load_model_path = 'checkpoints/model.pth'
   load_model_path = None

   batch_size = 16 # batch size
   use_gpu = torch.cuda.is_available() # use GPU or not
   #use_gpu = True # use GPU or not
   num_workers = 8 # how many workers for loading data
   print_freq = 20 # print info every N batch
   debug_file = '/tmp/debug' # if os.path.exists(debug_file): enter ipdb
   result_name = 'result'

   max_epoch = 10
   lr = 0.01 # initial learning rate
   lr_decay = 0.5 # when val_loss increase, lr = lr*lr_decay
   lr_stepsize=3 #learning step size
   weight_decay = 1e-5 #
   cropscale = 3.5
   image_size = 224
def parse(self, kwargs):
   '''
   :param self:
   :param kwargs:
   :return:
   '''
   for k, v in kwargs.items():
      if not hasattr(self, k):
          warnings.warn("Warning: opt has not attribut %s" %k)
      setattr(self, k, v)
   print('user config:')
   for k, v in self.__class__.__dict__.items():
      if not k.startswith('__'):
          print(k, getattr(self, k))

DefaultConfig.parse = parse
opt = DefaultConfig()