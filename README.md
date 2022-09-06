# Prototype Face Anti-spoofing
- This is a single shot face anti-spoofing prototype project.
- The deep learning framework is Pytorch. Python 3.7.9 is used.
# Installation
- sudo -s
- sh install_requirements.sh
- exit
## Face landmarks
- face_alignment is used for landmarks extraction. Page [face_alignment](https://github.com/1adrianb/face-alignment). Thanks to them.
### Landmarks extraction scripts
- cd detlandmark&&python3 detlandmark_imgs.py dataset_dir

- You can change corresponding directory and filename in config.py
- For example train_filelists=[
      ['raw/ClientRaw','raw/client_train_raw.txt',GENUINE],
      ['raw/ImposterRaw','imposter_train_raw.txt',ATTACK]]
     test_filelists=[
      ['raw/ClientRaw','raw/client_test_raw.txt',GENUINE],
      ['raw/ImposterRaw','raw/imposter_test_raw.txt',ATTACK]
      ]
## Method
- Our method is straightforward. Small patched containing a face is cropped with corresponding landmarks. A binary classification network is used to distinguish the attack patches.

## Training
- First, edit file *config.py*, choose the target network and proper batch_size.
- Then, in terminal command: `make clean&&make&&python3 main.py train`
## Inference
- In terminal command: `python3 inference.py inference --images='detlandmark/inference_images/*/*.jpg'`
- The inference report is result/inference.txt, you can check it in commad: `cat result/inference.txt`
## Visualize Dataset
- We have fixed the bug of choice wrong face in multiple detected faces with standard of coordinates. 
- To visualize cropped faces in dataset. Please run command: `python3 vis_cropface.py visualize`
- All faces will be shown in data/showcropface_train.jpg and data/showcropface_val.jpg

## Reference
- Single-Shot Face Anti-Spoofing for Dual Pixel Camera; Xiaojun Wu and al.