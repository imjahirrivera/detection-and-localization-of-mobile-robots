# Settings to run

## Python Version
3.8.0

## Virtual Environment
(python3.8.0) -m venv venv
-- ./venv/Scripts/activate
-- deactivate

## Ultralytics for YOLO(No specific version)
pip install ultralytics

## Pyzbar for QR (No specific version)

pip install pyzbar

## PyTorch to run process with GPU

1. Go to: https://pytorch.org/get-started/locally/
2. Copy and paste the command "Run this Command" into the terminal as shown in the image:
![image](https://github.com/user-attachments/assets/92b783a1-60bb-40ac-b0a2-e9198e890c52)

3. Modify code and add "--upgrade" between "install" and "torch". <br>
   Example: pip3 install *--upgrade* torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     
