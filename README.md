# How to Setup Environment for WSL2

## Windows Side

1. Install Windows GPU Drivers  
https://www.nvidia.com/Download/index.aspx

2. Install CUDA Toolkit in WSL2  
https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local

3. Add .wslconfig file  
Go to `%USERPROFILE%\.wslconfig`. Create it if it doesn't exist.  
Add the following:
```
[wsl2]
networkingMode=mirrored
```

4. Allow inbound connections through the Hyper-V firewall  
https://superuser.com/questions/1717753/how-to-connect-to-windows-subsystem-for-linux-from-another-machine-within-networ
Run Powershell with administrator privileges and input the following command:
```
Set-NetFirewallHyperVVMSetting -Name "{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}" -DefaultInboundAction Allow
```

## Ubuntu Side

1. Install Python 3.10  
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.10
```

2. Create .venv

3. Install Dependencies
```
pip install -r requirements.txt
```
This will return errors.

4. Install OpenCv
```
pip install opencv-python
```

5. Install PyTorch
https://pytorch.org/get-started/locally/

6. Install YOLOv8
https://docs.ultralytics.com/quickstart/
```
pip install ultralytics
```