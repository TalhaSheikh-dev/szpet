# szpet


#  Make sure to have ubuntu 20 lts

#  Install cuda
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py

# install pip

sudo apt update
sudo apt python3-pip

#  clone the repo
git clone https://github.com/TalhaSheikh-dev/szpet.git

#  startup script
#! /bin/bash
pip install -r /home/talhasheikh/szpet/requirements.txt
python3 /home/talhasheikh/szpet/main.py

#  test the startup script
sudo journalctl -u google-startup-scripts.service -f
