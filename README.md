# szpet

#  Make sure to have ubuntu 20 lts

#  Install cuda
- curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
- sudo python3 install_gpu_driver.py

# install pip

- sudo apt update
- sudo apt-get install python3-pip

#  clone the repo
- git clone https://github.com/TalhaSheikh-dev/szpet.git
- gsutil cp gs://szmodels/email_classification_model/best_model.pt /home/talhasheikh/szpet/weights

# installing dependencies
- sudo su
- pip install -r requirements.txt


#  startup script
- #! /bin/bash
- python3 /home/talhasheikh/szpet/main.py

#  test the startup script
- sudo journalctl -u google-startup-scripts.service
- grep startup-scrip /var/log/syslog


# chmod 777 both env and file to run
