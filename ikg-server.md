# Environment Installation

1. install nvidia driver 
```shell
sudo apt install nvidia-drivers-535 
```

2. install cuda 11.3
```shell
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run 
# omit the driver installation in this step 
sudo sh cuda_11.3.0_465.19.01_linux.run  
```

3. install anaconda
```shell
sudo mkdir /software 
# install anaconda dependencies
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 
curl -O https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
# change the installation location to /software/anaconda3 at this step
bash ~/Downloads/Anaconda3-2020.05-Linux-x86_64.sh
# init conda
bash /software/anaconda3/bin/conda init
```

4. create global environment variables and settings for all users.
See `cuda.sh` and `anaconda3.sh` in `/etc/profile.d`.
conda config file is located at `/etc/conda/.condarc`

5. create virtual environment with anaconda
In this step, a virtual environmnet `cosense3d` is created, packages are installed locally in the venv except the following dependencies are installed globally with apt.
```shell
sudo apt install build-essential python3-dev libopenblas-dev -y
```

6. install docker
```shell
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
# add user to docker group
sudo usermod -aG docker $USER
```

7. install portainer
```shell
# create the volume that Portainer Server will use to store its database
docker volume create portainer_data
# download and install the Portainer Server container
docker run -d -p 8000:8000 -p 9443:9443 --name portainer --restart=always \
-v /var/run/docker.sock:/var/run/docker.sock \
-v portainer_data:/data portainer/portainer-ce:latest

```

8. Problems:
- docker cannot pull images from docker hub: Error response from daemon: Get "https://registry-1.docker.io/v2/": dial tcp: lookup registry-1.docker.io on 127.0.0.53:53: read udp 127.0.0.1:34537->127.0.0.53:53: read: connection refused