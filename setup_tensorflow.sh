ssh-keygen
ssh-copy-id lanterns2.eecs.utk.edu
sudo chmod 777 /mydata

# install cuda toolkit 10.0
scp -r lanterns2.eecs.utk.edu:/local_scratch/CUDA /mydata
cd /mydata/CUDA
sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
sudo dpkg -i libcudnn7_7.4.1.5-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.4.1.5-1+cuda9.0_amd64.deb
# export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
cd ~/

rm ~/.bashrc
ln -s ~/Scripts/.bashrc ~/.bashrc
ln -s ~/Scripts/.bash_aliases ~/.bash_aliases
source ~/.bashrc

sudo apt-get update
sudo apt-get -y install python-dev python-pip

pip install --upgrade --user tensorflow-gpu
pip install --user pillow
pip install --user matplotlib

git clone https://github.com/tensorflow/models.git
mv ~/models ~/Tensorflow-models

mkdir /mydata/datasets
mkdir /mydata/datasets/google
mkdir /mydata/datasets/google/regular
scp -r com1504.eecs.utk.edu:~/TripletSent/datasets/google/regular/tfrecord /mydata/datasets/google/regular/
ln -s /mydata/datasets ~/TripletSent/datasets
