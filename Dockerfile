FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip 
  


RUN pip install scikit-learn
RUN pip install opencv-contrib-python
RUN pip install tensorflow-gpu==2.0.0
RUN pip install numpy
RUN pip install absl-py
RUN pip install tqdm
RUN pip install Pillow
RUN pip install matplotlib
RUN pip install pandas
RUN pip install argparse
RUN pip install jupyterlab
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install libglib2.0-0 libglib2.0-bin libglib2.0-dev -y
