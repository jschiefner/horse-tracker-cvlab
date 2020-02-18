# If using your own Docker image, use the following `FROM` command syntax substituting your image name
# FROM jupyter/minimal-notebook
# FROM ubuntu:18:04
# 
# USER root
# RUN apt-get update -y
# RUN apt-get upgrade -y
# # libopencv asks for location
# RUN apt-get install libopencv-dev -y
# RUN apt-get install python-opencv -y
# RUN apt-get install python3.6 -y
# 
# ADD https://github.com/krallin/tini/releases/download/v0.14.0/tini /tini
# RUN chmod +x /tini
# 
# # If using your own Docker image without jupyter installed:
# RUN pip3 install jupyter
# RUN pip3 install tensorflow==1.6.0
# RUN pip3 install keras==2.1.5
# RUN pip3 install numpy
# RUN pip3 install imutils
# RUN pip3 install opencv-python
# RUN pip3 install Pillow
# RUN pip3 install matplotlib
# RUN pip3 install h5py
# 

FROM final_base

RUN pip3 install imutils
RUN pip3 install cvutils
RUN pip3 install cvlib
RUN pip3 install opencv-contrib-python==4.0.0.21
RUN pip3 install progress
RUN pip3 install filterpy
ENV JUPYTER_TOKEN=d1aaa6e17d1498140034d348d85cafb964193a4fb806ab648f53f8401e6a62979896915d73afc8e2bfcd9b4e984243765341acef30de9e478b83c1a2895e6406

EXPOSE 8888
ENTRYPOINT ["/tini", "--"]
# --no-browser & --port aren't strictly necessary. presented here for clarity
CMD ["jupyter-notebook", "--no-browser", "--port=8888", "--ip=0.0.0.0", "--allow-root"]
# if running as root, you need to explicitly allow this:
# CMD ["jupyter-notebook", "--allow-root", "--no-browser", "--port=8888"]
WORKDIR /horse-tracker-cvlab


