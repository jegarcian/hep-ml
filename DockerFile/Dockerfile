FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

RUN apt-get -y update

RUN apt-get install -y net-tools libsm6 libxext6 libxrender-dev git protobuf-compiler python-pil python-lxml python-tk wget nano

RUN pip3 install --upgrade pip

RUN pip3 install \
    	 contextlib2 \
	 Cython \
	 matplotlib \
	 lxml \
    	 Pillow \
	 scipy \
	 sklearn \
	 tensorflow-hub \
	 tensorflow-plot \
	 requests \
	 theano \
	 lasagne \
	 opencv-python \
	 opencv-contrib-python \
	 keras \
	 coremltools \
	 pandas \
	 keras2onnx \
	 keras-maskrcnn \
         git+https://github.com/fizyr/keras-retinanet.git

RUN pip3 install pycocotools

#WORKDIR /home/jgarcian/HEPML

#CMD ["/bin/bash"]
