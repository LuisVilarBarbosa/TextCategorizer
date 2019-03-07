FROM continuumio/miniconda3:4.5.12

WORKDIR /usr/src/app

COPY text_categorizer/setup.py /usr/src/app/text_categorizer/setup.py

RUN conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

RUN python3 /usr/src/app/text_categorizer/setup.py install

CMD python3 /usr/src/app/text_categorizer/__main__.py
