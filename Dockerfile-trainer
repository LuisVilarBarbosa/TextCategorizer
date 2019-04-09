FROM continuumio/miniconda3:4.5.12

WORKDIR /usr/src/app

COPY text_categorizer/environment.yml /usr/src/app/text_categorizer/environment.yml

RUN conda env create --file /usr/src/app/text_categorizer/environment.yml

CMD /bin/bash -c "source activate text-categorizer && python3 /usr/src/app/text_categorizer $CONFIG_FILE"
