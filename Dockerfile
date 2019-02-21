FROM python:3.7.2-stretch

WORKDIR /usr/src/app

COPY text_categorizer/ /usr/src/app/text_categorizer/

RUN python3 /usr/src/app/text_categorizer/setup.py install

CMD python3 text_categorizer ${EXCEL_FILE}
