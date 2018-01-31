FROM continuumio/miniconda3

COPY ./content /app
COPY ./entry_point.sh /entry_point.sh

RUN conda env create -f /app/object_detection_environment.yml
RUN chmod +x /entry_point.sh

ENTRYPOINT ["/entry_point.sh"]
