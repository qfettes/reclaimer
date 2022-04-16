FROM python:3.9.6

COPY . /main
WORKDIR /main
RUN python -m pip install pymongo
CMD ["main.py"]
ENTRYPOINT ["python3"]
