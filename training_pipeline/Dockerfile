FROM tensorflow/tfx:1.13.0

WORKDIR /pipeline
COPY ./ ./

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
RUN sudo apt-get install git-lfs

RUN pip install --force-reinstall --no-deps attrs==22.1.0
RUN pip install -r requirements.txt

ENV PYTHONPATH="/pipeline:${PYTHONPATH}"
