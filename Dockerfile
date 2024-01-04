FROM bentoml/model-server:0.11.0-py37
MAINTAINER ersilia

RUN pip install rdkit
RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch
RUN conda install -c dglteam dgl
RUN pip install pandas

WORKDIR /repo
COPY . /repo
