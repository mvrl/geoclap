FROM daskdev/dask

RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        mercurial \
        subversion \
        wget \
        g++-11 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 

ENV PATH /opt/conda/bin:$PATH

RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate geoclap" >> ~/.bashrc && \
    /opt/conda/bin/conda clean -afy

RUN echo $(pwd)

COPY environment.yml /geoclap/

RUN cd /geoclap && conda env create -f environment.yml;

RUN chmod -R 777 /geoclap

RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

RUN pip install --force-reinstall charset-normalizer==3.1.0

SHELL ["/bin/bash", "--login", "-c"]                             
CMD [ "/bin/bash" ]