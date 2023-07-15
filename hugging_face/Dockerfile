FROM zironycho/pytorch:1120-cpu-py38

# Install libraries
COPY ./hugging_face/requirements.txt ./
RUN pip install -r requirements.txt && \
    rm ./requirements.txt

# Setup container directories
USER root
RUN mkdir -p /bae
RUN chown root /bae

# Copy local code to the container
COPY output/inception_finetuned/epoch=22-step=897.ckpt /bae/output/inception_finetuned/

# launch server with gunicorn
WORKDIR /bae
