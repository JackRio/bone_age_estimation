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
COPY lightning_logs/finetune_512_fold1_10_noaug/checkpoints/epoch=42-step=1118.ckpt /bae/output/inception_finetuned_1/
COPY lightning_logs/finetune_512_fold2_10_noaug/checkpoints/epoch=41-step=1092.ckpt /bae/output/inception_finetuned_2/
COPY lightning_logs/46tbfh60/checkpoints/epoch=24-step=650.ckpt /bae/output/inception_finetuned_3/

# launch server with gunicorn
WORKDIR /bae
