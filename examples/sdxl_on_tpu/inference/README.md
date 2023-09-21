## SDXL Inference on TPUv5e VM

How to deploy SD

### Creating a TPU VM

First, create a TPUv5e. Unlike other TPUs, the v5e requires creating a (queued resource)[https://cloud.google.com/tpu/docs/queued-resources].

You'll need a Google Cloud Platform account and TPUv5e quota.

```bash
export PROJECT_ID=<your-project-id>
export ZONE=us-east1-c # or your zone
export ACCELERATOR-TYPE=v5litepod-4
export RUNTIME_VERSION=v2-alpha-tpuv5-lite
export TPU_NAME=sdxl-inference
export QUEUED_RESOURCE_ID=sdxl-inference

gcloud alpha compute tpus queued-resources create ${QUEUED_RESOURCE_ID} --node-id ${TPU_NAME} --project ${PROJECT_ID} --zone ${ZONE} --accelerator-type ${ACCELERATOR_TYPE} --runtime-version ${RUNTIME_VERSION}
``` 

When you request queued resources, the request is added to a queue maintained by the Cloud TPU service. When it becomes available, it's assigned to your Google Cloud project for your immediate exclusive use. You can check the status of your queued resource.

```bash
gcloud alpha compute tpus queued-resources list
```

Once it is active, you can start using it.

```bash
NAME                                          ZONE        NODE_COUNT  ACCELERATOR_TYPE  TYPE  TOPOLOGY  STATE
sdxl-inference		                      us-east1-c  4           v5litepod-4                       ACTIVE
```

### Install dependencies

Before running the scripts, make sure to install the dependencies.

```bash
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME}  --project ${PROJECT_ID} --zone $ZONE
pip install diffusers transformers flax
pip install "jax[tpu]==0.4.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Deploy

The `app.py` file creates a gradio application on top of the vm.

```bash
python app.py
```

To run it in the background, you can use.

```bash
nohup python -u app.py
```

The url to the gradio space will be located in the nohup.out file

