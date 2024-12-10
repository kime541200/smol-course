### Docker

1. Build Docker Image
```bash
git clone https://github.com/<user_name>/smol-course.git
cd smol-course

sudo docker build -t smol-course --build-arg HF_TOKEN=<hf_token> -f ./0_start_with_docker/Dockerfile .
```

2. Start the container
```bash
export DATA_DIR=/data/smol-course-data  # choose a directory to keep data in this course

sudo docker run -it \
--init \
--rm \
--gpus '"device=0"' \
-v ${DATA_DIR}:/data \
-v ${PWD}:/home/workspace/smol-course \
-w /home/workspace/smol-course \
--entrypoint /bin/bash \
smol-course
``` 