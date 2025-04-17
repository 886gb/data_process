export DOCKER_IMAGE=aix-vm-runtime-241030:zhangtian
export CONTAINER_NAME="zt_datas"
# ------------------------------------------------------ 以下代码请勿改动 ------------------------------------------------------

docker run --name "$CONTAINER_NAME" \
           --detach \
           --ipc=host \
           --gpus all \
           --shm-size=1g \
           --volume "/nfs100/zhangtian/datas":/aix_datas \
           --volume "/models":/models \
           --workdir /aix_datas \
           --entrypoint "/bin/bash" \
           $DOCKER_IMAGE -c "tail -f /dev/null"

# ---------------------------------------------------------------------------------------------------------------------------
