## build

To build the image, in linux :

```bash
git clone https://github.com/mariusgarenaux/fine_tuning_acronym
cd fine_tuning_acronym
git checkout formation_continue
docker build -t tp_fine_tuning .
```

## run it locally

If you built it locally :

```bash
docker run -d --name tp-fine-tuning --runtime=nvidia --gpus all -p 8888:8888 tp_fine_tuning
```

> this should expose a jupyter lab locally, at : 127.0.0.1:8888/notebook/?token=token

> for CPU only, you can get rid of --runtime and --gpus params

To use image from docker hub :

```bash
docker run -d --name tp-fine-tuning --runtime=nvidia --gpus all -p 8888:8888 mgg3/tp_fine_tuning:0.1
```

> this should expose a jupyter lab locally, at : 127.0.0.1:8888/notebook/?token=token

## run it in datalab.univ-rennes.fr

In services, remove the text, and replace it by :

```bash
sudo docker run -d --name tp-fine-tuning --privileged --runtime=nvidia --gpus all -v /myhomedir:/root/myhomedir -w /root/myhomedir -e HOME=/root --network host -e PASSWORD=$USER_PASSWORD mgg3/tp_fine_tuning:0.1
```

> this will fetch from docker hub the image
