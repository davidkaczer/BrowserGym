## Install demo-agent

### Set up WebArena containers

1. Check that you are in the `docker` group by running `groups` in the terminal. If not, ask the host server admin to add you to the docker group. 
1. Follow the [WebArena setup instructions](https://github.com/web-arena-x/webarena/tree/main/environment_docker#individual-website), using `wget` to download the docker `.tar` and `.zim` images. Depending on your network speed, this can take a few hours as the docker images are very large.
1. Check that all containers are running with `docker ps` and that they are mapping the required ports. In the `NETWORK` column, you should see something like `0.0.0.0:7780->80/tcp, :::7780->80/tcp` with a different port mapping for each container.
1. Check that all containers are serving the webpages by running `wget http://localhost:7780/admin` and so on using the URLs in the setup instructions
1. Run `docker inspect shopping_admin` and note down the `IPAddress`, which should start with 172.17. Repeat for each container.

### Set up BrowserGym container from image

1. Ask David for the image
1. `docker load --input <image_file>.tar`
1. Get the image name using `docker images`
1. `docker run -dit --name browsergym --gpus all <image_name>`


### Verify that BrowserGym container is set up correctly

1. `docker exec -it browsergym bash`. This will open a shell inside the container as root.
1. `nvidia-smi` should show the host's GPUs.
1. `wget 172.17.foo.bar` with the container IPs from above should return HTML.

### Set up VSCode to work inside the container

1. Install the "Dev Containers" extension
1. Open a VSCode SSH session on the server on which the containers are running
1. Once the container is running, `CTRL+SHIFT+P`, select `Attach to Running Container`, select container from dropdown menu


### Set up evaluation
1. In the container shell, `cd /home/ubuntu/BrowserGym/demo_agent`
1. `conda activate webagents`
1. Edit `set_env_vars.sh` to enter the correct IPs for the servers (starting with 172.17). The lines should look like `export WA_SHOPPING_ADMIN="http://172.17.0.6:80/admin"`. Also set `CUDA_VISIBLE_DEVICES` to the correct number of GPUs.
1. `. ./set_env_vars.sh` (the leading dot is important)
1. Edit `run.sh` to enter the desired task range and model checkpoint.
1. `./run.sh`
1. An output folder should be created in `demo_agent/results`. See `experiment.log` in this folder for further debugging.

You can run MiniWob++ tasks by setting e.g. `--task_name miniwob.click-test` in `run.sh`.

To quickly collect eval stats, create a new folder, move all relevant output folders into this folder, then run `./sum_results.sh ./results/my_experiment`.

### Removing a container
1. `docker ps -a`
1. `docker stop <container_name>` if container is running
1. `docker container rm <container_name>`

### Manual BrowserGym container setup

Schematic notes, not complete:

1. `docker run -dit --name browsergym --gpus all ubuntu`
1. `docker exec -it <image_name> bash`
1. `apt-get update`, `apt-get install iputils-ping wget curl python3 pip git gh`
1. Install miniconda, create python 3.10 env,  `pip install torch` (IIRC this installs the correct CUDA version automatically)
1. git clone MiniWob and **this fork of BrowserGym** into `/home/ubuntu`, install them and their dependencies using the repo instructions
1. `apt-get install alsa-tools` (needed for playwright to resolve chromium dependencies), `playwright install chromium`, `playwright install-deps chromium`
1. Set bash scripts in this repo to executable
