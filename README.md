# Handwriting Recognition

# Build the Dockerfile to create image
docker build -t jupyter/hwr .
# Run jupyter notebook in docker image
docker run --rm -it -p 8888:8888 -v "(VOLUME)/GitHub/handwritingrecog/Notebook":/home/jovyan/work jupyter/hwr
# Open Jupyter Notebook server
http://127.0.0.1:8888/?token=(TOKEN)
