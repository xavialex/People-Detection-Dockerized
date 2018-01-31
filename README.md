# People Detection Web Service in Docker

People recognition through HTTP video signal in docker.

## Description

This project makes use of the SSD Mobilenet model trained in COCO dataset to perform people detection from the video signal received by an HTTP signal. It then publishes a UI delivered by an HTTP Server that shows the amount of people detected in that frame and that frame with the detection boxes and the likelihoods.

## Usage

Build the image through Docker Compose:

```
cd project_route
docker-compose -d --build .
```

The image will tak 2.67 GB, so it may take several minutes to get ready.  

Prepare the signal-sender machine to deliver the video signal through HTTP (for example with VLC). Then change the required variables stored in the ```.env``` file, which are:
> * **VIDEO_IP**: HTTP address sended for it's processing.
> * **IMAGE_REFRESH**: Time interval (in seconds) taken by the algorithm to process the next frame. 0.5 seconds by default.
> * **DOCKER_PORT**: Internal docker port from where launch ```ui.html```. 8085 by default.
> * **UI_PORT**: Port used by the processing machine to deliver the web UI. 8085 by default.

Finally, type ```docker-compose up```, open a web browser ando go to ```YOUR_IP:UI_PORT``` to check the result. Press *Ctrl + C* to stop execution.
