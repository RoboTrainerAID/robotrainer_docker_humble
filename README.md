# RoboTrainer Docker Humble

ros2 environment for RoboTrainer development.

## Getting Started
### Build the Docker Container
1) Clone repository.
2) Navigate to the `robotrainer_docker_polar_oh1` directory
3) Build container:
```bash
./build_docker.sh
```
4) Run the container and start all necessary ROS 2 processes using autostart:
```bash
./autostart.sh
```
* This will automatically:

    1) Run the container:
        ```bash
        ./start_docker.sh
        ```
    2) Start the Polar OH1 measurement:
        ```bash
        ros2 launch polar_oh1 ros2-polar_oh1.launch.py
        ```

### Accessing and Visualizing Data
5) To visulize the data, open a new terminal and connect to running container.
```bash 
docker exec -it robotrainer_polar_oh1 /bin/bash
```
6) Run PlotJuggler:
```bash 
ros2 run plotjuggler plotjuggler
```
### Data Collection
Once `autostart.sh` is executed, the system will automatically start collecting the following data:
* **Heart Rate (HR)**
* **PPG values** (multiple channels)
* **PPI values**
* **Heart Rate Variability (HRV)** (RMSSD/SDNN)