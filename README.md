# RoboTrainer Docker Humble

ros2 environment for RoboTrainer development.

## Getting Started
### Build the Docker Container
1) Clone the repository.
2) Navigate to the `robotrainer_docker_humble` directory.
3) Build the container:
```bash
./build_docker.sh
```
4) **Important:** The battery level will only be visible if you stop the Bluetooth daemon. Open an additional terminal and run:
```bash
sudo killall -9 bluetoothd
```
5) Run the container and start all necessary ROS 2 processes using the autostart script:
```bash
./autostart.sh
```
* This script will automatically:

    1) Start the Docker container:
        ```bash
        ./start_docker.sh
        ```
    2) Starts necessary system services within the container:
        ```bash
        /bin/bash -c 'sudo service dbus start && sudo bluetoothd
        ```
    3) Launch the Polar OH1 measurement process:
        ```bash
        ros2 launch polar_oh1 ros2-polar_oh1.launch.py
        ```

### Accessing and Visualizing Data
6) To visualize the data, open a new terminal and connect to the running container:
```bash 
docker exec -it robotrainer_humble /bin/bash
```
7) Run PlotJuggler to analyze the data:
```bash 
ros2 run plotjuggler plotjuggler
```
### Data Collection
Once `autostart.sh` is executed, the system will automatically start collecting the following data:
* **Heart Rate (HR)**
* **PPG values** (multiple channels)
* **PPI values**
* **Battery status**
* **Heart Rate Variability (HRV)** (RMSSD/SDNN)