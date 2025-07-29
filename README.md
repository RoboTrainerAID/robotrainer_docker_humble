# RoboTrainer Bayesian Optimization

## How to start

```bash
./start_docker.sh
ros2 launch robotrainer_bayesian_optimization launch.py

# Or with autostart
./autostart.sh
```

Test the service with:

```bash
ros2 service call /robotrainer_bayesian_optimization/update std_srvs/srv/Trigger
```
