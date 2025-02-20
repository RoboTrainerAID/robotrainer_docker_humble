## Polar OH1+

<img align="right" width="180" src="image/polar_oh1_sensor.webp">
The Polar OH1+ is an optical heart rate monitor that combines versatility, comfort and simplicity. It can be worn on the forearm or upper arm. It is mostly used for fitness objectives to read HR.


* Official website: [https://www.polar.com/us-en/sensors/oh1-optical-heart-rate-sensor](https://www.polar.com/us-en/sensors/oh1-optical-heart-rate-sensor)

## Requirments
1) Install Python Libraries: '''$ pip install pexpect bleak'''


## Node Informations
1) Node name: polar_oh1_node
2) Parameter:
* Device_Mac_Address : a string data type to connect own device via ble/; for example, _'A0:9E:1A:E0:BC:97'_

## Topic Information
### For raw data
1) _biosensors/polar_oh1/hr_ : 
  * type: standard_msg/Int32
  * detail: the Heart Rate (HR) signal. 

2) _biosensors/polar_oh1/battery_:
  * type: standard_msg/Int32
  * detail: the status of the battery level.


## Test the Node using Launch file

```bash
$ros2 launch polar_oh1 ros2-polar_oh1.launch.py
```

# Example of Published Topic Data
<p align="center">
<img src="/media/img/polar_h10_data.jpg" width="700" >
</p>
