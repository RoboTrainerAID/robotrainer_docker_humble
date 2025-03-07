## Polar OH1+

<img align="right" width="180" src="image/polar_oh1_sensor.webp">
The Polar OH1+ is an optical heart rate monitor that combines versatility, comfort and simplicity. It can be worn on the forearm or upper arm. It is mostly used for fitness objectives to read HR.

* Official website: [https://www.polar.com/us-en/sensors/oh1-optical-heart-rate-sensor](https://www.polar.com/us-en/sensors/oh1-optical-heart-rate-sensor)

## Requirements
1) Required Python Libraries: `$ pip install pexpect bleak`

## Node Informations
1) Node name: polar_oh1_node
2) Parameter Details:
* Device_Mac_Address : a string data type to connect own device via ble/; for example, _'A0:9E:1A:E0:BC:97'_

## Topic Information
### For raw data
1) _biosensors/polar_oh1/hr_ : 
  * type: standard_msg/Int32
  * description: the Heart Rate (HR) signal. 

2) _biosensors/polar_oh1/ppg_ch0_ :
  * type: standard_msg/Float32MultiArray
  * description: raw Photoplethysmography (PPG) data from channel 0.  

3) _biosensors/polar_oh1/ppg_ch1_ :
  * type: standard_msg/Float32MultiArray
  * description: raw Photoplethysmography (PPG) data from channel 1.  

4) _biosensors/polar_oh1/ppg_ch2_ :
  * type: standard_msg/Float32MultiArray
  * description: raw Photoplethysmography (PPG) data from channel 2.  

5) _biosensors/polar_oh1/ppg_ch3_ :
  * type: standard_msg/Float32MultiArray
  * description: raw Photoplethysmography (PPG) data from channel 3.  

6) _biosensors/polar_oh1/ppi_ :
  * type: standard_msg/Int32
  * description: the Peak-to-Peak Interval (PPI) signal.

7) _biosensors/polar_oh1/hrv_ :
  * type: standard_msg/Float32MultiArray
  * description: the Heart Rate Variability (HRV) signal.

## Test the Node using Launch file

```bash
ros2 launch polar_oh1 ros2-polar_oh1.launch.py
```
# Example of Published Topic Data
<p align="center">
<img src="/media/img/polar_h10_data.jpg" width="700" >
</p>
