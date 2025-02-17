# ## This ROS2 node for reading polar H10 data from bletooth protocols. This node is made based on this website; https://blog.alikhalil.tech/2014/11/polar-h7-bluetooth-le-heart-rate-sensor-on-ubuntu-14-04/
import asyncio
from bleak import BleakClient
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

class ros2_polar_h10(Node):
    def __init__(self):
        super().__init__('polar_h10_node')
        self.declare_parameter('Device_Mac_Address', 'A0:9E:1A:E0:BC:97')
        self.Parm_Device_Mac_Address = self.get_parameter('Device_Mac_Address').value
        self.sensor_status = 0
        self.Parm_Sensor_Enable = True  # Beispielwert, anpassen nach Bedarf

        self.get_logger().info("Start to connect to Polar H10")
        asyncio.run(self.connect_to_polar_h10())

    async def connect_to_polar_h10(self):
        polar_h10_state = False
        while not polar_h10_state:
            self.get_logger().info("Start of while loop")
            try:
                async with BleakClient(self.Parm_Device_Mac_Address) as client:
                    self.get_logger().info("[Polar] connected")
                    self.pub_polar_h10_hr = self.create_publisher(Int32, 'biosensors/polar_h10/hr', 10)
                    polar_h10_state = True
                    await client.start_notify("00002a37-0000-1000-8000-00805f9b34fb", self.notification_handler)
                    while rclpy.ok():
                        await asyncio.sleep(1)
                    await client.stop_notify("00002a37-0000-1000-8000-00805f9b34fb")
            except Exception as e:
                self.get_logger().info(f"[Polar] fail to connect: {e}")

    def notification_handler(self, sender, data):
        self.get_logger().info(f"Received data: {data}")

        # Beispiel: Herzfrequenz aus den Daten extrahieren und veröffentlichen
        if self.Parm_Sensor_Enable:
            hr = int(data[1])
            self.pub_polar_h10_hr.publish(Int32(data=hr))

def main(args=None):
    rclpy.init(args=args)
    node = ros2_polar_h10()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()