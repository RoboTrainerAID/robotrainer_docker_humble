import asyncio
from bleak import BleakClient
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float32MultiArray
 
class PolarOH1Node(Node):

    def __init__(self):
        super().__init__('polar_oh1_node')

        # Parameters
        self.declare_parameter('Device_Mac_Address', 'A0:9E:1A:E0:BC:97')
        self.device_mac = self.get_parameter('Device_Mac_Address').value
 
        # Publishers
        self.pub_hr = self.create_publisher(Int32, 'biosensors/polar_oh1/hr', 10)
        self.pub_ppg = self.create_publisher(Float32MultiArray, 'biosensors/polar_oh1/ppg', 10)
        self.pub_battery = self.create_publisher(Int32, 'biosensors/polar_oh1/battery', 10)
        self.get_logger().info("Starting connection to Polar OH1+...")

        asyncio.run(self.connect_to_polar())
 
    async def connect_to_polar(self):
        while True:
            try:
                async with BleakClient(self.device_mac) as client:
                    self.get_logger().info("[Polar OH1+] Connected")

                    # Enable PPG streaming (send write command)
                    await client.write_gatt_char("fb005c81-02e7-f387-1cad-8acd2d8df0c8", bytearray([0x01]))
                    self.get_logger().info("PPG Streaming Enabled")
 
 
                    # Subscribe to notifications
                    await client.start_notify("00002a37-0000-1000-8000-00805f9b34fb", self.hr_handler)  # Heart Rate
                    await client.start_notify("fb005c82-02e7-f387-1cad-8acd2d8df0c8", self.ppg_handler)  # PPG
                
 
                    # Read battery level every 10 seconds
                    while rclpy.ok():
                        battery_level = await client.read_gatt_char("00002a19-0000-1000-8000-00805f9b34fb")
                        battery_percentage = int(battery_level[0])
                        self.pub_battery.publish(Int32(data=battery_percentage))
                        self.get_logger().info(f"Battery Level: {battery_percentage}%")
                        await asyncio.sleep(10)  # Check battery every 10s

                    # Stop notifications before disconnecting
                    await client.stop_notify("00002a37-0000-1000-8000-00805f9b34fb")
                    await client.stop_notify("fb005c82-02e7-f387-1cad-8acd2d8df0c8")
                 
 
            except Exception as e:
                self.get_logger().error(f"Connection failed: {e}")
                await asyncio.sleep(5)  # Retry connection after 5 seconds
 
    def hr_handler(self, sender, data):
        """ Handle Heart Rate notifications """
        heart_rate = int(data[1])
        self.get_logger().error(f"Heart Rate data: {data}")
        self.pub_hr.publish(Int32(data=heart_rate))
        self.get_logger().info(f"Heart Rate: {heart_rate} BPM")
 
    def ppg_handler(self, sender, data):
        """ Handle PPG Raw Data notifications """
        ppg_values = [int(byte) for byte in data]  # Convert bytes to integers
        msg = Float32MultiArray(data=ppg_values)
        self.pub_ppg.publish(msg)
        self.get_logger().info(f"PPG Data: {ppg_values}")
 
 
def main(args=None):
    rclpy.init(args=args)
    node = PolarOH1Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
 
if __name__ == '__main__':

    main()
