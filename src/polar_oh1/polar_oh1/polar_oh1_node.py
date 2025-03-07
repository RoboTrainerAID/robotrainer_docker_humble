import asyncio
from bleak import BleakClient
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float32MultiArray
import struct
import math
import numpy as np

class PolarOH1Node(Node):

    CP_CHAR_UUID = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
    DATA_CHAR_UUID = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"
    POLAR_SERVICE_UUID = "fb005c80-02e7-f387-1cad-8acd2d8df0c8"
    HR_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

    def __init__(self):
        super().__init__('polar_oh1_node')

        # Get the MAC address of the Polar OH1+ device
        self.declare_parameter('Device_Mac_Address', 'A0:9E:1A:E0:BC:97')
        self.device_mac = self.get_parameter('Device_Mac_Address').value

        # Publishers
        self.pub_hr = self.create_publisher(Int32, 'biosensors/polar_oh1/hr', 10)
        self.pub_ppg_ch0 = self.create_publisher(Float32MultiArray, 'biosensors/polar_oh1/ppg_ch0', 10)
        self.pub_ppg_ch1 = self.create_publisher(Float32MultiArray, 'biosensors/polar_oh1/ppg_ch1', 10)
        self.pub_ppg_ch2 = self.create_publisher(Float32MultiArray, 'biosensors/polar_oh1/ppg_ch2', 10)
        self.pub_ppg_ch3 = self.create_publisher(Float32MultiArray, 'biosensors/polar_oh1/ppg_ch3', 10)
        self.pub_ppi = self.create_publisher(Int32, 'biosensors/polar_oh1/ppi', 10)
        self.pub_hrv = self.create_publisher(Float32MultiArray, 'biosensors/polar_oh1/hrv', 10)

        # PPI values for HRV analysis
        self.ppi_values = []

        # Start connection to Polar OH1+
        self.get_logger().info("🔄 Starting connection to Polar OH1+...")
        asyncio.ensure_future(self.connect_to_polar())

    async def connect_to_polar(self):
        while True:
            try:
                async with BleakClient(self.device_mac) as client:
                    self.get_logger().info("✅ [Polar OH1+] Connected!")
                    
                    # Subscribe to notifications
                    await client.start_notify(self.HR_CHAR_UUID, self.hr_handler)
                    await client.start_notify(self.DATA_CHAR_UUID, self.notification_handler)
                    await self.enable_measurements(client)
                    
                    # Keep the connection alive
                    while client.is_connected:
                        await asyncio.sleep(5)

            except Exception as e:
                self.get_logger().error(f"⚠️ Connection error: {e}, retrying in 5 seconds...")
                await asyncio.sleep(5) # Retry connection after 5 seconds

    async def enable_measurements(self, client):
        cmd_ppg = bytearray([0x02, 0x01, 0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x16, 0x00])
        cmd_ppi = bytearray([0x02, 0x03])
        await client.write_gatt_char(self.CP_CHAR_UUID, cmd_ppg)
        await asyncio.sleep(1)
        await client.write_gatt_char(self.CP_CHAR_UUID, cmd_ppi)

    def hr_handler(self, sender, data):
        heart_rate = int(data[1])
        if heart_rate > 0:  # Ignore 0 values
            self.pub_hr.publish(Int32(data=heart_rate))
            self.get_logger().info(f"❤️ Heart Rate: {heart_rate} BPM")

    def notification_handler(self, sender, data):
        type1, _, type2 = struct.unpack("<BqB", data[0:10])
        if type1 == 1:
            self.parse_ppg(data)
        elif type1 == 3:
            self.parse_ppi(data)

    def parse_ppg(self, data):
        def get_ppg_value(subdata):
            return struct.unpack("<i", subdata + (b'\0' if subdata[2] < 128 else b'\xff'))[0]

        numSamples = math.floor((len(data) - 10) / 12)
        ppg_values = {0: [], 1: [], 2: [], 3: []}

        for x in range(numSamples):
            for y in range(4):
                ppg_values[y].append(get_ppg_value(data[10 + x * 12 + y * 3:(10 + x * 12 + y * 3) + 3]))

        self.publish_ppg(ppg_values)
    
    def parse_ppi(self, data):
        numSamples = math.floor((len(data) - 10) / 6)
        for x in range(numSamples):
            start = 10 + x * 6
            sample = data[start:start + 6]
            hr, ppi, errEst, flags = struct.unpack("<BHHB", sample)
            self.publish_ppi(ppi)

    def publish_ppg(self, ppg_values):
        if len(ppg_values) >= 4:
            if self.context.ok():
                self.pub_ppg_ch0.publish(Float32MultiArray(data=ppg_values[0]))
                self.pub_ppg_ch1.publish(Float32MultiArray(data=ppg_values[1]))
                self.pub_ppg_ch2.publish(Float32MultiArray(data=ppg_values[2]))
                self.pub_ppg_ch3.publish(Float32MultiArray(data=ppg_values[3]))
                self.get_logger().info(f"📡 PPG Ch0: {ppg_values[0][0]} | Ch1: {ppg_values[1][0]} | Ch2: {ppg_values[2][0]} | Ch3: {ppg_values[3][0]}")

    def publish_ppi(self, ppi_value):
        if self.context.ok():
            self.ppi_values.append(ppi_value)
            
            # Begrenze auf 30 Werte für eine stabile HRV-Berechnung
            if len(self.ppi_values) > 30:
                self.ppi_values.pop(0)

            # HRV berechnen
            hrv_values = self.calculate_hrv()
            if hrv_values:
                hrv_data = Float32MultiArray(data=[hrv_values["RMSSD"], hrv_values["SDNN"]])
                self.pub_hrv.publish(hrv_data)
                self.get_logger().info(f"📊 HRV (RMSSD): {hrv_values['RMSSD']:.2f} ms | SDNN: {hrv_values['SDNN']:.2f} ms")

            # PPI-Wert publizieren
            self.pub_ppi.publish(Int32(data=ppi_value))
            self.get_logger().info(f"❤️ PPI: {ppi_value} ms")

    def calculate_hrv(self):
        if len(self.ppi_values) < 2:
            return None  # not enough data

        ppi_array = np.array(self.ppi_values)
        diffs = np.diff(ppi_array)  # diffences between successive PPI values

        # RMSSD: Root of the mean square difference value
        rmssd = np.sqrt(np.mean(diffs ** 2))

        # SDNN: Standard deviation of all PPI values
        sdnn = np.std(ppi_array)

        return {"RMSSD": rmssd, "SDNN": sdnn}

def main(args=None):
    rclpy.init(args=args)
    node = PolarOH1Node()
    try:
        asyncio.run(node.connect_to_polar())
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
