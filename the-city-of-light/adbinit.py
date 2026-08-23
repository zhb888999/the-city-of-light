import subprocess
import time

def check_device(ip, port):
    result = subprocess.run(['adb', 'devices'], capture_output=True)
    for line in result.stdout.decode('utf-8').splitlines():
        if f"{ip}:{port}" in line:
            return True
    return False

def connect_device(ip, port):
    result = subprocess.run(['adb', 'connect', f'{ip}:port'], capture_output=True)
    if "connected" in result.stdout.decode('utf-8'):
        return True

def init_adb(ip, port):
    if check_device(ip, port):
        return True
    if connect_device(ip, port):
        return True
    if check_device(ip, port):
        return True
    for i in range(20):
        print("wait for connect!")
        time.sleep(0.5)
        if check_device(ip, port):
            return True
    return False

def enable_tcpip(port):
    subprocess.run(['adb', 'tcpip', f'{port}'])

def restart_server():
    subprocess.run(['adb', 'kill-server'])
    subprocess.run(['adb', 'start-server'])
