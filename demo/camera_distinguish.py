import cv2
import usb.core

try_ids = [0,1,2,3]
caps = []
ids = []

def open_all():
    for id in try_ids :
        cap = cv2.VideoCapture()
        cap.open(id)
        if cap.isOpened():
            print('Source[{}] Opened'.format(id))
            caps.append(cap)
            ids.append(id)
        else:
            print('Source[{}] Cannot open'.format(id))
            cap.release()





def close_all():
    for (cap, id) in zip(caps, ids):
        cap.release()
        print('Source[{}] Released'.format(id))


if __name__ == '__main__':

    # all_devs = list(usb.core.find(find_all=True))
    # # print(all_devs)
    # [print(dev) for dev in all_devs]

    usb.core.show_devices(verbose=True)

    # find our device
    # dev = usb.core.find(idVendor=0xfffe, idProduct=0x0001)
    #
    # # was it found?
    # if dev is None:
    #     raise ValueError('Device not found')

    import win32com.client

    wmi = win32com.client.GetObject("winmgmts:")
    print('_'*42,'device ids:')

    hub = 'USB\ROOT_HUB'
    viahub = 'USB\VIA_ROOT_HUB'
    keyb = r'USB\VID_04D9&PID_1702\5&270230AB&0&2'
    virtual_pointer = r'VUSB\VID_2109&PID_3431\6&2DF641F&0&1' # aka general mouse
    prefixes = (hub, viahub, keyb, virtual_pointer)

    ports = {}
    ports[r'\5&104B4E52&0&2'] = 'B1'
    ports[r'\5&104B4E52&0&1'] = 'B2'
    ports[r'\5&1E58346&0&4'] = 'B3'
    ports[r'\5&1E58346&0&3'] = 'B4'
    ports[r'\7&250AC7DE&0&1'] = 'B5'
    ports[r'\7&250AC7DE&0&2'] = 'B6'

    ports[r'\5&104B4E52&0&5'] = 'F1'
    ports[r'\5&106F75EA&0&1'] = 'F2'

    cam = r'USB\VID_1908&PID_2311'
    cams = [cam]
    lencam = len(cam)

    for usb in wmi.InstancesOf("Win32_USBHub"):
        # print(usb.DeviceID)
        id = str(usb.DeviceID)

        # if any(ext in url_string for ext in extensionsToCheck):
        if not id.startswith(prefixes):
            if id.startswith(tuple(cams)):
                port = id[lencam:]
                if port in ports.keys():

                    print('Cam found on {} '.format(ports[port]))

                else:
                    print('Cam found elsewhere = {} '.format(port)  )
            else:
                print('DeviceID: ' + id)

    # print('_'*42,'manufacturer:')
    # for usb in wmi.InstancesOf ("Win32_UsbController"):
    #
    #     print('Manufacturer: ' + str(usb.Manufacturer))


ports = {}
port = 'USB\ROOT_HUB\4&DAD28DC&0'



black_data = '''
bcdUSB:             0x0200
bDeviceClass:         0xEF
bDeviceSubClass:      0x02
bDeviceProtocol:      0x01
bMaxPacketSize0:      0x40 (64)
idVendor:           0x1908
idProduct:          0x2311
bcdDevice:          0x0100
iManufacturer:        0x01
iProduct:             0x02
iSerialNumber:        0x00
bNumConfigurations:   0x01
'''
clips_data = '''
bcdUSB:             0x0200
bDeviceClass:         0xEF
bDeviceSubClass:      0x02
bDeviceProtocol:      0x01
bMaxPacketSize0:      0x40 (64)
idVendor:           0x1908
idProduct:          0x2311
bcdDevice:          0x0100
iManufacturer:        0x01
iProduct:             0x02
iSerialNumber:        0x00
bNumConfigurations:   0x01
'''

gray_data = '''
bcdUSB:             0x0200
bDeviceClass:         0xEF
bDeviceSubClass:      0x02
bDeviceProtocol:      0x01
bMaxPacketSize0:      0x40 (64)
idVendor:           0x1908
idProduct:          0x2311
bcdDevice:          0x0100
iManufacturer:        0x01
iProduct:             0x02
iSerialNumber:        0x00
bNumConfigurations:   0x01
'''

round_data = '''
bcdUSB:             0x0200
bDeviceClass:         0xEF
bDeviceSubClass:      0x02
bDeviceProtocol:      0x01
bMaxPacketSize0:      0x40 (64)
idVendor:           0x046D (Logitech Inc.)
idProduct:          0x0992
bcdDevice:          0x0005
iManufacturer:        0x00
iProduct:             0x00
iSerialNumber:        0x02
bNumConfigurations:   0x01
'''

blue_data = '''
bcdUSB:             0x0200
bDeviceClass:         0xEF
bDeviceSubClass:      0x02
bDeviceProtocol:      0x01
bMaxPacketSize0:      0x40 (64)
idVendor:           0x199E
idProduct:          0x8101
bcdDevice:          0x0100
iManufacturer:        0x01
iProduct:             0x02
iSerialNumber:        0x03
bNumConfigurations:   0x01
'''