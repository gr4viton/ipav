import cv2
import win32com.client
import findHomeography as fh

import os
import datetime as dt
import time

captures = []
opened_ids = []

def open_all(try_ids):
    for id in try_ids:
        if id not in opened_ids:
            cap = cv2.VideoCapture()
            cap.open(id)
            if cap.isOpened():
                print('Source[{}] Opened'.format(id))
                captures.append(cap)
                opened_ids.append(id)
            else:
                print('Source[{}] Cannot open'.format(id))
                cap.release()
        else:
            print('Source[{}] Already opened'.format(id))

def image_all():
    whole = None
    for cap in captures:
        ret, im = cap.read()
        cv2.imshow('im', im)
        if whole is None:
            whole = im
        else:
            fh.joinIm([whole, im],vertically=0)
    cv2.imshow('whole image', whole)


def close_all(only_this=[]):
    if only_this == []:
        only_this = opened_ids
    for (cap, id) in zip(captures, only_this):
        cap.release()
        print('Source[{}] Released'.format(id))

def print_usb_info():
    hub = 'USB\ROOT_HUB'
    viahub = 'USB\VIA_ROOT_HUB'
    keyb = r'USB\VID_04D9&PID_1702\5&270230AB&0&2'
    virtual_pointer = r'VUSB\VID_2109&PID_3431\6&2DF641F&0&1' # aka general mouse
    prefixes = (hub, viahub, keyb, virtual_pointer)

    ports = {}
    # generic cam ports ==
    ports[r'\5&104B4E52&0&2'] = 'B1 generic'
    ports[r'\5&104B4E52&0&1'] = 'B2 generic'
    ports[r'\5&1E58346&0&4'] = 'B3 generic'
    ports[r'\5&1E58346&0&3'] = 'B4 generic'
    ports[r'\7&250AC7DE&0&1'] = 'B5 generic'
    ports[r'\7&250AC7DE&0&2'] = 'B6 generic'

    ports[r'\5&104B4E52&0&5'] = 'F1 generic'
    ports[r'\5&106F75EA&0&1'] = 'F2 generic'
    # round cam ports ==
    ports[r'\B91FC2E2'] = 'F1 round'

    cam = r'USB\VID_1908&PID_2311' # generic camera = clips, black, gray
    round = r'USB\VID_046D&PID_0992'
    cams = [cam, round]
    lencam = len(cam)



    wmi = win32com.client.GetObject("winmgmts:")
    print('_'*42,'device opened_ids:')

    for usb in wmi.InstancesOf("Win32_USBHub"):
        # print(usb.DeviceID)
        id = str(usb.DeviceID)
        # print(type(usb))
        # product = str(usb.DeviceID)
        # usb.
        # print('DeviceID: {}  Product: {}'.format(id,product) )
        # if any(ext in url_string for ext in extensionsToCheck):
        if not id.startswith(prefixes):
            if id.startswith(tuple(cams)):
                port = id[lencam:]
                if port in ports.keys():

                    print('>>> {} '.format(ports[port]))

                else:
                    print('>>> Cam found elsewhere = {} '.format(port)  )
            else:
                print('>>> DeviceID: ' + id)

    # print('_'*42,'manufacturer:')
    # for usb in wmi.InstancesOf ("Win32_UsbController"):
    #
    #     print('Manufacturer: ' + str(usb.Manufacturer))

def cap_show(cap):
    ret, im = cap.read()

    txt = 'Source[{}] image'.format(id)
    cv2.imshow(txt, im)
    return im

def save_image(im):

    format_str = "%Y-%m-%d %H_%M_%S"
    i = dt.datetime.now().strftime(format_str)
    path = os.path.abspath(''.join([r'D:\DEV\PYTHON\pyCV\calibration\_pics\\', str(i), ' image.png']))
    print('path =', path)
    cv2.imwrite(path,im)

if __name__ == '__main__':
    print_usb_info()

    try_ids = [0,1,2,3]
    try_ids = [1]
    open_all(try_ids)
    # image_all()
    id = try_ids[0]
    # captures = []
    # cams = [id for id in try_ids]

    while(True):
        # print(captures)
        index = opened_ids.index(id)
        cap = captures[index]
        if cap is not None:
            im = cap_show(cap)
        key = cv2.waitKey(1) & 0xFF

        if int(key) == 255:
            # print(key)
            continue
        if key == ord('q'):
            break
        elif key == ord('b'):
            print('it')
        else:
            key_int = int(key) - 48
            print(opened_ids, key_int)
            if key_int in opened_ids:
                print('Selecting source[{}] '.format(key_int))
                id = key_int
            else:
                open_all([key_int])
                # if 0 < key_int and key_int < len(captures) :
                if key_int in captures:
                    cap = captures[captures.index(key_int)]
                    ret, im = cap.read()
                    print(im.shape)
                    close_all([cap])
                    if im is not None:
                        print('Choosing source[{}] '.format(key_int))
                        id = key_int
                    else:
                        print('Source[{}] not opened'.format(key_int))
            # if key in function_keys
            if key == ord('s'):
                save_image(im)
            if key == ord('a'):
                for q in range(4,0,-1):
                    print('Taking 20 screenshots in {}'.format(q))
                    time.sleep(1)
                for q in range(20):
                    key = cv2.waitKey(1)
                    im = cap_show(cap)
                    save_image(im)
                    time.sleep(1)

    close_all()
    # while(True):
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
round_ports = '''
USB\VID_046D&PID_0992\B91FC2E2 = F1
USB\VID_046D&PID_0992\B91FC2E2 = F2

on B5:
DeviceID: USB\VIA_ROOT_HUB\5&2564895&0
DeviceID: USB\VID_046D&PID_0992\B91FC2E2

on B6:
DeviceID: USB\VIA_ROOT_HUB\5&2564895&0
DeviceID: USB\VID_046D&PID_0992\B91FC2E2
'''

order = '''
>>> B5 generic
>>> B6 generic
>>> F1 generic

>>> B4 generic
>>> B5 generic
>>> B2 generic

>>> B4 generic
>>> B1 generic
>>> B3 generic


____________________________________________________
>>> B1 generic
>>> B3 generic
>>> B2 generic

>>> B4 generic
>>> B3 generic
>>> B2 generic

>>> B4 generic
>>> B1 generic
>>> B2 generic
____________________________________________________
>>> B4 generic
>>> B3 generic
>>> B6 generic

>>> B3 generic
>>> B5 generic
>>> B6 generic

>>> B3 generic
>>> B5 generic
>>> B2 generic

>>> B5 generic
>>> B6 generic
>>> B2 generic
____________________________________________________
>>> F2 generic
>>> B4 generic
>>> B2 generic

>>> B4 generic
>>> B2 generic
>>> F1 generic

>>> B1 generic
>>> B3 generic
>>> F1 generic

>>> F2 generic
>>> B1 generic
>>> B3 generic

>>> B5 generic
>>> B6 generic
>>> F1 generic
____________________________________________________

THIS IS IT!
>>> F2 generic
>>> B4 generic
>>> B1 generic
>>> B3 generic
>>> B5 generic
>>> B6 generic
>>> B2 generic
>>> F1 generic
'''

correlation_CVindex_portOrder = '''
Black Gray Clips  Round
Top Bottom Left
A = F1
B = F2
1 = B1 ...
____________________________________________________
413 = CGB
134 = GBC
012 = BCG


413 = GCB
134 = CBG
012 = BGC

reconnect B
413 = GCB
134 = CBG
012 = BGC

not closed
reconnect G
413 = GCB
134 = CBG
012 = BGC

not closed
reconnect C
413 = GCB
134 = CBG
012 = BGC

closed
reconnect C
413 = GCB
134 = CBG
012 = BGC


USE THIS
B413 = RGCB
134 = CBG
0123 = RBGC
'''

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