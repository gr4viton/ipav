import cv2
import win32com.client
import findHomeography as fh
import numpy as np

import os
import datetime as dt
import time


class Distinguish():
    captures = {}
    min_id, max_id = (0, 10)

    def open_all(self, try_ids):
        for id in try_ids:
            if id not in self.captures.keys():
                if self.min_id <= id <= self.max_id:
                    print('Trying to open: [{}]'.format(try_ids))
                    cap = cv2.VideoCapture()
                    cap.open(id)
                    if cap.isOpened():
                        print('Source[{}] Opened'.format(id))
                        self.captures[id] = cap
                        # opened_ids.append(id)
                    else:
                        print('Source[{}] Cannot open'.format(id))
                        cap.release()
            else:
                print('Source[{}] Already opened'.format(id))

    def image_all(self):
        whole = None
        for cap in self.captures:
            ret, im = cap.read()
            cv2.imshow('im', im)
            if whole is None:
                whole = im
            else:
                fh.joinIm([whole, im],vertically=0)
        cv2.imshow('whole image', whole)


    def close_all(self, only_this=[]):
        if only_this == []:
            only_this = self.captures.keys()
        for id in only_this:
            self.captures[id].release()
            print('Source[{}] Released'.format(id))

    def print_usb_info(self):
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


    def im_show(self, im):
        # txt = 'Source[{}] = {} = image'.format(self.id, self.names[self.id])
        txt = ''
        cv2.imshow(txt, im)
        return im


    def cap_show(self, cap):
        ret, im  = cap.read()
        self.im_show(im)
        return im

    def save_image(self, im):

        folder = r'D:\DEV\PYTHON\pyCV\calibration\_pics'

        name = self.names[self.id]
        def load_matrix(folder,file):
            path = os.path.join(folder,name, file)
        format_str = "%Y-%m-%d %H_%M_%S"
        i = dt.datetime.now().strftime(format_str)
        # path = os.path.abspath(''.join([r'D:\DEV\PYTHON\pyCV\calibration\_pics\\', ]))
        file = ''.join([str(i), ' image.png'])
        path = os.path.join(folder, name, file)
        print('path =', path)
        cv2.imwrite(path,im)

    def __init__(self):
        self.print_usb_info()

        self.try_ids = [0,1,2,3]
        self.try_ids = [0]
        self.open_all(self.try_ids)
        print(self.captures.keys())
        # image_all()
        self.id = self.try_ids[0]

        names = {}
        names[0] = 'round'
        names[1] = 'black'
        names[2] = 'gray'
        names[3] = 'clips'
        self.names = names
        self.undistort = False

        # captures = []
        # cams = [id for id in try_ids]
        mtx, dist = (None, None)
        while(True):
            # print(captures)
            self.cap = self.captures[self.id]
            cap = self.cap
            if cap is not None:
                if not self.undistort:
                    self.im = self.cap_show(cap)
                else:
                    self.im = self.cap_show(cap)
                    # dst = cv2.undistort(im, mtx, dist, None, newcameramtx)
                    # # crop the image
                    # x,y,w,h = roi
                    # dst = dst[y:y+h, x:x+w]

                    h,  w = self.im.shape[:2]
                    # undistort
                    mapx,mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
                    dst = cv2.remap(self.im, mapx, mapy, cv2.INTER_LINEAR)

                    # crop the image
                    x,y,w,h = roi
                    dst = dst[y:y+h, x:x+w]
                    # cv2.imwrite('calibresult.png',dst)
                    self.im = self.im_show(dst)
                    # print(self.im.shape)

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
                print(self.captures.keys(), key_int)
                if key_int in self.captures.keys():
                    print('Selecting source[{}] '.format(key_int))
                    self.id = key_int
                else:
                    self.open_all([key_int])
                    # if 0 < key_int and key_int < len(captures) :
                    if key_int in self.captures.keys():
                        cap = self.captures[key_int]
                        ret, im = cap.read()
                        print(im.shape)

                        if im is not None:
                            print('Choosing source[{}] '.format(key_int))
                            self.id = key_int
                        else:
                            print('Source[{}] cannot be read from'.format(key_int))
                            self.close_all([self.id])
                # if key in function_keys
                if key == ord('s'):
                    self.save_image(self.im)
                elif key == ord('d'):

                    def load_calib(name):
                        folder = r'D:\DEV\PYTHON\pyCV\calibration\_pics'


                        def load_matrix(folder,file):
                            path = os.path.join(folder,name, file)
                            mat = np.loadtxt(path)
                            return np.array(mat)

                        mtx = load_matrix(folder, 'Intrinsic.txt')
                        dist = load_matrix(folder, 'Distortion.txt')
                        print('Intrinsic: {}\nDistortion: {}'.format(mtx,dist))
                        return mtx, dist

                    mtx, dist = load_calib(names[self.id])
                    h,  w = self.im.shape[:2]
                    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
                    # undistort

                    dst = cv2.undistort(self.im, mtx, dist, None, newcameramtx)

                    # crop the image
                    x,y,w,h = roi
                    dst = dst[y:y+h, x:x+w]
                    # cv2.imwrite('calibresult.png',dst)
                    self.im = dst

                    self.undistort = True

                elif key == ord('u'):
                    self.undistort = False

                elif key == ord('a'):
                    for q in range(4,0,-1):
                        print('Taking 20 screenshots in {}'.format(q))
                        self.wait_show()
                    for q in range(20):
                        self.wait_show()
                        self.save_image(self.im)

        self.close_all()
        # while(True):
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

    def wait_show(self, interval=1, step=0.2):
        # step = 0.02
        # interval = 1
        for q in range(0, int(interval*1000), int(step*1000)):
            key = cv2.waitKey(1)
            self.im = self.cap_show(self.cap)
            time.sleep(step)


if __name__ == '__main__':
    d = Distinguish()

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