# Wingbeats Pi

<p align="center"> <img src="https://github.com/wingbeats/wingbeats_pi/blob/master/wingbeats.png"></p>

Raspberry Pi 3 bundle for Kaggle dataset [Wingbeats](https://www.kaggle.com/potamitis/wingbeats) 

We present a code bundle that allows wingbeat recordings to be transmitted through the Bluetooth wireless mode of a smartphone/tablet to a raspberry pi 3. The user sends the recording(s) and the result of the recognition taking place on pi3 appears on screen. We have embedded a CNN model on Python 2.7 and Python 3.5, Raspbian GNU/Linux 9 (stretch).

The weights are derived on a desktop computer using Keras 2.1.3 and after being converted to a Tensorflow 1.4 graph are ported on a pi3. The latter is only used to predict the correct class of the snippet.

The pi3 implementation suggests that a real-time wingbeat recognizer can be directly embedded to insect traps. Alternatively, a stand-alone pi3 can receive snippets through a Wifi functionality from a network of insect traps deployed anywhere in the world.

**_The following steps has been executed succesfully using android smartphones and iMac._**

## Step 1 - Bluetooth

First, we need to make Pi able to receive files over bluetooth.
```
sudo apt-get install obexpushd
```
```
sudo nano /etc/systemd/system/dbus-org.bluez.service
```
Add the ' -C' flag to the end of the 'ExecStart=' line. It should look like this:
```
...
...
ExecStart=/usr/lib/bluetooth/bluetoothd -C
...
...
```
Save the file and reboot. Then create the bluetooth storage folder.
```
sudo mkdir /bluetooth
```
To run and automate the obexpushd service on boot, create this file:
```
sudo nano /etc/systemd/system/obexpush.service
```
and add the following: 
```
[Unit]
Description=OBEX Push service
After=bluetooth.service
Requires=bluetooth.service

[Service]
ExecStart=/usr/bin/obexpushd -B -o /bluetooth -n

[Install]
WantedBy=multi-user.target
```
Save the file and set that to autostart with:
```
sudo systemctl enable obexpush
```
Reboot and press 'Make Discoverable' in bluetooth's setings to be able to find Pi and pair it with any device. If devices were already paired before the obexpushd service, re-pair them.

You should now be able to send files from your phone/tablet (or other device) to the Pi, they will appear in the /bluetooth directory. Notice that they will be owned by root, so you'll need sudo to access them.

## Step 2 - Libraries Installation

### *"python 2.7"* ###

run in terminal:
```
sudo apt-get install libblas-dev liblapack-dev python-dev libatlas-base-dev gfortran python-setuptools
sudo pip2 install http://ci.tensorflow.org/view/Nightly/job/nightly-pi/lastSuccessfulBuild/artifact/output-artifacts/tensorflow-1.5.0-cp27-none-any.whl
sudo apt-get install libffi-dev
sudo pip2 install pysoundfile
```
### *"python 3.5"* ###

run in terminal:
```
sudo apt-get install libblas-dev liblapack-dev python3-dev libatlas-base-dev gfortran python3-setuptools
curl -o /tmp/tensorflow-1.5.0-cp34-none-any.whl -O http://ci.tensorflow.org/view/Nightly/job/nightly-pi-python3/lastSuccessfulBuild/artifact/output-artifacts/tensorflow-1.5.0-cp34-none-any.whl
mv /tmp/tensorflow-1.5.0-cp34-none-any.whl /tmp/tensorflow-1.5.0-cp35-none-any.whl
sudo pip3 install /tmp/tensorflow-1.5.0-cp35-none-any.whl
sudo rm /tmp/tensorflow-1.5.0-cp35-none-any.whl
sudo apt-get install libffi-dev
sudo pip3 install pysoundfile
```
## Step 3 - Classifying

Download and extract the wingbeats_pi-master.zip and cd inside wingbeats_pi-master folder. Choose samples from test_corpus folder and copy them into your device. You should be able to run the app anytime in terminal, with the following commands:

### *"python 2.7"* ###
run in terminal:
```
python2 wingbeats.py
```
### *"python 3.5"* ###
run in terminal:
```
python3 wingbeats.py
```
