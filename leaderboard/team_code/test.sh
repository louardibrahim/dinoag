#!/bin/bash

# Set environment variables
# USER=root
PASSWORD=password1
DEBIAN_FRONTEND=noninteractive 
DEBCONF_NONINTERACTIVE_SEEN=true

# Update system and install required packages
apt-get update && \
    echo "tzdata tzdata/Areas select America" > /tmp/tz.txt && \
    echo "tzdata tzdata/Zones/America select New York" >> /tmp/tz.txt && \
    debconf-set-selections /tmp/tz.txt && \
    apt-get install -y abiword gnupg apt-transport-https wget software-properties-common ratpoison novnc websockify libxv1 libglu1-mesa xauth x11-utils xorg tightvncserver

# Download required deb files
wget https://svwh.dl.sourceforge.net/project/virtualgl/2.6.3/virtualgl_2.6.3_amd64.deb
wget https://iweb.dl.sourceforge.net/project/turbovnc/2.2.4/turbovnc_2.2.4_amd64.deb
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb

# Install Google Chrome and other required software
apt install -y ./google-chrome-stable_current_amd64.deb
dpkg -i virtualgl_*.deb
dpkg -i turbovnc_*.deb

# Setup VNC password
mkdir -p ~/.vnc
echo $PASSWORD | vncpasswd -f > ~/.vnc/passwd
chmod 0600 ~/.vnc/passwd

# Configure Ratpoison
echo "set border 1" > ~/.ratpoisonrc
echo "exec google-chrome --no-sandbox" >> ~/.ratpoisonrc

# Create a self-signed certificate for noVNC
openssl req -x509 -nodes -newkey rsa:2048 -keyout ~/novnc.pem -out ~/novnc.pem -days 3650 -subj "/C=US/ST=NY/L=NY/O=NY/OU=NY/CN=NY emailAddress=email@example.com"

# Expose port 80 and start TurboVNC server and websockify
/opt/TurboVNC/bin/vncserver
websockify -D --web=/usr/share/novnc/ --cert=~/novnc.pem 20000 localhost:5901

# Keep the script running
tail -f /dev/null
