# CamCaliber
This repository contains TensorFlow tutorials in the field of deep learning network (CNN).
Dynamic Camera Caliber System

The system uses image processing to control traffic. Traffic density of lanes is calculated using image processing which is done using images of lanes that are captured using a camera and compared to reference images of lanes with no traffic. According to the traffic densities on all roads, our model will allocate intelligently the time period of green light for each road. We have chosen image processing for calculation of traffic density as cameras are readily available infrastructure on road intersections.
Requirements

    HQ Cam
    

Dependencies

    Python 3.5+
    OpenCV 3.2.0
    TensorFlow

Set up

    Log into your ThingSpeak account.
    Create a channel.
    Get the API write key for the channel.
    Paste the value in sample.py
    Add the analysis.m file to the channel.
    Once the script is running on the Pi (directions below) the data can be seen on the ThingSpeak dashboard.

Running the script

To run the script,

    Clone this file and .
    Run python3 .py
