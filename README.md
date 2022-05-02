# PatrolBot
```diff
August 2021 - May 2022
Authors: Jesus Aguilera, Brandon Banuelos, Connor Callister, Max Orloff, Michael Stepzinski
Purpose: CS 426 Senior Project in Computer Science, Spring 2022, at UNR, CSE Department
```
The Patrol Bot is a robot which utilizes machine learning to help detect bike theft.
The robot is meant to survey UNR’s campus and provide real time alerts to campus police regarding potential threats.
The robot accepts manual controls from users to facilitate campus traversal.
Information such as camera feed from the robot, current position, log data, and recordings are readily available on the system’s user interface so officers can quickly understand all of the data processed.
Alert classification is done with the use of an object detection model which classifies people, bicycles, bolt cutters, and angle grinders.
With the use of the object detection model, the system is able to make predictions regarding the threat of a given scenario, and these alerts are presented on the system’s user interface.
All of the data from the logs can be saved into csv files with the press of a button.
The settings page allows users to enable the display of bounding boxes for key items, toggle items to be detected, toggle the machine learning models, and choose between a light or dark mode theme.

Our system will help lower the strain on campus police by helping detect and stop the most prevalent crime on campus: bike theft.
Fewer officers would need to be around campus bike racks throughout the day, and more students would be able to fully immerse themselves in their studies without having to worry about their ride home.
