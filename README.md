# TFT_AI_doodoongdeungjang-V1
![down](https://user-images.githubusercontent.com/55366212/92316965-23ef3180-f036-11ea-849d-ac8cd2b0efba.jpg)

The Trial of A.I. project in TFT season3.5.

This is an opensource project of TFT, which is an Auto Battler game made by RIOT.
Because there is no API of Riot, we need to make an image detection.
We use Yolov5 for image detection. we trained items, itemboxes, and other champions with Yolov5.

<div>
  <img src="https://user-images.githubusercontent.com/55366212/92317304-1f2c7c80-f03a-11ea-9b2a-70e43025ec3f.jpg">
  <img src="https://user-images.githubusercontent.com/55366212/92317333-6450ae80-f03a-11ea-9e89-9bef55b87fb2.jpg">
  <img src="https://user-images.githubusercontent.com/55366212/92317381-d32e0780-f03a-11ea-815d-448fa9fe225e.jpg">
 </div>

yolov5 link: https://github.com/ultralytics/yolov5
download.pt_files here: 

The Ideas of reinforcement_Learning

There is three_Agents.

1. Running_act:
   These agent carries out sushi, which selects a champion on the spinning disk.
   There are two variables we got, champion and item. 
   So, we can make a selection for sushi-lists.
   we can evaluate those objects for give them score.
2. 
