cd ../src
#python demo.py ctdet --demo /userdata/liyj/data/test_data/pose/2022-09-09-15-41-06/image_rect_color --load_model ../models/ctdet_coco_dla_2x.pth
python demo.py multi_pose --demo /userdata/liyj/data/test_data/pose/2022-09-09-15-41-06/image_rect_color --load_model ../models/multi_pose_dla_3x.pth
