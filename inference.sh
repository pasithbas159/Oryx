export LOWRES_RESIZE=384x32
export VIDEO_RESIZE="0x64"
export HIGHRES_BASE="0x32"
export MAXRES=1536
export MINRES=0
export VIDEO_MAXRES=480
export VIDEO_MINRES=288

python inference_image.py --model-path /workspace/Oryx/models/THUdyh-Oryx-7B --image-file /workspace/Oryx/data/839_20180820_predict_SendRid_page_5_img_2_sub1.jpeg