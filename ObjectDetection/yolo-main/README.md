

# 安装环境
pip install -r requirements.txt

# 压缩
tar -czvf xxx.tar.gz source_file

# 解压缩
tar -xzvf yoloair-main.tar.gz -C /root/autodl-tmp

# 删除文件
rm -rf fileName

# 安装虚拟仓库
conda create -n yoloair python=3.8 pytorch==1.7.0 cudatoolkit=11 torchvision -c pytorch -y # cuda 10.2 的版本过低？这个是虚拟驱动的版本吗？



https://docs.qq.com/sheet/DT3RWbkhHaUJmWXBQ?tab=BB08J2?scene=7f7046ec46dbfa788dce174eBNOXo1



// 训练方式

python train.py --data coco128.yaml --cfg configs/yolov5/yolov5s.yaml --weight yolov5s.pt

# demo1 0.92
python train.py --data pcb_xl.yaml --cfg configs/test/yolov5s6.yaml --weight yolov5s6.pt --cache
python train.py --data pcb_xl.yaml --cfg configs/test/yolov5s6.yaml --weight "" --cache --epochs 300
# demo2 v7不包含pt 0.27
python train.py --data pcb_xl.yaml --cfg configs/yolov7/train/yolov7.yaml --cache
# 效果不好
python train.py --data pcb_xl.yaml --cfg configs/yolov7/train/yolov7_tiny --weight yolov7_tiny.pt --cache
# 效果一般， 增加注意力机制SE，相比原始提升0.04个点
python train.py --data pcb_xl.yaml --cfg configs/yolov5/yolov5s-SE.yaml --cache
# 注意力机制CA 提升0.21个点 
python train.py --data pcb_xl.yaml --cfg configs/attention/yolov5s_CA.yaml --cache
# 0.67 exp23
python train.py --data pcb_xl.yaml --cfg configs/attention/yolov5s_GAMAttention.yaml --cache

# 更换骨干网络 ghost只有0.41
python train.py --data pcb_xl.yaml --cfg configs/yolov5/yolov5s-ghost.yaml --cache




# 尝试两种不同的改进方式
# 1、轻量化slim 可以用预训练权重，因为只修改了head部分。neck和head；保证掉点不会很严重
# 2、如果不行的话增加轮次，都不带pt。先进backbone; convnext_base/swin/ 


# 首先轻量化neck减少参数量； 使用heavy argumentation让参数量减少的同时效果不会很下降；

python train.py --data pcb_xl.yaml --cfg configs/yolov5-Improved/slimneck/slimneck-yolov5s.yaml --cache 

# slimeneck+suffle
python train.py --data pcb_xl.yaml --cfg configs/test/slimneck-yolov5s-Shuffle.yaml --cache --epochs 100 --weight  runs/train/exp54/weights/best.pt

# + shuffle
python train.py --data pcb_xl.yaml --cfg configs/test/yolov5s6-Shuffle.yaml --cache --epochs 300 --weight ""
python train.py --data pcb_xl.yaml --cfg configs/test/yolov5s6-Shuffle.yaml --cache --epochs 100 --weight  runs/train/exp57/weights/best.pt --evolve

# + slimeneck




# 比对不同的基本模型
# yolov5n
python train.py --cfg configs/yolov5/yolov5n6.yaml --weight yolov5n.pt --cache --epochs 100

# yolov5m
python train.py --cfg configs/yolov5/yolov5m6.yaml --weight yolov5m.pt --cache --epochs 100

# yolov5s+asff 直接失败




# 调整超参数
python train.py --cfg configs/yolov5-Improved/slimneck/slimneck-yolov5s.yaml --weight yolov5s6.pt --cache --hyp data/hyps/hyp.scratch-high.yaml

python train.py --cfg configs/yolov5-Improved/slimneck/slimneck-yolov5s.yaml --weight yolov5s6.pt --cache --hyp data/hyps/hyp.scratch-med.yaml

python train.py --cfg configs/yolov5-Improved/slimneck/slimneck-yolov5s.yaml --weight yolov5s6.pt --cache --hyp data/hyps/hyp.scratch-low.yaml




# 超参数优化,基本上无法提升，还有掉点的情况
python train.py --cfg configs/yolov5-Improved/slimneck/slimneck-yolov5s.yaml --weight runs/train/exp31/weights/best.pt --cache --hyp data/hyps/hyp.scratch-low.yaml --epochs 100

python train.py --cfg configs/yolov5-Improved/slimneck/slimneck-yolov5s.yaml --weight runs/train/exp31/weights/best.pt --cache --hyp data/hyps/hyp.scratch-low.yaml --epochs 100 --evolve

python train.py --cfg configs/yolov5-Improved/slimneck/slimneck-yolov5s.yaml --weight runs/train/exp31/weights/best.pt --cache --hyp data/hyps/hyp.scratch-med.yaml --epochs 50 --evolve



# 检测方式

python detect.py  --weights ..\runs\train\exp34\weights\best.pt --source ..\01_short_02.jpg



# acc
python train.py --cache


conda create -n yoloain python=3.8 pytorch==1.7.0 cudatoolkit=10.2 torchvision -c pytorch -y