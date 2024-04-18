# python train_mnist.py --batch-size 64 --dataset texture --lr 0.001 --epochs 20 --img-size 224
# python train_mnist.py --batch-size 64 --dataset texture --lr 0.01 --epochs 20 --img-size 224
python train_mnist.py --batch-size 128 --dataset mnist --lr 0.0001 --epochs 20 --img-size 64
python train_mnist.py --batch-size 128 --dataset mnist --lr 0.001 --epochs 20 --img-size 64
