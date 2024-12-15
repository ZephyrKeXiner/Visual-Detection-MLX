import numpy as np
from mlx.data.datasets import load_cifar10
from mlx.data.datasets import load_mnist


def get_cifar10(batch_size, root=None):
    tr = load_cifar10(root=root)

    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    def normalize(x):
        x = x.astype("float32") / 255.0
        return (x - mean) / std

    tr_iter = (
        tr.shuffle()
        .to_stream()
        .image_random_h_flip("image", prob=0.5)
        .pad("image", 0, 4, 4, 0.0)
        .pad("image", 1, 4, 4, 0.0)
        .image_random_crop("image", 32, 32)
        .key_transform("image", normalize)
        .batch(batch_size)
        .prefetch(4, 4)
    )

    test = load_cifar10(root=root, train=False)
    test_iter = test.to_stream().key_transform("image", normalize).batch(batch_size)

    return tr_iter, test_iter

def get_mnist(batchsize, root=None):
    tr = load_mnist(root=root, train=True)

    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
    
    def normalize(x):
        x = x.astype / 225.0
        return (x - mean) / std
    
    tr_iter = (
        tr.shuffle()
        .to_stream()
        .image_random_h_flip("image", prob=0.5)
        .pad("image", 0, 4, 4, 0.0)
        .pad("image", 1, 4, 4, 0.0)
        .image_random_crop("image", 32, 32)
        .key_transform("image", normalize)
        .batch(batch_size)
        .prefetch(4, 4)
    )

    test = load_mnist(root=root, train=False)
    test_iter = test.to_stream().key_transform("image", normalize).batch(batch_size)

    return tr_iter, test_iter

if __name__ == "__main__":
    batch_size = 64
    tr_iter, test_iter = get_mnist(batch_size)

    # 打印训练迭代器中的前几个批次
    print("Training Iterator:")
    for batch_counter, batch in enumerate(tr_iter):
        print(f"Batch {batch_counter}")
        print(f"Images shape: {batch['image'].shape}")
        print(f"Labels shape: {batch['label'].shape}")
        if batch_counter >= 2:  # 只打印前3个批次
            break

    # 打印测试迭代器中的前几个批次
    print("\nTest Iterator:")
    for batch_counter, batch in enumerate(test_iter):
        print(f"Batch {batch_counter}")
        print(f"Images shape: {batch['image'].shape}")
        print(f"Labels shape: {batch['label'].shape}")
        if batch_counter >= 2:  # 只打印前3个批次
            break
