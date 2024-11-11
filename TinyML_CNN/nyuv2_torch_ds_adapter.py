from functools import partial

import albumentations as A
import cv2
import numpy as np
import tensorflow as tf
import os
import csv

from config import cfg
from tensorflow_model_optimization.python.core.keras.compat import keras

class NYUV2DataSet: 
    def __init__(self, dataset_path, csv_name, crop_size, scale_size=None):
        self.dataset_path = dataset_path
        self.input_images = []
        self.depth_images = []
        self.crop_size = crop_size
        self.scale_size = scale_size

        self.image_transforms = A.Compose([
            A.HorizontalFlip(p=0.2),
            A.RandomCrop(width=crop_size[0], height=crop_size[1]),
            A.Resize(width=self.scale_size[0], height=self.scale_size[1]),
        ])

        with open(os.path.join(dataset_path, csv_name)) as training_csv:
            csv_content = csv.reader(training_csv, delimiter=',')
            for row in csv_content:
                input_image_path = os.path.join(*(row[0].split(os.path.sep)[1:]))
                depth_image_path = os.path.join(*(row[1].split(os.path.sep)[1:]))
                self.input_images.append(os.path.join(self.dataset_path,input_image_path))
                self.depth_images.append(os.path.join(self.dataset_path,depth_image_path))

        self.dataset_size = len(self.input_images)


    def __len__(self) :
        return self.dataset_size
    
    def __getitem__(self, idx):
        image = cv2.imread(self.input_images[idx]).astype("float32")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        depth_image = cv2.imread(self.depth_images[idx], cv2.IMREAD_UNCHANGED).astype("float32")

        image /= 255
        depth_image /= 255

        image = self.image_transforms(image=image)["image"]
        depth_image = self.image_transforms(image=depth_image)["image"]

        tf_image = tf.convert_to_tensor(image)
        tf_depth_image = tf.expand_dims(tf.convert_to_tensor(depth_image),2)
        
        return tf_image, tf_depth_image
    
    def on_epoch_end(self):
        pass


def get_nyu2_data_generator(batch_size, dataset_path, csv_name, crop_size, scale_size):

    dataset = NYUV2DataSet(dataset_path, csv_name, crop_size, scale_size)

    output_signature = (tf.TensorSpec(shape=(*scale_size, 1), dtype=tf.float32),
                        tf.TensorSpec(shape=(*scale_size, 1), dtype=tf.float32))
    
    def generator(ds):
        for sample in ds:
            image, depth_image = sample
            yield (image, depth_image)
    
    data_generator = tf.data.Dataset.from_generator(partial(generator, ds=dataset), output_signature=output_signature)

    data_generator = data_generator.batch(batch_size=batch_size)

    return data_generator
    




# class BaseDataset(Dataset):
#     def __init__(self, crop_size, fold_ratio=1, args=None, is_maxim=True):
#         self.count = 0
#         self.fold_ratio = fold_ratio
#         self.is_maxim = is_maxim

#         train_transform = [
#             A.HorizontalFlip(),
#             A.RandomCrop(crop_size[1], crop_size[0]),
#         ]
#         test_transform = [
#             A.CenterCrop(crop_size[1], crop_size[0]),
#         ]
#         self.train_transform = train_transform
#         self.test_transform = test_transform
#         self.to_tensor = transforms.ToTensor()
#         self.args = args

#     def augment_training_data(self, image, depth):
#         H, W, C = image.shape

#         image, depth = self.common_augment(image, depth, self.train_transform)

#         self.count += 1

#         return image, depth

#     def common_augment(self, image, depth, transform):
#         additional_targets = {"depth": "mask"}
#         aug = A.Compose(transforms=transform, additional_targets=additional_targets)
#         augmented = aug(image=image, depth=depth)
#         image = augmented["image"]
#         depth = augmented["depth"]

#         if self.is_maxim:
#             image = self.apply_ai8x_transforms(image)
#             depth = self.apply_ai8x_transforms(depth)
#         return image, depth

#     def apply_ai8x_transforms(self, x):
#         import ai8x

#         x = self.to_tensor(x)
#         x = ai8x.normalize(self.args)(x)
#         x = ai8x.fold(fold_ratio=self.fold_ratio)(x)
#         return x

#     def augment_test_data(self, image, depth):
#         image, depth = self.common_augment(image, depth, self.test_transform)

#         return image, depth


# class NYUv2Depth(BaseDataset):
#     def __init__(
#         self,
#         data_path,
#         args,
#         filenames_path,
#         is_train=True,
#         crop_size=(448, 576),
#         scale_size=None,
#         fold_ratio=1,
#     ):
#         super().__init__(
#             crop_size,
#             fold_ratio=fold_ratio,
#             args=args,
#             is_maxim=getattr(args, "is_maxim", True),
#         )

#         self.scale_size = scale_size

#         self.is_train = is_train
#         self.data_path = Path(data_path)

#         self.image_path_list = []
#         self.depth_path_list = []
#         self.base_dir = Path(filenames_path).parent

#         txt_path = Path(filenames_path)
#         if is_train:
#             txt_path /= "nyu2_train.csv"
#             self.data_path = Path(self.data_path / "nyu2_train")
#         else:
#             txt_path /= "nyu2_test.csv"
#             self.data_path = Path(self.data_path / "nyu2_test")

#         import pandas as pd

#         self.df = pd.read_csv(txt_path, header=None, names=["img_path", "depth_path"])
#         phase = "train" if is_train else "test"
#         print("Dataset: NYU Depth V2")
#         print("# of %s images: %d" % (phase, len(self.df)))

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         img_path = str(self.base_dir / self.df.loc[idx, "img_path"])
#         gt_path = str(self.base_dir / self.df.loc[idx, "depth_path"])

#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
#         depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype("float32")

#         if self.is_train:
#             image, depth = self.augment_training_data(image, depth)
#         else:
#             image, depth = self.augment_test_data(image, depth)

#         if self.scale_size:
#             image = cv2.resize(image, (self.scale_size[1], self.scale_size[0]))
#             depth = cv2.resize(depth, (self.scale_size[1], self.scale_size[0]))

#         image = np.expand_dims(image, axis=2)
#         depth = np.expand_dims(depth, axis=2)
#         depth = depth.astype("float32")
#         image = image.astype("float32")
#         depth /= 1000.0
#         depth = np.clip(depth, 0, 1)
#         image /= 255.0

#         return image, depth

# def get_tf_nyuv2_ds(data_path, args):
#     nyuv2_ds_train = NYUv2Depth(
#         data_path=data_path,
#         filenames_path=data_path,
#         args=args,
#         is_train=True,
#         crop_size=args.crop_size,
#         scale_size=args.target_size,
#         fold_ratio=args.out_fold_ratio,
#     )
#     nyuv2_ds_test = NYUv2Depth(
#         data_path=data_path,
#         filenames_path=data_path,
#         is_train=False,
#         crop_size=args.crop_size,
#         scale_size=args.target_size,
#         fold_ratio=args.out_fold_ratio,
#         args=args,
#     )
#     _ = nyuv2_ds_train[0]

#     def generator(ds):
#         for sample in ds:
#             img, depth = sample
#             yield (img, depth)

#     output_signature = (
#         tf.TensorSpec(shape=(*args.target_size, 1), dtype=tf.float32),
#         tf.TensorSpec(shape=(*args.target_size, 1), dtype=tf.float32),
#     )

#     val_size = int(0.2 * len(nyuv2_ds_train))  # 20% of the dataset

#     seed_generator = torch.Generator().manual_seed(111)
#     train_dataset, val_dataset = random_split(
#         nyuv2_ds_train,
#         [len(nyuv2_ds_train) - val_size, val_size],
#         generator=seed_generator,
#     )
#     datasets = []

#     print("Train size: ", len(train_dataset))
#     print("Val size: ", len(val_dataset))
#     print("Test size: ", len(nyuv2_ds_test))

#     for ds in [train_dataset, val_dataset, nyuv2_ds_test]:
#         if cfg.do_overfit:
#             ds = Subset(ds, range(1))
#         elif cfg.do_subsample:
#             ds = Subset(ds, range(0, 1000))
#         tf_dataset = (
#             tf.data.Dataset.from_generator(
#                 partial(generator, ds=ds), output_signature=output_signature
#             )
#             .batch(args.batch_size)
#             .prefetch(1)
#         )
#         datasets.append(tf_dataset)
#     return datasets