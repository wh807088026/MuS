import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from matplotlib import pyplot as plt
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from varname import argname

from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
import json

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x
ALPHABET='Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
SPECIAL_ALPHABET='ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω'
IMG_HEIGHT = 32
ENGLISH_WORDS_PATH = 'files/english_words.txt'
def printShape(temp=None, isTamp=False):
    """
    测试输出
    """
    if temp is None:
        return
    try:
        if isTamp:
            if isinstance(temp, list):
                info_tr = ('%s.len: %s, %s = %s' %
                           (argname('temp'), len(temp), argname('temp'), temp))
            else:
                info_tr = ('%s.shape: %s, %s = %s' %
                           (argname('temp'), temp.shape, argname('temp'), temp))
        else:
            if isinstance(temp, list):
                info_tr = ('%s.len: %s' %
                           (argname('temp'), len(temp)))
            else:
                info_tr = ('%s.shape: %s' %
                           (argname('temp'), temp.shape))
    except ValueError:
        info_tr = 'no attribute shape'
    print(info_tr)


def get_transform(grayscale=False, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


class ImagePickle:
    """  pickle image 打包与解码
    """

    def __init__(self, datasets=None, datasets_path=None, collator_resolution=16, num_examples=15,
                 target_transform=None,
                 min_virtual_size=0):
        # self.datasets = {}
        # for dataset_name in sorted(datasets.split(',')):
        #     dataset = dataset_class(os.path.join(datasets_path, f'{dataset_name}-32.pickle'), **kwargs)
        #     self.datasets[dataset_name] = dataset
        # self.alphabet = ''.join(sorted(set(''.join(d.alphabet for d in self.datasets.values()))))
        self.NUM_EXAMPLES = num_examples
        self.min_virtual_size = min_virtual_size
        self.datasets = datasets
        if self.datasets == 'unifont':
            self.base_path = os.path.join(datasets_path, f'{datasets}.pickle')
            self.file_to_store = open(self.base_path, "rb")
            self.IMG_DATA = pickle.load(self.file_to_store)
            print(self.IMG_DATA[0].keys())
        else:
            self.base_path = os.path.join(datasets_path, f'{datasets}-32.pickle')
            self.file_to_store = open(self.base_path, "rb")
            self.IMG_DATA = pickle.load(self.file_to_store)
            self.IMG_DATATR = self.IMG_DATA['train']
            self.IMG_DATATE = self.IMG_DATA['test']
            # base_path=DATASET_PATHS
        self.IMG_DATATR = dict(list(self.IMG_DATATR.items()))  # [:NUM_WRITERS])
        if 'None' in self.IMG_DATATR.keys():
            del self.IMG_DATATR['None']

        self.alphabet = ''.join(sorted(set(''.join(d['label'] for d in sum(self.IMG_DATATR.values(), [])))))
        self.author_id = list(self.IMG_DATATR.keys())
        self.floder = '/mnt/ssd/Iam_database/IAM_pkl'
        self.floder1 = '/mnt/ssd/Iam_database/unifont'
        self.transform = get_transform(grayscale=True)
        self.target_transform = target_transform

        self.collate_fn = TextCollator(collator_resolution)

    def __len__(self):
        return max(len(self.author_id), self.min_virtual_size)

    @property
    def num_writers(self):
        return len(self.author_id)

    def __getitem__(self, index):
        NUM_SAMPLES = self.NUM_EXAMPLES
        index = index % len(self.author_id)

        author_id = self.author_id[index]

        self.IMG_DATA_AUTHOR = self.IMG_DATA[author_id]
        random_idxs = np.random.choice(len(self.IMG_DATA_AUTHOR), NUM_SAMPLES, replace=True)

        rand_id_real = np.random.choice(len(self.IMG_DATA_AUTHOR))
        real_img = self.transform(self.IMG_DATA_AUTHOR[rand_id_real]['img'].convert('L'))
        real_labels = self.IMG_DATA_AUTHOR[rand_id_real]['label'].encode()

        imgs = [np.array(self.IMG_DATA_AUTHOR[idx]['img'].convert('L')) for idx in random_idxs]
        labels = [self.IMG_DATA_AUTHOR[idx]['label'].encode() for idx in random_idxs]

        max_width = 192  # [img.shape[1] for img in imgs]

        imgs_pad = []
        imgs_wids = []

        for img in imgs:
            img = 255 - img
            img_height, img_width = img.shape[0], img.shape[1]
            outImg = np.zeros((img_height, max_width), dtype='float32')
            outImg[:, :img_width] = img[:, :max_width]

            img = 255 - outImg

            imgs_pad.append(self.transform(Image.fromarray(img.astype(np.uint8))))
            imgs_wids.append(img_width)

        imgs_pad = torch.cat(imgs_pad, 0)

        item = {
            'simg': imgs_pad,  # N 个图像的宽度 [list(N)]
            'swids': imgs_wids,  # N 张图片 （15） 来自同一作者 [N (15), H (32), MAX_W (192)]
            'img': real_img,  # 输入图像 [1, H (32), W]
            'label': real_labels,  # 输入图像的标签 [byte]
            'img_path': 'img_path',
            'idx': 'indexes',
            'wcl': index  # 作者ID [int]
        }
        return item

    def encoderIAM15(self):
        """ pickle image 打包
        {'img': <PIL.Image.Image image mode=RGB size=80x32 at 0x7F16CBF319A0>, 'label': 'first','img_id': 56761}
        IMG_DATA = {'train': {'000': [{}{}]},
                    'test': {'002': [{}{}]}
                    }
        """

        dataset_name1 = sorted(self.IMG_DATATR)
        dataset_name2 = sorted(self.IMG_DATATE)
        IMG_DATATR15 = {}
        IMG_DATATE15 = {}
        for i in dataset_name1:
            imgs = {}
            for dataset in self.IMG_DATATR[i]:
                # 构建文件名（你可以根据需要修改文件名的构建方式）
                width, height = dataset['img'].size

                imgs_id = dataset['img_id']

                imgs[f'{imgs_id}'] = width

            sorted_data = {k: v for k, v in sorted(imgs.items(), key=lambda item: item[1], reverse=True)}
            keys_list = list(sorted_data.keys())[:15]
            keys_list = list(map(int, keys_list))
            img15 = []
            for dataset in self.IMG_DATATR[i]:
                item_dict = {}
                if dataset['img_id'] in keys_list:
                    item_dict = {
                        'img': dataset['img'],
                        'label': dataset['label'],
                        'img_id': dataset['img_id'],
                    }
                    img15.append(item_dict)
            IMG_DATATR15[f'{i}'] = img15

        for i in dataset_name2:
            imgs = {}
            for dataset in self.IMG_DATATE[i]:
                # 构建文件名（你可以根据需要修改文件名的构建方式）
                width, height = dataset['img'].size

                imgs_id = dataset['img_id']

                imgs[f'{imgs_id}'] = width

            sorted_data = {k: v for k, v in sorted(imgs.items(), key=lambda item: item[1], reverse=True)}
            keys_list = list(sorted_data.keys())[:15]
            keys_list = list(map(int, keys_list))
            img15 = []
            for dataset in self.IMG_DATATE[i]:
                item_dict = {}
                if dataset['img_id'] in keys_list:
                    item_dict = {
                        'img': dataset['img'],
                        'label': dataset['label'],
                        'img_id': dataset['img_id'],
                    }
                    img15.append(item_dict)
            IMG_DATATE15[f'{i}'] = img15

        IMG_DATA15 = {
            'train': IMG_DATATR15,
            'test': IMG_DATATE15
        }
        floder15 = os.path.join(self.floder, 'IAM-32-15.pickle')
        with open(floder15, 'wb') as f:
            pickle.dump(IMG_DATA15, f)

    def encoderIAM(self):
        """
        pickle image 打包
        {'img': <PIL.Image.Image image mode=RGB size=80x32 at 0x7F16CBF319A0>, 'label': 'first','img_id': 56761}
        IMG_DATA = {'train': {'000': [{}{}]},
                    'test': {'002': [{}{}]}
                    }
        """
        dataset_name1 = sorted(self.IMG_DATATR)
        dataset_name2 = sorted(self.IMG_DATATE)
        IMG_DATATR15 = {}
        IMG_DATATE15 = {}
        for i in dataset_name1:
            dataset_r = self.IMG_DATATR[i]
            imgs = {}
            for dataset in self.IMG_DATATR[i]:
                # 构建文件名（你可以根据需要修改文件名的构建方式）
                width, height = dataset['img'].size

                imgs_id = dataset['img_id']

                imgs[f'{imgs_id}'] = width

            sorted_data = {k: v for k, v in sorted(imgs.items(), key=lambda item: item[1], reverse=True)}
            keys_list = list(sorted_data.keys())[:15]
            keys_list = list(map(int, keys_list))
            img15 = []
            for dataset in self.IMG_DATATR[i]:
                item_dict = {}
                if dataset['img_id'] in keys_list:
                    item_dict = {
                        'img15': dataset['img'],
                        'label15': dataset['label'],
                        'img_id15': dataset['img_id'],
                    }
                    img15.append(item_dict)
            dataset_r = dataset_r + img15
            IMG_DATATR15[f'{i}'] = dataset_r

        for i in dataset_name2:
            dataset_r = self.IMG_DATATE[i]
            imgs = {}
            for dataset in self.IMG_DATATE[i]:
                # 构建文件名（你可以根据需要修改文件名的构建方式）
                width, height = dataset['img'].size

                imgs_id = dataset['img_id']

                imgs[f'{imgs_id}'] = width

            sorted_data = {k: v for k, v in sorted(imgs.items(), key=lambda item: item[1], reverse=True)}
            keys_list = list(sorted_data.keys())[:15]
            keys_list = list(map(int, keys_list))
            img15 = []
            for dataset in self.IMG_DATATE[i]:
                item_dict = {}
                if dataset['img_id'] in keys_list:
                    item_dict = {
                        'img15': dataset['img'],
                        'label15': dataset['label'],
                        'img_id15': dataset['img_id'],
                    }
                    img15.append(item_dict)
            dataset_r = dataset_r + img15
            IMG_DATATE15[f'{i}'] = dataset_r

        for i in IMG_DATATR15:
            print(IMG_DATATR15[i])
            return
        IMG_DATA15 = {
            'train': IMG_DATATR15,
            'test': IMG_DATATE15
        }
        # floder15 = os.path.join(self.floder, 'IAM-32-15.pickle')
        # with open(floder15, 'wb') as f:
        #     pickle.dump(IMG_DATA15, f)

    def decoderIAM15(self, Tr=True):
        """ pickle image 解码
        """
        if Tr:
            dataset_name = sorted(self.IMG_DATATR)
            folder = os.path.join(self.floder, f'train15')
            os.makedirs(folder, exist_ok=True)
            for i in dataset_name:
                author_dir = os.path.join(folder, i)
                os.makedirs(author_dir, exist_ok=True)
                imgs = {}
                # dataset:0x7F0C983E9FD0 >, 'label': 'PRESIDENT', 'img_id': 1497}
                for index, dataset in enumerate(self.IMG_DATATR[i]):
                    # print(dataset)
                    # 构建文件名（你可以根据需要修改文件名的构建方式）
                    width, height = dataset['img'].size
                    item_dict = {
                        'img_id': dataset['img_id'],
                        'width': width,
                        'img': dataset['img']
                    }
                    imgs[f'{index + 1}'] = item_dict
                # dataset_width = sorted(dataset_width)
                sorted_data = dict(sorted(imgs.items(), key=lambda x: x[1]['width'], reverse=True))

                for index, j in enumerate(sorted_data):
                    if index > 14:
                        break
                    file_name = f"{sorted_data[j]['img_id']}.png"  # 假设使用图像的唯一标识作为文件名
                    sorted_data[j]['img'].save(os.path.join(author_dir, file_name))
        else:
            dataset_name = sorted(self.IMG_DATATE)
            folder = os.path.join(self.floder, f'test15')
            os.makedirs(folder, exist_ok=True)
            for i in dataset_name:
                author_dir = os.path.join(folder, i)
                os.makedirs(author_dir, exist_ok=True)
                imgs = {}
                # dataset:0x7F0C983E9FD0 >, 'label': 'PRESIDENT', 'img_id': 1497}
                for index, dataset in enumerate(self.IMG_DATATE[i]):
                    # print(dataset)
                    # 构建文件名（你可以根据需要修改文件名的构建方式）
                    width, height = dataset['img'].size
                    item_dict = {
                        'img_id': dataset['img_id'],
                        'width': width,
                        'img': dataset['img']
                    }
                    imgs[f'{index + 1}'] = item_dict
                # dataset_width = sorted(dataset_width)
                sorted_data = dict(sorted(imgs.items(), key=lambda x: x[1]['width'], reverse=True))

                for index, j in enumerate(sorted_data):
                    if index > 14:
                        break
                    file_name = f"{sorted_data[j]['img_id']}.png"  # 假设使用图像的唯一标识作为文件名
                    sorted_data[j]['img'].save(os.path.join(author_dir, file_name))

    def decoderIAM(self, Tr=True):
        """pickle image 解码
        """

        if Tr:
            dataset_name = sorted(self.IMG_DATATR)
            folder = os.path.join(self.floder, f'train')
            os.makedirs(folder, exist_ok=True)
            for i in dataset_name:
                author_dir = os.path.join(folder, i)
                os.makedirs(author_dir, exist_ok=True)
                # dataset:0x7F0C983E9FD0 >, 'label': 'PRESIDENT', 'img_id': 1497}
                for dataset in self.IMG_DATATR[i]:
                    # print(dataset)
                    # 构建文件名（你可以根据需要修改文件名的构建方式）
                    file_name = f"{dataset['img_id']}.png"  # 假设使用图像的唯一标识作为文件名
                    dataset['img'].save(os.path.join(author_dir, file_name))
        else:
            dataset_name = sorted(self.IMG_DATATE)
            folder = os.path.join(self.floder, f'test')
            os.makedirs(folder, exist_ok=True)
            for i in dataset_name:
                author_dir = os.path.join(folder, i)
                os.makedirs(author_dir, exist_ok=True)
                # dataset:0x7F0C983E9FD0 >, 'label': 'PRESIDENT', 'img_id': 1497}
                for dataset in self.IMG_DATATE[i]:
                    # print(dataset)
                    # 构建文件名（你可以根据需要修改文件名的构建方式）
                    file_name = f"{dataset['img_id']}.png"  # 假设使用图像的唯一标识作为文件名
                    dataset['img'].save(os.path.join(author_dir, file_name))

    def decoderUni(self, Tr=True):
        """pickle image 解码
        """
        if Tr:
            dataset_name = sorted(self.IMG_DATATR)
            folder = os.path.join(self.floder, f'train')
            os.makedirs(folder, exist_ok=True)
            for i in dataset_name:
                author_dir = os.path.join(folder, i)
                os.makedirs(author_dir, exist_ok=True)
                # dataset:0x7F0C983E9FD0 >, 'label': 'PRESIDENT', 'img_id': 1497}
                for dataset in self.IMG_DATATR[i]:
                    # print(dataset)
                    # 构建文件名（你可以根据需要修改文件名的构建方式）
                    file_name = f"{dataset['img_id']}.png"  # 假设使用图像的唯一标识作为文件名
                    dataset['img'].save(os.path.join(author_dir, file_name))
        else:
            dataset_name = sorted(self.IMG_DATATE)
            folder = os.path.join(self.floder, f'test')
            os.makedirs(folder, exist_ok=True)
            for i in dataset_name:
                author_dir = os.path.join(folder, i)
                os.makedirs(author_dir, exist_ok=True)
                # dataset:0x7F0C983E9FD0 >, 'label': 'PRESIDENT', 'img_id': 1497}
                for dataset in self.IMG_DATATE[i]:
                    # print(dataset)
                    # 构建文件名（你可以根据需要修改文件名的构建方式）
                    file_name = f"{dataset['img_id']}.png"  # 假设使用图像的唯一标识作为文件名
                    dataset['img'].save(os.path.join(author_dir, file_name))

    def test(self):
        dataset_name = sorted(self.IMG_DATATR)
        print(dataset_name)


class ttfModule:
    """
    ttf module
    """

    def __init__(self, input_folder, output_folder='', alphabet=ALPHABET + SPECIAL_ALPHABET, device='cuda',
                 input_type='times'):
        super(ttfModule, self).__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.device = device
        self.alphabet = alphabet
        self.input_type = input_type
        self.transform = get_transform(grayscale=True)
        self.ttfImg = ttfToImage(self.input_folder)
        self.symbols = self.get_symbols('unifont')
        self.textImg = self.get_textImg(self.input_type)

    def get_symbols(self, input_type):
        """

        Args:
            input_type:

        Returns:

        """
        with open("../files/{}.pickle".format(input_type), "rb") as f:
            symbols = pickle.load(f)

        symbols = {sym['idx'][0]: sym['mat'].astype(np.float32).flatten() for sym in symbols}
        # self.special_symbols = [self.symbols[ord(char)] for char in special_alphabet]
        symbols = [symbols[ord(char)] for char in self.alphabet]
        symbols.insert(0, np.zeros_like(symbols[0]))
        symbols = np.stack(symbols)
        return torch.from_numpy(symbols).float().to(self.device)

    def get_textImg(self, input_type):
        """

        Args:
            input_type:

        Returns:

        """
        with open("../files/{}.pickle".format(input_type), "rb") as f:
            testImg = pickle.load(f)
        id_dict = {entry['img_id']: entry for entry in testImg}
        id_dict = dict(list(id_dict.items()))
        idx = np.random.choice(len(id_dict))
        real_labels = id_dict[idx]['label'].encode()
        img = np.array(id_dict[idx]['img'].convert('L'))
        max_width = 192  # [img.shape[1] for img in imgs]

        img = 255 - img
        img_height, img_width = img.shape[0], img.shape[1]
        outImg = np.zeros((img_height, max_width), dtype='float32')
        outImg[:, :img_width] = img[:, :max_width]
        img = 255 - outImg
        img = self.transform((Image.fromarray(img)))

        item = {
            'img': img,
            'label': real_labels,
            'idx': idx
        }

        print(img.shape)

        return item

    def encoder(self):
        """
        pickle image 打包
        {'img': <PIL.Image.Image image mode=RGB size=80x32 at 0x7F16CBF319A0>, 'label': 'first','img_id': 56761}
        """
        item = self.ttfImg.getitem()
        pickle_path = os.path.join(self.input_folder, 'files/{}.pickle'.format(self.input_type))
        with open(pickle_path, 'wb') as f:
            pickle.dump(item, f)

    def decoder(self):
        """pickle image 解码
        """
        dataset_name = sorted(self.IMG_DATATR)
        folder = os.path.join(self.floder, f'train')
        os.makedirs(folder, exist_ok=True)
        for i in dataset_name:
            author_dir = os.path.join(folder, i)
            os.makedirs(author_dir, exist_ok=True)
            # dataset:0x7F0C983E9FD0 >, 'label': 'PRESIDENT', 'img_id': 1497}
            for dataset in self.IMG_DATATR[i]:
                # print(dataset)
                # 构建文件名（你可以根据需要修改文件名的构建方式）
                file_name = f"{dataset['img_id']}.png"  # 假设使用图像的唯一标识作为文件名
                dataset['img'].save(os.path.join(author_dir, file_name))

    def test(self):

        tensor_data = torch.tensor([[18, 8, 16, 8, 2, 6, 15, 19, 3, 10, 61, 8, 18, 0, 0, 0],
                                    [36, 19, 15, 12, 4, 19, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [15, 8, 6, 15, 19, 2, 12, 21, 13, 15, 6, 19, 6, 10, 13, 2],
                                    [8, 2, 17, 10, 15, 13, 2, 19, 11, 8, 0, 0, 0, 0, 0, 0],
                                    [22, 15, 19, 16, 7, 10, 12, 6, 13, 16, 8, 21, 7, 19, 3, 10],
                                    [21, 15, 10, 12, 20, 13, 10, 18, 19, 3, 0, 0, 0, 0, 0, 0],
                                    [17, 15, 13, 13, 20, 10, 2, 11, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [12, 24, 3, 14, 13, 16, 4, 19, 2, 0, 0, 0, 0, 0, 0, 0]],
                                   device='cuda:0')
        tensor_data = torch.tensor([[18, 8, 16, 8, 2, 6, 15, 19, 3, 10, 61, 8, 18, 0, 0, 0]],
                                   device='cuda:0')
        data = self.symbols[tensor_data]

        # printShape(tensor_data)
        # printShape(self.symbols)
        # printShape(data, True)

    def forward(self, QR):

        return


def get_transform(grayscale=False, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)

def calculate_image_size(word, font_path, fixed_height):
    """

    Args:
        word:
        font_path:
        fixed_height:

    Returns:

    """
    # 计算图像的宽度
    font_size = calculate_font_size(word, font_path, fixed_height)
    font = ImageFont.truetype(font_path, font_size)
    text_width = calculate_text_width(word, font)
    return text_width, fixed_height


def calculate_font_size(word, font_path, fixed_height):
    """

    Args:
        word:
        font_path:
        fixed_height:

    Returns:

    """
    # 计算字体大小
    font_size = 1
    while True:
        font = ImageFont.truetype(font_path, font_size)
        text_height = font.getbbox(word)[3]
        if text_height >= fixed_height:
            break
        font_size += 1
    return font_size


def calculate_text_width(text, font):
    """

    Args:
        text:
        font:

    Returns:

    """
    # 获取文本的宽度
    return font.getbbox(text)[2]


class ttfToImage:
    """
        ttfToImage
    """

    def __init__(self, input_folder, output_folder='', alphabet=ALPHABET, device='cuda', input_type='times',
                 words=None):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.device = device
        self.alphabet = alphabet
        self.input_type = input_type
        self.ttf_path = os.path.join(input_folder, 'files/{}.ttf'.format(self.input_type))
        self.lex = self.getWords(os.path.join(input_folder, ENGLISH_WORDS_PATH), words)
        self.transform = get_transform(grayscale=True)

    def generate_word_image(self, word, fixed_height=IMG_HEIGHT):
        img_width, img_height = calculate_image_size(word, self.ttf_path, fixed_height)
        img = Image.new("L", (img_width, fixed_height), "white")
        draw = ImageDraw.Draw(img)
        font_size = calculate_font_size(word, self.ttf_path, fixed_height)
        font = ImageFont.truetype(self.ttf_path, font_size)
        y = (fixed_height - font.getbbox(word)[3]) // 2
        draw.text((0, y), word, font=font, fill="black", spacing=0)
        return img

    def save_word_image(self, img, imgName):
        img.save(os.path.join(self.output_folder, '{}.png'.format(imgName)))

    def getWords(self, english_folder, words=None):
        """

        Returns:

        """
        if words is None:
            with open(english_folder, 'rb') as f:
                lexR = f.read().splitlines()
        else:
            lexR = words
        # print('lexR:{}'.format(lexR))
        lex = []
        for word in lexR:
            try:
                word = word.decode("utf-8")
            except:
                continue
            if len(word) < 20:
                lex.append(word)

        # print('lex:{}'.format(lex))
        return lex

    def process_range(self, batch_size=100):
        """

        Args:
            batch_size:
        """
        with ThreadPoolExecutor(max_workers=32) as executor:
            start = 0
            end = len(self.lex)
            for i in tqdm(range(start, end, batch_size), desc="Processing Batches"):
                futures = [executor.submit(self.forward, j) for j in range(i, min(i + batch_size, end))]
                # 等待所有任务完成
                for future in futures:
                    future.result()

    def getitem(self):
        """

        Returns:ttfImage

        """
        ttfImage = []
        start = 0
        end = len(self.lex)
        for i in tqdm(range(start, end), desc="Getting items"):
            img = self.generate_word_image(self.lex[i])
            item_dict = {
                'img': img,
                'label': self.lex[i],
                'img_id': i,
            }
            ttfImage.append(item_dict)

        # print(len(ttfImage))
        # print(ttfImage)
        return ttfImage

    def getTextItem(self):
        """

        Returns:ttfImage

        """
        imgs_pad = []
        label_pad = []
        idx_pad = []
        start = 0
        end = len(self.lex)
        # for i in tqdm(range(start, end), desc="Getting TextItem"):
        for i in range(start, end):
            text_label = self.lex[i].encode()
            img = self.generate_word_image(self.lex[i])
            text_img = np.array(img.convert('L'))
            max_width = 192  # [img.shape[1] for img in imgs]
            text_img = 255 - text_img
            img_height, img_width = text_img.shape[0], text_img.shape[1]
            outImg = np.zeros((img_height, max_width), dtype='float32')
            outImg[:, :img_width] = text_img[:, :max_width]
            text_img = 255 - outImg
            text_img = np.array(self.transform((Image.fromarray(text_img))))
            text_img = np.array(text_img)
            text_img = torch.Tensor(text_img)

            imgs_pad.append(text_img)
            label_pad.append(text_label)
            idx_pad.append(i)
        imgs_pad = torch.cat(imgs_pad, dim=0)
        ttfImage = {
            'timg': imgs_pad,
            'tlabel': label_pad,
            'tidx': idx_pad,
        }
        # print(len(ttfImage))
        # print(ttfImage)
        return ttfImage

    def test(self):
        print("Test")
        self.generate_word_image("Helloaaa", '0')

    def save_CPU(self):
        start = 0
        end = len(self.lex)
        for i in tqdm(range(start, end), desc="Save CPU"):
            img = self.generate_word_image(self.lex[i])
            self.save_word_image(img, str(i))

    def forward(self, i):
        """

        Args:
            i:
        """
        img = self.generate_word_image(self.lex[i])
        self.save_word_image(img, str(i))


if __name__ == '__main__':
    current_file_path = os.path.abspath(__file__)

    # 使用 os.path.dirname 获取上层目录
    parent_directory = os.path.dirname(os.path.dirname(current_file_path))
    outFolder = '/mnt/ssd/Iam_database/times_pkl'
    os.makedirs(outFolder, exist_ok=True)
    ttfToPickle = ttfModule(parent_directory)
    ttfImg = ttfToImage(parent_directory, outFolder)
    ttfToPickle.test()
    # ttfImg.forward()
    # ttfImg.getitem()
    # print(parent_directory)
