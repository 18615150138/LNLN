import pickle
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader


__all__ = ['MMDataLoader']


class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.train_mode = args['base']['train_mode']
        self.datasetName = args['dataset']['datasetName']
        self.dataPath = args['dataset']['dataPath']
        self.missing_rate_eval_test = args['base']['missing_rate_eval_test']
        self.missing_seed = args['base']['seed']


        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
        }
        DATA_MAP[self.datasetName]()

    def __init_mosi(self):
        with open(self.dataPath, 'rb') as f:
            data = pickle.load(f)
        self.data = data #这是个引用方式，相当于没开新内存

        self.text = data[self.mode]['text_bert'].astype(np.float32)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)

        # #when run on mosei dataset ,it well warning memoryerror
        # self.vision = data[self.mode]['vision'].astype(np.float16)
        # #这个是创建了新对象保存数据，要开新内存，所以这就是为什么不管是先存self.data=data,后存self.vision还是先存self.vision,后存self.data=data爆内存的原因
        # self.audio = data[self.mode]['audio'].astype(np.float16)


        self.rawText = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']
        self.labels = {
            #这里只取了多模态标签
            'M': data[self.mode][self.train_mode+'_labels'].astype(np.float32),
            'missing_rate_l': np.zeros_like(data[self.mode][self.train_mode+'_labels']).astype(np.float32),
            'missing_rate_a': np.zeros_like(data[self.mode][self.train_mode+'_labels']).astype(np.float32),
            'missing_rate_v': np.zeros_like(data[self.mode][self.train_mode+'_labels']).astype(np.float32),
        }

        #如果是sims则可以取到三个单模态标签
        if self.datasetName == 'sims':
            for m in "TAV":
                self.labels[m] = data[self.mode][self.train_mode+'_labels_'+m]

        self.audio_lengths = data[self.mode]['audio_lengths']
        self.vision_lengths = data[self.mode]['vision_lengths']
        self.audio[self.audio == -np.inf] = 0

        if self.mode == 'train':
            missing_rate = [np.random.uniform(size=(len(data[self.mode][self.train_mode+'_labels']), 1)) for i in range(3)]
            # missing_rate = [
            #     np.array([[0.5], [0.7], [0.2], [0.9]]),  # 文本模态的缺失率
            #     np.array([[0.6], [0.1], [0.8], [0.4]]),  # 音频模态的缺失率
            #     np.array([[0.3], [0.9], [0.5], [0.7]])  # 视频模态的缺失率
            # ]
            for i in range(3):
                sample_idx = random.sample([i for i in range(len(missing_rate[i]))], int(len(missing_rate[i])/2))
                missing_rate[i][sample_idx] = 0
            # 对于文本模态：sample_idx = [0, 2]
            # 对于音频模态：sample_idx = [1, 3]
            # 对于视频模态：sample_idx = [0, 2]
            # missing_rate = [
            #     np.array([[0.0], [0.7], [0.0], [0.9]]),  # 文本模态的缺失率
            #     np.array([[0.6], [0.0], [0.8], [0.0]]),  # 音频模态的缺失率
            #     np.array([[0.0], [0.9], [0.0], [0.7]])  # 视频模态的缺失率
            # ]
            self.labels['missing_rate_l'] = missing_rate[0]
            self.labels['missing_rate_a'] = missing_rate[1]
            self.labels['missing_rate_v'] = missing_rate[2]
            #这样设置缺失率的话，就是这一批数据里，我有一半是missing_rate=0的，另一半的missing_rate是一个随机值
            #也就是train里一半的样本是没有缺失的，另一半里是有缺失的，缺失程度missing_rate随机而定，至于一个样本的missing_rate_i怎样影响样本i还没看到
        else:
            missing_rate = [self.missing_rate_eval_test * np.ones((len(data[self.mode][self.train_mode+'_labels']), 1)) for i in range(3)]
            self.labels['missing_rate_l'] = missing_rate[0]
            self.labels['missing_rate_a'] = missing_rate[1]
            self.labels['missing_rate_v'] = missing_rate[2]
            # self.labels['missing_rate_l'] = np.array([[0.5], [0.5], [0.5], [0.5]])
            # self.labels['missing_rate_a'] = np.array([[0.5], [0.5], [0.5], [0.5]])
            # self.labels['missing_rate_v'] = np.array([[0.5], [0.5], [0.5], [0.5]])

        self.text_m, self.text_length, self.text_mask, self.text_missing_mask = self.generate_m(self.text[:,0,:], self.text[:,1,:], None,
                                                                                missing_rate[0], self.missing_seed, mode='text')
        Input_ids_m = np.expand_dims(self.text_m, 1)
        Input_mask = np.expand_dims(self.text_mask, 1)
        Segment_ids = np.expand_dims(self.text[:,2,:], 1)
        self.text_m = np.concatenate((Input_ids_m, Input_mask, Segment_ids), axis=1) 

        self.audio_m, self.audio_length, self.audio_mask, self.audio_missing_mask = self.generate_m(self.audio, None, self.audio_lengths,
                                                                                    missing_rate[1], self.missing_seed, mode='audio')
        self.vision_m, self.vision_length, self.vision_mask, self.vision_missing_mask = self.generate_m(self.vision, None, self.vision_lengths,
                                                                                    missing_rate[2], self.missing_seed, mode='vision')

    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()


    def generate_m(self, modality, input_mask, input_len, missing_rate, missing_seed, mode='text'):
        
        if mode == 'text':
            input_len = np.argmin(input_mask, axis=1)
        elif mode == 'audio' or mode == 'vision':
            input_mask = np.array([np.array([1] * length + [0] * (modality.shape[1] - length)) for length in input_len])
        np.random.seed(missing_seed)

        #使用均匀分布生成一个与 input_mask 形状相同的随机矩阵，并与 missing_rate 比较，生成 missing_mask。
        missing_mask = (np.random.uniform(size=input_mask.shape) > missing_rate.repeat(input_mask.shape[1], 1)) * input_mask

        
        assert missing_mask.shape == input_mask.shape
        
        if mode == 'text':
            # CLS SEG Token unchanged.
            for i, instance in enumerate(missing_mask):
                instance[0] = instance[input_len[i] - 1] = 1
            
            modality_m = missing_mask * modality + (100 * np.ones_like(modality)) * (input_mask - missing_mask) # UNK token: 100.
        elif mode == 'audio' or mode == 'vision':
            modality_m = missing_mask.reshape(modality.shape[0], modality.shape[1], 1) * modality
        
        return modality_m, input_len, input_mask, missing_mask


    def __len__(self):
        return len(self.labels['M'])

    def __getitem__(self, index):
        if (self.mode == 'train') and (index == 0):
            # missing_rate = [np.random.uniform(0, 0.5, size=(len(self.data[self.mode][self.train_mode+'_labels']), 1)) for i in range(3)]
            missing_rate = [np.random.uniform(size=(len(self.data[self.mode][self.train_mode+'_labels']), 1)) for i in range(3)]
            
            for i in range(3):
                sample_idx = random.sample([i for i in range(len(missing_rate[i]))], int(len(missing_rate[i])/2))
                missing_rate[i][sample_idx] = 0

            self.labels['missing_rate_l'] = missing_rate[0]
            self.labels['missing_rate_a'] = missing_rate[1]
            self.labels['missing_rate_v'] = missing_rate[2]

            self.text_m, self.text_length, self.text_mask, self.text_missing_mask = self.generate_m(self.text[:,0,:], self.text[:,1,:], None,
                                                                                    missing_rate[0], self.missing_seed, mode='text')
            Input_ids_m = np.expand_dims(self.text_m, 1)
            Input_mask = np.expand_dims(self.text_mask, 1)
            Segment_ids = np.expand_dims(self.text[:,2,:], 1)
            self.text_m = np.concatenate((Input_ids_m, Input_mask, Segment_ids), axis=1) 

            self.audio_m, self.audio_length, self.audio_mask, self.audio_missing_mask = self.generate_m(self.audio, None, self.audio_lengths,
                                                                                        missing_rate[1], self.missing_seed, mode='audio')
            self.vision_m, self.vision_length, self.vision_mask, self.vision_missing_mask = self.generate_m(self.vision, None, self.vision_lengths,
                                                                                    missing_rate[2], self.missing_seed, mode='vision')


        sample = {
            'text': torch.Tensor(self.text[index]),
            'text_m': torch.Tensor(self.text_m[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'audio_m': torch.Tensor(self.audio_m[index]),
            'vision': torch.Tensor(self.vision[index]),
            'vision_m': torch.Tensor(self.vision_m[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        }

        return sample


def MMDataLoader(args):
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args['base']['batch_size'],
                       num_workers=args['base']['num_workers'],
                       shuffle=True if ds == 'train' else False)
        for ds in datasets.keys()
    }
    
    return dataLoader


def MMDataEvaluationLoader(args):
    datasets = MMDataset(args, mode='test')

    dataLoader = DataLoader(datasets,
                       batch_size=args['base']['batch_size'],
                       num_workers=args['base']['num_workers'],
                       shuffle=False)
    
    return dataLoader