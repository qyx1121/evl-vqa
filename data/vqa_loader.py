import sys
sys.path.insert(0, '../')

import av
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import collections
import os
import os.path as osp
import h5py
import random as rd
import numpy as np
import os.path as osp
import cv2
from data.preprocess import transform
from torchvision import transforms
from .transform import create_random_augment, random_resized_crop

class VideoQADataset(Dataset):
    def __init__(
        self,
        csv_path,
        qmax_words=20,
        amax_words=5,
        bert_tokenizer=None,
        a2id=None,
        video_dir = '',
        args = None
    ):
        """
        :param csv_path: path to a csv containing columns video_id, question, answer
        :param features: dictionary mapping video_id to torch tensor of features
        :param qmax_words: maximum number of words for a question
        :param amax_words: maximum number of words for an answer
        :param bert_tokenizer: BERT tokenizer
        :param a2id: answer to index mapping
        :param ivqa: whether to use iVQA or not
        :param max_feats: maximum frames to sample from a video
        """
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir
        self.qmax_words = qmax_words
        self.amax_words = amax_words
        self.bert_tokenizer = bert_tokenizer
        self.mode = osp.basename(csv_path).split('.')[0] #train, val or test
        self.a2id = a2id
        self.preproc = transform()
        self.interpolation = 'bicubic'
        self.auto_augment = 'rand-m7-n4-mstd0.5-inc1'
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
        self.num_frames = args.num_frames
        self.sampling_rate = 16

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        
        cur_sample = self.data.loc[index]
        vid_id = cur_sample["video"]
        vid_id = str(vid_id)
        qid = str(cur_sample['qid'])
        question_txt = cur_sample['question']
        question_embd = self.bert_tokenizer.encode(
                question_txt,
                add_special_tokens=True,
                padding="longest",
                max_length=self.qmax_words,
                truncation=True,
            )[:self.qmax_words]
        
        num_pad = self.qmax_words - len(question_embd)

        question_embd = question_embd + [0] * num_pad
        question_embd = torch.tensor(
            question_embd,
            dtype=torch.long
        )

        type, answer = 0, 0

        question_id = vid_id+'_'+qid         
        
        answer_txts = cur_sample["answer"]
        answer_id = self.a2id.get(answer_txts, -1) 
        
        question_id = qid

        video_dir = osp.join(self.video_dir, "video" + vid_id) + '.mp4'
        #video_paths = [osp.join(video_dir, i) for i in os.listdir(video_dir)]
        #video_paths = sorted(video_paths)
        container = av.open(video_dir)
        frames = {}
        for frame in container.decode(video=0):
            frames[frame.pts] = frame
        container.close()
        frames = [frames[k] for k in sorted(frames.keys())]
        frame_idx = self._random_sample_frame_idx(len(frames))
        frames = [frames[x].to_rgb().to_ndarray() for x in frame_idx]
        frames = torch.as_tensor(np.stack(frames)).float() / 255.

        if self.auto_augment is not None:
            aug_transform = create_random_augment(
                input_size=(frames.size(1), frames.size(2)),
                auto_augment=self.auto_augment,
                interpolation=self.interpolation,
            )
            frames = frames.permute(0, 3, 1, 2) # T, C, H, W
            frames = [transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))]
            frames = aug_transform(frames)
            frames = torch.stack([transforms.ToTensor()(img) for img in frames])
            frames = frames.permute(0, 2, 3, 1)

        frames = (frames - self.mean) / self.std
        frames = frames.permute(3, 0, 1, 2) # C, T, H, W
        frames = random_resized_crop(
            frames, 224, 224,
        )
       
        return {
            "video_id": vid_id,
            "question": question_embd,
            "question_txt": question_txt,
            "type": type,
            "answer":0,
            "answer_id": answer_id,
            "answer_txt": answer_txts,
            "question_id": question_id,
            "images": frames
        }
    
    def _random_sample_frame_idx(self, len):
        frame_indices = []

        if self.sampling_rate < 0: # tsn sample
            seg_size = (len - 1) / self.num_frames
            for i in range(self.num_frames):
                start, end = round(seg_size * i), round(seg_size * (i + 1))
                frame_indices.append(np.random.randint(start, end + 1))
        elif self.sampling_rate * (self.num_frames - 1) + 1 >= len:
            for i in range(self.num_frames):
                frame_indices.append(i * self.sampling_rate if i * self.sampling_rate < len else frame_indices[-1])
        else:
            start = np.random.randint(len - self.sampling_rate * (self.num_frames - 1))
            frame_indices = list(range(start, start + self.sampling_rate * self.num_frames, self.sampling_rate))

        return frame_indices


def videoqa_collate_fn(batch):
    """
    :param batch: [dataset[i] for i in N]
    :return: tensorized batch with the question and the ans candidates padded to the max length of the batch
    """
    qmax_len = max(len(batch[i]["question"]) for i in range(len(batch)))
    
    for i in range(len(batch)):
        if len(batch[i]["question"]) < qmax_len:
            batch[i]["question"] = torch.cat(
                [
                    batch[i]["question"],
                    torch.zeros(qmax_len - len(batch[i]["question"]), dtype=torch.long),
                ],
                0,
            )
    
    if not isinstance(batch[0]["answer"], int):
        amax_len = max(x["answer"].size(1) for x in batch)
        for i in range(len(batch)):
            if batch[i]["answer"].size(1) < amax_len:
                batch[i]["answer"] = torch.cat(
                    [
                        batch[i]["answer"],
                        torch.zeros(
                            (
                                batch[i]["answer"].size(0),
                                amax_len - batch[i]["answer"].size(1),
                            ),
                            dtype=torch.long,
                        ),
                    ],
                    1,
                )

    return default_collate(batch)


def get_videoqa_loaders(args, a2id, bert_tokenizer, test_mode):
    
    if test_mode:
        test_dataset = VideoQADataset(
            csv_path=args.test_csv_path,
            qmax_words=args.qmax_words,
            amax_words=args.amax_words,
            bert_tokenizer=bert_tokenizer,
            a2id=a2id,video_dir = args.video_dir
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size_val,
            num_workers=args.num_thread_reader,
            shuffle=False,
            drop_last=False,
            collate_fn=videoqa_collate_fn,
        )
        train_loader, val_loader = None, None
    else:
        
        train_dataset = VideoQADataset(
        csv_path=args.train_csv_path,
        qmax_words=args.qmax_words,
        amax_words=args.amax_words,
        bert_tokenizer=bert_tokenizer,
        a2id=a2id,
        video_dir = args.video_dir,
        args = args
        )
        train_sampler = None
        shuffle = True
        if args.distribute:
            train_sampler = DistributedSampler(train_dataset)
            shuffle = False
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_thread_reader,
            shuffle=shuffle,
            drop_last=True,
            sampler = train_sampler,
            collate_fn=videoqa_collate_fn,
        )
        val_loader = None
        '''
        if args.dataset.split('/')[0] in ['tgifqa','tgifqa2', 'msrvttmc']:
            args.val_csv_path = args.test_csv_path
        
        val_dataset = VideoQADataset(
            csv_path=args.val_csv_path,
            qmax_words=args.qmax_words,
            amax_words=args.amax_words,
            bert_tokenizer=bert_tokenizer,
            a2id=a2id,
            video_dir = args.video_dir
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size_val,
            num_workers=args.num_thread_reader,
            shuffle=False,
            collate_fn=videoqa_collate_fn,
        )'''
        test_loader = None

    return (train_loader, val_loader, test_loader)