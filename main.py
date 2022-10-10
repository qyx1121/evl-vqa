#!/usr/bin/env python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import argparse
from datetime import datetime
import builtins
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import BertTokenizer
from models.model import EVLTransformer
from models.vision_transformer import vit_presets
from models.weight_loaders import weight_loader_fn_dict
from data.vqa_loader import get_videoqa_loaders
from utils import compute_a2v, get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

def setup_print(is_master: bool):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            now = datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_steps', type=int,
                        help='number of training steps')
    parser.add_argument('--eval_only', action='store_true',
                        help='run evaluation only')
    parser.add_argument('--epochs', type=int,
                        default=30)
    parser.add_argument('--save_freq', type=int, default=5000,
                        help='save a checkpoint every N steps')
    parser.add_argument('--eval_freq', type=int, default=5000,
                        help='evaluate every N steps')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print log message every N steps')

    parser.add_argument('--backbone', type=str, choices=vit_presets.keys(), default='ViT-L/14-lnpre',
                        help='the backbone variant used to generate image feature maps')
    parser.add_argument('--backbone_path', type=str,
                        help='path to pretrained backbone weights')
    parser.add_argument('--backbone_type', type=str, default='clip', choices=weight_loader_fn_dict.keys(),
                        help='type of backbone weights (used to determine how to convert state_dict from different pretraining codebase)')
    parser.add_argument('--finetune_backbone', action='store_true',
                        help='finetune backbone weights')
    parser.add_argument('--decoder_num_layers', type=int, default=4,
                        help='number of decoder layers')
    parser.add_argument('--decoder_qkv_dim', type=int, default=1024,
                        help='q (k, v) projection output dimensions in decoder attention layers')
    parser.add_argument('--decoder_num_heads', type=int, default=16,
                        help='number of heads in decoder attention layers')
    parser.add_argument('--decoder_mlp_factor', type=float, default=4.0,
                        help='expansion factor of feature dimension in the middle of decoder MLPs')
    parser.add_argument('--num_classes', type=int, default=400,
                        help='number of classes')
    parser.add_argument('--cls_dropout', type=float, default=0.5,
                        help='dropout rate applied before the final classification linear projection')
    parser.add_argument('--decoder_mlp_dropout', type=float, default=0.5,
                        help='dropout rate applied in MLP layers in the decoder')
    parser.add_argument('--no_temporal_conv', action='store_false', dest='temporal_conv',
                        help='disable temporal convolution on frame features')
    parser.add_argument('--no_temporal_pos_embed', action='store_false', dest='temporal_pos_embed',
                        help='disable temporal position embeddings added to frame features')
    parser.add_argument('--no_temporal_cross_attention', action='store_false', dest='temporal_cross_attention',
                        help='disable temporal cross attention on frame query and key features')
    parser.set_defaults(temporal_conv=True, temporal_pos_embed=True, temporal_cross_attention=True)

    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='optimizer weight decay')
    parser.add_argument('--disable_fp16', action='store_false', dest='fp16',
                        help='disable fp16 during training or inference')
    parser.set_defaults(fp16=False)

    parser.add_argument('--batch_split', type=int, default=1,
                        help='optionally split the batch into smaller shards and forward/backward one shard '
                             'at a time to avoid out-of-memory error.')

    args = parser.parse_args()

    args.clip = True
    args.cls = False
    args.distribute = False
    args.n_proc = 1
    args.num_frames = 8
    args.vocab_path = "data/msrvtt/vocab.json"
    args.amax_words = 5
    args.qmax_words = 20
    args.train_csv_path = "data/msrvtt/train.csv"
    args.num_thread_reader = 4
    args.video_dir = "/home/qinyixin/data/dataset/MSRVTT-QA/video"

    if args.distribute:
        dist.init_process_group('nccl')
        setup_print(dist.get_rank() == 0)
        cuda_device_id = dist.get_rank() % torch.cuda.device_count()
        torch.cuda.set_device(cuda_device_id)
        if cuda_device_id ==0:
            writer = SummaryWriter('logs/ddp_{}_{}'.format("cls" if args.cls else "sim", args.num_frames))
    
    else:
        writer = SummaryWriter('logs/ddp_{}_{}'.format("cls" if args.cls else "sim", args.num_frames))
    model = EVLTransformer(
        backbone_name=args.backbone,
        backbone_type=args.backbone_type,
        backbone_path='ckpts/ViT-L-14.pt',
        backbone_mode='finetune' if args.finetune_backbone else ('freeze_fp16' if args.fp16 else 'freeze_fp32'),
        decoder_num_layers=args.decoder_num_layers,
        decoder_qkv_dim=args.decoder_qkv_dim,
        decoder_num_heads=args.decoder_num_heads,
        decoder_mlp_factor=args.decoder_mlp_factor,
        enable_temporal_conv=args.temporal_conv,
        enable_temporal_pos_embed=args.temporal_pos_embed,
        enable_temporal_cross_attention=args.temporal_cross_attention,
        decoder_mlp_dropout=args.decoder_mlp_dropout,
        num_frames=args.num_frames,
    )
    print(model)
    model.cuda()
    
    cuda_device_id = 0

    map_location = {'cuda:%d' % 0: 'cuda:%d' % cuda_device_id}
    ckpt = torch.load("ckpts/k400_vitl14_8f_dec4x1024.pth", map_location=map_location)['model']
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        if 'module' in k:
            name = k[7:]
            new_state_dict[name] = v
    model.load_state_dict(new_state_dict,strict=False)
    if args.distribute:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cuda_device_id], output_device=cuda_device_id,
        )
    
    else:
        model = torch.nn.DataParallel(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    a2id, id2a, a2v = compute_a2v(
            vocab_path=args.vocab_path,
            bert_tokenizer=bert_tokenizer,
            amax_words=args.amax_words,
        )
    train_loader, val_loader, test_loader = get_videoqa_loaders(args, a2id, bert_tokenizer, False)
    scheduler = get_cosine_schedule_with_warmup(
            optimizer, 0 , len(train_loader) * args.epochs
        )

    last_epoch = -1
    total_loss = 0
    acc = 0 
    n_iter = 0 
    for epoch in range(last_epoch + 1, args.epochs):
        if args.distribute:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
        for iter, batch in enumerate(train_loader):
            n_iter = epoch * len(train_loader) + iter
            video_frames = batch['images'].cuda()
            question = batch['question'].cuda()
            question_mask = (question > 0).float()
            answer_id = batch['answer_id'].cuda()
            model.module._compute_answer_embedding(a2v)
            predicts = model(
                video_frames, question, question_mask
            )

            loss = criterion(predicts,answer_id)
            total_loss += loss

            if args.distribute:
                dist.barrier()
                dist.all_reduce(loss,op=dist.ReduceOp.SUM)
                dist.all_reduce(total_loss,op=dist.ReduceOp.SUM)
                loss = loss / args.n_proc
                total_loss = total_loss / args.n_proc
            loss.backward()
            
            predicted = torch.max(predicts, dim=1).indices
            score = (predicted == answer_id).sum().item() / predicted.shape[0]
            acc = acc + score
            optimizer.step()
            optimizer.zero_grad()
            
            if n_iter % 10 ==0:
                print("epoch: {} iter: {}  acc :{}  loss:{} \n".format(epoch,iter,acc,loss))
            
            if (args.distribute and cuda_device_id == 0) or not args.distribute:
                writer.add_scalar("loss", total_loss, n_iter)
                writer.add_scalar("train_acc", acc, n_iter)
                acc = 0
                total_loss = 0
            
            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
            scheduler.step()


def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader):
    tot, hit1, hit5 = 0, 0, 0
    eval_st = datetime.now()
    for data, labels in loader:
        data, labels = data.cuda(), labels.cuda()
        assert data.size(0) == 1
        if data.ndim == 6:
            data = data[0] # now the first dimension is number of views

        with torch.no_grad():
            logits = model(data)
            scores = logits.softmax(dim=-1).mean(dim=0)

        tot += 1
        hit1 += (scores.topk(1)[1] == labels).sum().item()
        hit5 += (scores.topk(5)[1] == labels).sum().item()

        if tot % 20 == 0:
            print(f'[Evaluation] num_samples: {tot}  '
                  f'ETA: {(datetime.now() - eval_st) / tot * (len(loader) - tot)}  '
                  f'cumulative_acc1: {hit1 / tot * 100.:.2f}%  '
                  f'cumulative_acc5: {hit5 / tot * 100.:.2f}%')

    sync_tensor = torch.LongTensor([tot, hit1, hit5]).cuda()
    dist.all_reduce(sync_tensor)
    tot, hit1, hit5 = sync_tensor.cpu().tolist()

    print(f'Accuracy on validation set: top1={hit1 / tot * 100:.2f}%, top5={hit5 / tot * 100:.2f}%')


if __name__ == '__main__': main()
