import os
import torch
import yaml #用于读取YAML格式的配置文件。
import argparse
from core.dataset import MMDataLoader
from core.losses import MultimodalLoss
from core.scheduler import get_scheduler
from core.utils import setup_seed, get_best_results
from models.lnln import build_model
from core.metric import MetricsTop
import warnings
import time
#git config --global http.postBuffer 524288000

print(f"Current Process ID: {os.getpid()}")

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
USE_CUDA = torch.cuda.is_available()
print('torch.cuda.is_available()',torch.cuda.is_available())
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

parser = argparse.ArgumentParser() 
parser.add_argument('--config_file', type=str, default='') 
parser.add_argument('--seed', type=int, default=-1) 
opt = parser.parse_args()
print('opt',opt)

def main():
    best_valid_results, best_test_results = {}, {}

    #sims
    config_file = 'configs/train_sims.yaml' if opt.config_file == '' else opt.config_file

    with open(config_file) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    print('arags: ',args)

    seed = args['base']['seed'] if opt.seed == -1 else opt.seed
    setup_seed(seed)
    print("seed is fixed to {}".format(seed))

    ckpt_root = os.path.join('ckpt', args['dataset']['datasetName'])
    if not os.path.exists(ckpt_root):
        os.makedirs(ckpt_root)
    print("ckpt root :", ckpt_root)

    #构建模型
    print('构建模型')
    model = build_model(args).to(device)

    #加载数据
    print('加载数据')
    dataLoader = MMDataLoader(args)

    #定义优化器、学习率调度器和损失函数
    #使用AdamW优化器，根据配置文件中的学习率和权重衰减参数进行初始化。
    print('定义优化器、学习率调度器和损失函数')
    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args['base']['lr'],
                                 weight_decay=args['base']['weight_decay'])

    #根据配置文件中的参数获取学习率调度器。
    scheduler_warmup = get_scheduler(optimizer, args)

    #根据配置文件中的参数初始化多模态损失函数。
    loss_fn = MultimodalLoss(args)

    #定义评估指标
    metrics = MetricsTop(train_mode = args['base']['train_mode']).getMetics(args['dataset']['datasetName'])

    for epoch in range(1, args['base']['n_epochs']+1):
        start_time=time.time()

        print('epoch is : ',epoch,'train',train)
        train(model, dataLoader['train'], optimizer, loss_fn, epoch, metrics)

        if args['base']['do_validation']:
            valid_results = evaluate(model, dataLoader['valid'], loss_fn, epoch, metrics)
            best_valid_results = get_best_results(valid_results, best_valid_results, epoch, model, optimizer, ckpt_root, seed, save_best_model=False)
            print(f'Current Best Valid Results: {best_valid_results}')

        test_results = evaluate(model, dataLoader['test'], loss_fn, epoch, metrics)
        best_test_results = get_best_results(test_results, best_test_results, epoch, model, optimizer, ckpt_root, seed, save_best_model=True)
        print(f'Current Best Test Results: {best_test_results}\n')
        scheduler_warmup.step()

        end_time=time.time()
        elapsed_time = end_time - start_time
        print(f'Epoch {epoch} completed in {elapsed_time:.2f} seconds\n') #一次36.55s 200次得跑2h


def train(model, train_loader, optimizer, loss_fn, epoch, metrics):
    y_pred, y_true = [], []
    loss_dict = {}

    model.train()
    for cur_iter, data in enumerate(train_loader):
        #完整数据
        complete_input = (data['vision'].to(device), data['audio'].to(device), data['text'].to(device))
        #设置了缺失的数据
        incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))

        sentiment_labels = data['labels']['M'].to(device)

        #语言模态的完整性标签值=1-语言模态的缺失率标签值，用于完备性检测模块的损失函数Lcc
        completeness_labels = 1. - data['labels']['missing_rate_l'].to(device)

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!这个有效性标签，为啥要设置成batch_size*8个1，batch_size*8个0呢？ 8为啥，1，0又是为啥呢？
        #答：这个8应该是 h_l_1（16，8，128）来的
        #在对抗学习中，effectiveness_labels 的设置是为了区分真实样本和生成的对抗样本。具体来说，这些标签用于训练一个判别器（discriminator），该判别器的任务是区分输入数据是真实的还是由生成器（generator）生成的。
        effectiveness_labels = torch.cat([torch.ones(len(sentiment_labels)*8), torch.zeros(len(sentiment_labels)*8)]).long().to(device)

        label = {'sentiment_labels': sentiment_labels, 'completeness_labels': completeness_labels, 'effectiveness_labels': effectiveness_labels}

        #print('cur_iter:', cur_iter)
        out = model(complete_input, incomplete_input)

        #lnln完事之后计算损失函数
        loss = loss_fn(out, label)

        loss['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(out['sentiment_preds'].cpu())
        y_true.append(label['sentiment_labels'].cpu())

        if cur_iter == 0:
            for key, value in loss.items():
                loss_dict[key] = value.item()
        else:
            for key, value in loss.items():
                loss_dict[key] += value.item()

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)

    loss_dict = {key: value / (cur_iter+1) for key, value in loss_dict.items()}

    print(f'Train Loss Epoch {epoch}: {loss_dict}')
    print(f'Train Results Epoch {epoch}: {results}')




def evaluate(model, eval_loader, loss_fn, epoch, metrics):
    loss_dict = {}

    y_pred, y_true = [], []

    model.eval()
    
    for cur_iter, data in enumerate(eval_loader):
        complete_input = (None, None, None)
        incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))

        sentiment_labels = data['labels']['M'].to(device)
        completeness_labels = 1. - data['labels']['missing_rate_l'].to(device)
        effectiveness_labels = torch.cat([torch.ones(len(sentiment_labels)*8), torch.zeros(len(sentiment_labels)*8)]).long().to(device)
        label = {'sentiment_labels': sentiment_labels, 'completeness_labels': completeness_labels, 'effectiveness_labels': effectiveness_labels}
        
        with torch.no_grad():
            out = model(complete_input, incomplete_input)

        loss = loss_fn(out, label)

        y_pred.append(out['sentiment_preds'].cpu())
        y_true.append(label['sentiment_labels'].cpu())

        if cur_iter == 0:
            for key, value in loss.items():
                try:
                    loss_dict[key] = value.item()
                except:
                    loss_dict[key] = value
        else:
            for key, value in loss.items():
                try:
                    loss_dict[key] += value.item()
                except:
                    loss_dict[key] += value
    
    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)
    
    # print(f'Test Loss Epoch {epoch}: {loss_dict}')
    # print(f'Test Results Epoch {epoch}: {results}')

    return results


if __name__ == '__main__':
    main()


