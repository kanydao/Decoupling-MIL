from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record
from Adapter_decouple import get_model_transformer
from dataset import Dataset
from train_transformer import train
from test_transformer import test
import option
from tqdm import tqdm
from config import *
import os


freeze = True

# pretrained_path = "/mnt/889cdd89-1094-48ae-b221-146ffe543605/zht/weakly-polyp/final-multi/rtfm105-i3d.pkl"

if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)

    train_nloader = DataLoader(Dataset(args.source, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args.source, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args.source, test_mode=True),
                             batch_size=1, shuffle=False,
                             num_workers=args.workers, pin_memory=False)
    
    gpus = args.gpus
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    pretrained_path = args.pretrained_model
    model = get_model_transformer()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model.to(device), device_ids=gpus, output_device=gpus[0])
    weights = torch.load(pretrained_path, map_location=device)

    model.load_state_dict(weights, strict=False)


    if freeze:
        for name, para in model.named_parameters():
            if "decoder" in name:
                if "adapter" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))
            else:
                print("training {}".format(name))

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.Adam(model.parameters(),
                           lr=config.lr[0], weight_decay=0.0005)

    test_info = {"epoch": [], "test_AUC": [], "test_AP": []}
    best_AUC = -1
    output_path = args.output 

    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, device)

        if step % 5 == 0:
            auc, ap = test(test_loader, model, args, device)
            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)
            test_info["test_AP"].append(ap)

            if test_info["test_AUC"][-1] > best_AUC:
                best_AUC = test_info["test_AUC"][-1]
                torch.save(model.state_dict(), output_path + args.model_name + '{}-i3d.pkl'.format(step))
                save_best_record(test_info, os.path.join(output_path, '{}-step-AUC.txt'.format(step)))
    torch.save(model.state_dict(), os.path.join(output_path, args.model_name + 'final.pkl'))

