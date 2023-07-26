import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
import os


def test(dataloader, model, args, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)
        gt = list(np.load(os.path.join(args.source, 'gt-colon.npy')))
        for i, (input, filename) in enumerate(dataloader):
            input = input.to(device)
            filename = filename[0].split('.npy')[0].split('/')[-1]

            input = input.squeeze(2)
            pred_temp = torch.zeros(0)
            len_num_seg = input.shape[1]

            for j in range(input.shape[1]//32+1):
                start_idx = j * 32
                end_idx = (j + 1)*32

                input_tmp = input[:, start_idx:end_idx, :]
                if input_tmp.shape[1] < 32:
                    for last in range((32-input_tmp.shape[1])):
                        input_tmp = torch.cat((input_tmp, input[:, -1, :].unsqueeze(1)), dim=1)
                x, cls_tokens, cls_prob,  scores, _, embeddings = model(input_tmp)
                embeddings = embeddings.squeeze(0)

                logits = torch.squeeze(scores, 2)
                logits = torch.mean(logits, 0) 
                sig = logits
                pred_temp = torch.cat((pred_temp, sig))
            pred = torch.cat((pred, pred_temp[:len_num_seg]))
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        
        rec_auc = auc(fpr, tpr)
        
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)

        print('auc : ' + str(rec_auc))
        print('AP : ' + str(pr_auc))
        
        return rec_auc, pr_auc

