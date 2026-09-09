import torch
import torch.nn.functional as F
import numpy as np

def evaluate_model(model, test_loader, masks, device="cuda"):
    mse_mask, be_mask, ce_mask, alg_mask = masks
    c_lengths = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 32, 6]

    model.eval()
    model.to(device)

    total_samples = 0

    cont_mae_sum = cont_mse_sum = cont_correct = cont_count = 0
    bin_correct = bin_total = bin_tp = bin_fp = bin_fn = 0
    cat_correct = cat_total = cat_top5_correct = 0

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            recon, _, _ = model(x)

            batch_size = x.size(0)
            total_samples += batch_size

            if mse_mask.any():
                p, t = recon[:, mse_mask], x[:, mse_mask]
                cont_mae_sum += torch.abs(p - t).sum().item()
                cont_mse_sum += ((p - t) ** 2).sum().item()
                cont_correct += (torch.abs(p - t) < 0.05).sum().item()
                cont_count += p.numel()

            if be_mask.any():
                p, t = torch.sigmoid(recon[:, be_mask]) > 0.5, x[:, be_mask]
                bin_correct += (p == t).sum().item()
                bin_total += t.numel()
                bin_tp += ((p == 1) & (t == 1)).sum().item()
                bin_fp += ((p == 1) & (t == 0)).sum().item()
                bin_fn += ((p == 0) & (t == 1)).sum().item()

            cat_mask = ce_mask | alg_mask
            if cat_mask.any():
                pred_cat = recon[:, cat_mask]
                true_cat = x[:, cat_mask]
                idx = 0
                for length in c_lengths:
                    if idx + length > pred_cat.size(1):
                        break
                    p_group, t_group = pred_cat[:, idx:idx+length], true_cat[:, idx:idx+length]
                    p_cls, t_cls = torch.argmax(p_group, dim=1), torch.argmax(t_group, dim=1)
                    cat_correct += (p_cls == t_cls).sum().item()
                    cat_total += batch_size
                    if length >= 5:
                        _, top5 = torch.topk(p_group, k=5, dim=1)
                        cat_top5_correct += (top5 == t_cls.unsqueeze(1)).any(dim=1).sum().item()
                    else:
                        cat_top5_correct += (p_cls == t_cls).sum().item()
                    idx += length

    results = {'samples': total_samples}

    if cont_count > 0:
        results['continuous'] = {
            'mae': cont_mae_sum / cont_count,
            'rmse': np.sqrt(cont_mse_sum / cont_count),
            'acc@5%': cont_correct / cont_count,
            'count': mse_mask.sum().item()
        }

    if bin_total > 0:
        p, r = bin_tp / (bin_tp + bin_fp + 1e-8), bin_tp / (bin_tp + bin_fn + 1e-8)
        results['binary'] = {
            'acc': bin_correct / bin_total,
            'f1': 2 * p * r / (p + r + 1e-8),
            'precision': p,
            'recall': r,
            'count': be_mask.sum().item()
        }

    if cat_total > 0:
        results['categorical'] = {
            'top1': cat_correct / cat_total,
            'top5': cat_top5_correct / cat_total,
            'count': (ce_mask.sum() + alg_mask.sum()).item()
        }

    scores, weights = [], []
    if 'continuous' in results:
        scores.append(max(0, 1 - results['continuous']['mae']))
        weights.append(results['continuous']['count'])
    if 'binary' in results:
        scores.append(results['binary']['acc'])
        weights.append(results['binary']['count'])
    if 'categorical' in results:
        scores.append(results['categorical']['top1'])
        weights.append(results['categorical']['count'])

    if weights:
        results['overall'] = np.average(scores, weights=weights)

    return results


def print_results(r):
    print(f"Samples: {r['samples']}")
    print(f"Overall: {r.get('overall', 0):.4f}")
    if 'continuous' in r:
        c = r['continuous']
        print(f"Continuous ({c['count']}): MAE={c['mae']:.4f} RMSE={c['rmse']:.4f} Acc@5%={c['acc@5%']:.4f}")
    if 'binary' in r:
        b = r['binary']
        print(f"Binary ({b['count']}): Acc={b['acc']:.4f} F1={b['f1']:.4f} P={b['precision']:.4f} R={b['recall']:.4f}")
    if 'categorical' in r:
        c = r['categorical']
        print(f"Categorical ({c['count']}): Top-1={c['top1']:.4f} Top-5={c['top5']:.4f}")