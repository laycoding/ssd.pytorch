# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp, decode, nms


class PrecisionLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, top_k, encode_target, nms_thresh, conf_thresh,
                 use_gpu=True):
        super(PrecisionLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.variance = cfg['variance']
        self.top_k = top_k
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.nms_thresh = nms_thresh
        self.softmax = nn.Softmax(dim=-1)
        self.conf_thresh = conf_thresh

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        # confused here, why stuck at loc_data size 1
        num_priors = (priors.size(0))
#         prior_data = priors.view(1, num_priors, 4)
#         print(prior_data.size())
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        # [num, num_priors, 4]
        conf_t = torch.LongTensor(num, num_priors) 
        # [num_priors] top class label for each prior
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        
        conf_preds = self.softmax(conf_data.view(num, num_priors,
                                    self.num_classes))
        # print(conf_preds.max()) 0.9
        conf_preds_trans = conf_preds.transpose(2,1)
        # [num, num_classes, num_priors]
        conf_p = torch.zeros(num, num_priors, num_classes)
        # [num, num_priors, num_classes]
        loc_p = torch.zeros(num, num_priors, 4)
        # Decode predictions into bboxes
        for i in range(num):
#             print("shape of loc")
#             print(loc_data[i].size())
#             print("shape of prior")
#             print(priors.size())            
            decoded_boxes = decode(loc_data[i], priors, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds_trans[i].clone()
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                # fliter low conf predictions
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = Variable(decoded_boxes[l_mask].view(-1, 4), requires_grad=False)
                # idx of highest scoring and non-overlapping boxes per class
                # boxes [num_priors(has been flitered), 4] location preds for i'th image
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                conf_p[i, ids, cl] = conf_preds[i, ids, cl] # [num, num_priors, num_classes]
                loc_p[i, ids] = loc_data[i, ids] # [num, num_priors, 4]
        # check each result if match the ground truth
        effect_conf = conf_p != 0
        effect_loc = loc_p != 0
        num_effect = effect_loc.view(num, -1).sum(dim=1, keepdim=True)
        # num_effect [num ,1]
        conf_t = conf_t.view(num, num_priors, -1).expand_as(conf_p)
        # [num_priors, num_classes]
        loss_l = F.smooth_l1_loss(loc_p[effect_loc], loc_t[effect_loc], size_average=False)
        loss_c = F.cross_entropy(conf_p[effect_conf], conf_t[effect_conf], size_average=False)
        print(loss_c.size())

        N = num_effect.data.sum()
        loss_l /= N.float()
        loss_c /= N.float()
        return loss_l, loss_c
