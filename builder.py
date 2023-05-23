import torch
import torch.nn as nn
from functools import partial
from torch.nn import functional as F
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch.nn.parallel as parallel

import loader
import os 
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

class UCDIR(nn.Module):

    def __init__(self, base_encoder, dim=128, K_A=65536, K_B=65536,
                 m=0.999, T=0.1, mlp=False, selfentro_temp=0.2,
                 num_cluster=None, cwcon_filterthresh=0.2,num_workers=4):

        super(UCDIR, self).__init__()

        # k_A k_B: 各自数据集的大小
        self.K_A = K_A
        self.K_B = K_B
        self.m = m
        self.T = T
        self.num_workers = num_workers

        self.selfentro_temp = selfentro_temp
        self.num_cluster = num_cluster
        self.cwcon_filterthresh = cwcon_filterthresh

        # partial: 将原函数中的某些参数设为固定值，返回一个新函数
        # SplitBatchNorm: simulate multi-gpu behavior of BatchNorm in one gpu by splitting alone the batch dimension
        norm_layer = partial(SplitBatchNorm, num_splits=2)

        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim, norm_layer=norm_layer)
        self.cluster_result = None

        if mlp:  # hack: brute-force replacement
            # a = self.encoder_q.fc.weight.shape
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        self.encoder_q = self.encoder_q.cuda()
        self.encoder_k = self.encoder_k.cuda()

        # create the queues
        # moco的队列
        self.register_buffer("queue_A", torch.randn(dim, K_A))
        # normalize使得queue_A, queue_B的所有元素都在0到1之间，使得向量在计算余弦相似度时不会受到向量长度的影响
        self.queue_A = F.normalize(self.queue_A, dim=0)
        # 队列的指针
        self.register_buffer("queue_A_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_B", torch.randn(dim, K_B))
        self.queue_B = F.normalize(self.queue_B, dim=0)
        self.register_buffer("queue_B_ptr", torch.zeros(1, dtype=torch.long))

        self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-8)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        # 参数动量更新(moco)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_singlegpu(self, keys, key_ids, domain_id):
        # 按照key_ids把keys.T也就是(dim,batch_size)复制到队列中
        if domain_id == 'A':
            self.queue_A.index_copy_(1, key_ids, keys.T)
        elif domain_id == 'B':
            self.queue_B.index_copy_(1, key_ids, keys.T)

    @torch.no_grad()
    def _batch_shuffle_singlegpu(self, x):

        idx_shuffle = torch.randperm(x.shape[0]).cuda()
        # 用 argsort 找到 shuffle 后的 tensor 中每个元素原本在原始 tensor 中的位置
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_singlegpu(self, x, idx_unshuffle):

        return x[idx_unshuffle]
    

    def forward(self, im_q_A, im_q_B, divide=None,im_k_A=None, im_id_A=None,
                im_k_B=None, im_id_B=None, is_eval=False,
                criterion=None, all_probs=None, all_ids=None, divide_all_losses=None,
                traindirA=None, traindirB=None):
        if divide == None:
            # if is_eval is not True:
            #     im_id_A = im_id_A.cuda(0)
            #     im_id_B = im_id_B.cuda(0)
            # a = self.cluster_result
            im_q = torch.cat([im_q_A, im_q_B], dim=0)

            if is_eval:
                k = self.encoder_k(im_q)
                k = F.normalize(k, dim=1)

                k_A, k_B = torch.split(k, im_q_A.shape[0])
                return k_A, k_B

            # 提取128维的特征
            q = self.encoder_q(im_q)
            q = F.normalize(q, dim=1)

            # q_A的特征和q_B的特征
            q_A, q_B = torch.split(q, im_q_A.shape[0])

            im_k = torch.cat([im_k_A, im_k_B], dim=0)

            with torch.no_grad():
                self._momentum_update_key_encoder()

                im_k, idx_unshuffle = self._batch_shuffle_singlegpu(im_k)

                k = self.encoder_k(im_k)
                k = F.normalize(k, dim=1)

                k = self._batch_unshuffle_singlegpu(k, idx_unshuffle)

                k_A, k_B = torch.split(k, im_k_A.shape[0])

            # 把im_id_A/B的特征换成k_A/k_B
            self._dequeue_and_enqueue_singlegpu(k_A, im_id_A, 'A')
            self._dequeue_and_enqueue_singlegpu(k_B, im_id_B, 'B')

            loss_instcon_A, \
            loss_instcon_B = self.instance_contrastive_loss(q_A, k_A, im_id_A,
                                                            q_B, k_B, im_id_B,
                                                            criterion)

            losses_instcon = {'domain_A': loss_instcon_A,
                              'domain_B': loss_instcon_B}

            cluster_result = self.cluster_result

            if cluster_result is not None:

                loss_cwcon_A, \
                loss_cwcon_B = self.cluster_contrastive_loss(q_A, k_A, im_id_A,
                                                             q_B, k_B, im_id_B,
                                                             cluster_result)
                
                losses_cwcon = {'domain_A': loss_cwcon_A,
                                'domain_B': loss_cwcon_B}

                losses_selfentro = self.self_entropy_loss(q_A, q_B, cluster_result)

                losses_distlogit = self.dist_of_logit_loss(q_A, q_B, cluster_result, self.num_cluster)

                return losses_instcon, q_A, q_B, losses_selfentro, losses_distlogit, losses_cwcon
            else:
                return losses_instcon, None, None, None, None, None
            
        else:
            im_q = torch.cat([im_q_A, im_q_B], dim=0)
            # 提取128维的特征
            q = self.encoder_q(im_q)
            q = F.normalize(q, dim=1)

            # q_A的特征和q_B的特征
            features1, features2 = torch.split(q, im_q_A.shape[0])
            # im_id_A: index of features im_id_B: domain_id
            loss = self.cluster_divide_loss(features1, features2,im_id_A,im_id_B)
            return loss
            

    def instance_contrastive_loss(self,
                                  q_A, k_A, im_id_A,
                                  q_B, k_B, im_id_B,
                                  criterion):

        # 将q_A和k_A按照batch维度求点积，最后求和
        # 求a的两个transformation的点积
        l_pos_A = torch.einsum('nc,nc->n', [q_A, k_A]).unsqueeze(-1)
        l_pos_B = torch.einsum('nc,nc->n', [q_B, k_B]).unsqueeze(-1)

        # q_A中的特征与队列中特征的内积
        l_all_A = torch.matmul(q_A, self.queue_A.clone().detach())
        l_all_B = torch.matmul(q_B, self.queue_B.clone().detach())

        mask_A = torch.arange(self.queue_A.shape[1]).cuda() != im_id_A[:, None]
        # a = im_id_A[:,None]
        l_neg_A = torch.masked_select(l_all_A, mask_A).reshape(q_A.shape[0], -1)

        mask_B = torch.arange(self.queue_B.shape[1]).cuda() != im_id_B[:, None]
        l_neg_B = torch.masked_select(l_all_B, mask_B).reshape(q_B.shape[0], -1)

        logits_A = torch.cat([l_pos_A, l_neg_A], dim=1)
        logits_B = torch.cat([l_pos_B, l_neg_B], dim=1)

        logits_A /= self.T
        logits_B /= self.T

        labels_A = torch.zeros(logits_A.shape[0], dtype=torch.long).cuda()
        labels_B = torch.zeros(logits_B.shape[0], dtype=torch.long).cuda()

        # 第一个位置是对的
        loss_A = criterion(logits_A, labels_A)
        loss_B = criterion(logits_B, labels_B)

        return loss_A, loss_B

    def cluster_contrastive_loss(self, q_A, k_A, im_id_A, q_B, k_B, im_id_B, cluster_result):

        all_losses = {'domain_A': [], 'domain_B': []}

        for domain_id in ['A', 'B']:
            if domain_id == 'A':
                im_id = im_id_A.cuda(0)
                q_feat = q_A
                k_feat = k_A
                queue = self.queue_A.clone().detach()
            else:
                im_id = im_id_B.cuda(0)
                q_feat = q_B
                k_feat = k_B
                queue = self.queue_B.clone().detach()

            mask = 1.0
            for n, (im2cluster, prototypes) in enumerate(zip(cluster_result['im2cluster_' + domain_id],
                                                             cluster_result['centroids_' + domain_id])):
                
                # batch中每张图片对应的聚类中心
                cor_cluster_id = im2cluster[im_id]

                # 在 PyTorch 中，contiguous() 函数用于返回一个连续的内存块，其中包含与原张量相同的数据。
                # 当对张量进行转置、切片或其他操作时，它们的底层数据可能不再是连续的。
                # 这可能会影响某些操作的性能，因为它们需要连续的内存块才能正常工作。
                # 在这种情况下，可以使用 contiguous() 函数来返回一个新的连续张量，以便进行后续操作。
                
                # 聚类中心处相同处为1
                mask *= torch.eq(cor_cluster_id.contiguous().view(-1, 1),
                                 im2cluster.contiguous().view(1, -1)).float().cuda(0)  # batch size x queue length

                # batch中每一个特征与queue中所有特征内积 / 温度
                all_score = torch.div(torch.matmul(q_feat, queue), self.T)

                exp_all_score = torch.exp(all_score)

                # re = torch.log(exp_all_score.sum(1, keepdim=True))
                # 从哪个维度sum就把哪个维度求和掉
                # 论文中公式（2）的分母部分
                log_prob = all_score - torch.log(exp_all_score.sum(1, keepdim=True))

                # t = (mask * log_prob).sum(1)
                # r = mask.sum(1) + 1e-8
                # 对相同聚类中心的实例特征内积求和 / |P(i)|
                log_prob = log_prob.cuda(0)
                mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

                # 每张图片对应的聚类中心的特征
                cor_proto = prototypes[cor_cluster_id]
                # nc,nc->n对每行进行点积
                # nc = torch.einsum('nc,nc->n', [k_feat, cor_proto])
                # 图片的另一个transform与聚类中心进行点积
                k_feat = k_feat.cuda(0)
                cor_proto = cor_proto.cuda(0)
                prototypes = prototypes.cuda(0)
                inst_pos_value = torch.exp(
                    torch.div(torch.einsum('nc,nc->n', [k_feat, cor_proto]), self.T))  # N
                # 图片的另一个transform与所有聚类中心进行点积
                inst_all_value = torch.exp(
                    torch.div(torch.einsum('nc,ck->nk', [k_feat, prototypes.T]), self.T))  # N x r
                # 聚类中心与另一个transform的特征的点积大于与所有与聚类中心特征的点积一定比例时才认为这个聚类中心是可用的
                filters = ((inst_pos_value / torch.sum(inst_all_value, dim=1)) > self.cwcon_filterthresh).float()

                filters_sum = filters.sum()

                # f = filters * mean_log_prob_pos
                loss = - (filters * mean_log_prob_pos).sum() / (filters_sum + 1e-8)

                all_losses['domain_' + domain_id].append(loss)
                # a = torch.stack(all_losses['domain_A'])

        return torch.mean(torch.stack(all_losses['domain_A'])), torch.mean(torch.stack(all_losses['domain_B']))

    def self_entropy_loss(self, q_A, q_B, cluster_result):

        losses_selfentro = {}
        for feat_domain in ['A', 'B']:
            if feat_domain == 'A':
                feat = q_A
            else:
                feat = q_B

            cross_proto_domains = ['A', 'B']
            for cross_proto_domain in cross_proto_domains:
                for n, (im2cluster, self_proto, cross_proto) in enumerate(
                        zip(cluster_result['im2cluster_' + feat_domain],
                            cluster_result['centroids_' + feat_domain],
                            cluster_result['centroids_' + cross_proto_domain])):

                    if str(self_proto.shape[0]) in self.num_cluster:

                        key_selfentro = 'feat_domain_' + feat_domain + '-proto_domain_' \
                                        + cross_proto_domain + '-cluster_' + str(cross_proto.shape[0])
                        if key_selfentro in losses_selfentro.keys():
                            losses_selfentro[key_selfentro].append(self.self_entropy_loss_onepair(feat, cross_proto))
                        else:
                            losses_selfentro[key_selfentro] = [self.self_entropy_loss_onepair(feat, cross_proto)]
        return losses_selfentro

    def self_entropy_loss_onepair(self, feat, prototype):

        # 求出对每个聚类中心的特征点积
        feat = feat.cuda(0)
        prototype = prototype.cuda(0)
        logits = torch.div(torch.matmul(feat, prototype.T), self.selfentro_temp)
        # A*B: 逐元素相乘
        self_entropy = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * F.softmax(logits, dim=1), dim=1))

        return self_entropy

    def dist_of_logit_loss(self, q_A, q_B, cluster_result, num_cluster):

        all_losses = {}

        for n, (proto_A, proto_B) in enumerate(zip(cluster_result['centroids_A'],
                                                   cluster_result['centroids_B'])):

            if str(proto_A.shape[0]) in num_cluster:
                domain_ids = ['A', 'B']

                for domain_id in domain_ids:
                    if domain_id == 'A':
                        feat = q_A
                    elif domain_id == 'B':
                        feat = q_B
                    else:
                        feat = torch.cat([q_A, q_B], dim=0)

                    loss_A_B = self.dist_of_dist_loss_onepair(feat, proto_A, proto_B)

                    key_A_B = 'feat_domain_' + domain_id + '_A_B' + '-cluster_' + str(proto_A.shape[0])
                    if key_A_B in all_losses.keys():
                        all_losses[key_A_B].append(loss_A_B.mean())
                    else:
                        all_losses[key_A_B] = [loss_A_B.mean()]

        return all_losses
    
    def cluster_divide_loss(self, features1, features2,index,domain_id):

        cluster_result = self.cluster_result
            
        # cur_cwcon_weight =1
            
        if domain_id == 'A':
            queue = self.queue_A.clone().detach()
        else:
            queue = self.queue_B.clone().detach()
            
        loss = 0
        mask = 1.0
        index = index.cuda(0)
        if len(index) == 0:
            return loss
        else:
            for n, (im2cluster, prototypes) in enumerate(zip(cluster_result['im2cluster_' + domain_id],
                                                             cluster_result['centroids_' + domain_id])):

                # close对应的聚类中心
                cor_cluster_id = im2cluster[index]
                # 聚类中心处相同处为1
                mask *= torch.eq(cor_cluster_id.contiguous().view(-1, 1),
                                 im2cluster.contiguous().view(1, -1)).float()  # batch size x queue lengthh
                mask = mask.cuda(0)
                # batch中每一个特征与queue中所有特征内积 / 温度
                all_score1 = torch.div(torch.matmul(features1, queue), self.T)
                all_score2 = torch.div(torch.matmul(features2, queue), self.T)
                exp_all_score1 = torch.exp(all_score1)
                exp_all_score2 = torch.exp(all_score2)
                # 论文中公式（2）的分母部分
                log_prob1 = all_score1 - torch.log(exp_all_score1.sum(1, keepdim=True))
                log_prob2 = all_score2 - torch.log(exp_all_score2.sum(1, keepdim=True))
                # t = (mask * log_prob).sum(1)
                # r = mask.sum(1) + 1e-8
                # 对相同聚类中心的实例特征内积求和 / |P(i)|
                log_prob1 = log_prob1.cuda(0)
                log_prob2 = log_prob2.cuda(0)
                mean_log_prob_pos1 = (mask * log_prob1).sum(1) / (mask.sum(1) + 1e-8)
                mean_log_prob_pos2 = (mask * log_prob2).sum(1) / (mask.sum(1) + 1e-8)

                if len(index) != 0:
                    loss1 = - mean_log_prob_pos1.sum() / (len(index) + 1e-8)
                    loss2 = - mean_log_prob_pos2.sum() / (len(index) + 1e-8)
                else:
                    loss1 = - mean_log_prob_pos1.sum()
                    loss2 = - mean_log_prob_pos2.sum()
                loss = (loss1 + loss2) / 2

            return loss
            
            
    
    def dist_of_dist_loss_onepair(self, feat, proto_1, proto_2):
        
        # 计算batch中聚类概率的余弦距离
        proto1_distlogits = self.dist_cal(feat, proto_1)
        proto2_distlogits = self.dist_cal(feat, proto_2)
        # cross-domain distance-of-distance L2 distance
        loss_A_B = F.pairwise_distance(proto1_distlogits, proto2_distlogits, p=2) ** 2

        return loss_A_B

    def dist_cal(self, feat, proto, temp=0.01):

        feat = feat.cuda(0)
        proto = proto.cuda(0)
        proto_logits = F.softmax(torch.matmul(feat, proto.T) / temp, dim=1)

        # normalize: 每一行进行归一化，每一行二范数都为1
        # 计算cosine distance
        # 如果希望得到类似于距离的表示，余弦距离 = 1 - 余弦相似度，其取值范围为[ 0， 2 ]，即相同的两个向量余弦距离为0
        proto_distlogits = 1.0 - torch.matmul(F.normalize(proto_logits, dim=1), F.normalize(proto_logits.T, dim=0))

        return proto_distlogits


# SplitBatchNorm: simulate multi-gpu behavior of BatchNorm in one gpu by splitting alone the batch dimension
# implementation adapted from https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py
class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape

        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)

            outcome = F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return F.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)
            


            
