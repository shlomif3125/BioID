from dataclasses import dataclass
from typing import Optional
from typing_extensions import TypedDict
from torch import nn
import torch
from torch.nn.functional import normalize
import torch.nn.functional as F
from misc.utils import centroids_init
from torchmetrics import F1Score, Precision, Recall, AUROC


class LRSchedulerConfig(TypedDict):
    lr_class: torch.optim.lr_scheduler._LRScheduler
    lr_scheduler_kwargs: dict
    lr_scheduler_has_monitor: bool
    lr_initial_value: int


@dataclass
class NamedWeightedLossClass:
    name: str    
    weight: float
    func: Optional[nn.Module] = None
    is_loss: bool = True
    is_metric: bool = True
    optimizer: Optional[torch.optim.Optimizer] = None
    lr_scheduler_config: Optional[LRSchedulerConfig] = None
    
    
class SubSetCELoss(nn.Module):
    def __init__(self, extra_net, label_ind=0):
        super().__init__()
        self.extra_net = extra_net
        self.loss = nn.CrossEntropyLoss()
        self.label_ind = label_ind
    
    def forward(self, emb, y):
        out = self.extra_net(emb)
        return self.loss(out, y[self.label_ind])
    

# class ContrastiveLoss(nn.Module):
#     def __init__(
#         self,
#         pos_ratio: float = 1.0,
#         neg_ratio: float = 1.0,
#     ):
#         super().__init__()
#         self.pos_ratio = pos_ratio
#         self.neg_ratio = neg_ratio

#     def forward(self, x, subject):
#         subject_mat = subject.repeat((subject.shape[0], 1))
#         subject_mat = (subject_mat == subject_mat.T).float()
#         subject_mat = subject_mat.triu(diagonal=1)
#         pairwise_cosine_similarity = cosine_similarity(x[None, :], x[:, None], dim=-1)

#         pos_loss = (
#             1 - (pairwise_cosine_similarity * subject_mat).sum() / subject_mat.sum()
#         )
#         neg_loss = (pairwise_cosine_similarity * (1 - subject_mat)).sum() / (
#             1 - subject_mat
#         ).sum()

#         return (
#             pos_loss * self.pos_ratio
#             + neg_loss * self.neg_ratio
#         )    
    

# class CentroidsControlledArcFaceLoss(nn.Module):
#     def __init__(self, num_classes, embedding_size, margin=28.6, scale=64, init_centroids=True, train_centroids=False, *args, **kwargs):        
#         super().__init__()
#         self.loss = losses.ArcFaceLoss(num_classes, embedding_size, margin, scale, *args, **kwargs)
                
#         if init_centroids:
#             W = centroids_init(num_classes, embedding_size).T
#             self.loss.load_state_dict({'W': W})
        
#         if not train_centroids:
#             self.loss.W.requires_grad = False
            
#     def forward(self, emb, y):
#         return self.loss(emb, y)


class Top1AccForSubset(nn.Module):
    def __init__(self, extra_net, label_ind=0):
        super().__init__()
        self.extra_net = extra_net
        self.label_ind = label_ind
    
    def forward(self, emb, y):
        out = self.extra_net(emb)
        return (out.argmax(1) == y[self.label_ind]).float().mean()        


class F1ScoreForSubset(nn.Module):
    def __init__(self, extra_net, label_ind=0):
        super().__init__()
        self.extra_net = extra_net
        self.label_ind = label_ind
        self.sm = nn.Softmax(1)
        self.f1score = F1Score(task='multiclass', average='macro', num_classes=3)
    
    def forward(self, emb, y):
        out = self.extra_net(emb)
        probs = self.sm(out)
        return self.f1score(probs, y[self.label_ind])


class PrecisionForSubset(nn.Module):
    def __init__(self, extra_net, label_ind=0):
        super().__init__()
        self.extra_net = extra_net
        self.label_ind = label_ind
        self.sm = nn.Softmax(1)
        self.precision = Precision(task="multiclass", average='macro', num_classes=3)
    
    def forward(self, emb, y):
        out = self.extra_net(emb)
        probs = self.sm(out)
        return self.precision(probs, y[self.label_ind])
    
class RecallForSubset(nn.Module):
    def __init__(self, extra_net, label_ind=0):
        super().__init__()
        self.extra_net = extra_net
        self.label_ind = label_ind
        self.sm = nn.Softmax(1)
        self.recall = Recall(task="multiclass", average='macro', num_classes=3)
    
    def forward(self, emb, y):
        out = self.extra_net(emb)
        probs = self.sm(out)
        return self.recall(probs, y[self.label_ind])
    
class ROCAUCForSubset(nn.Module):
    def __init__(self, extra_net, label_ind=0):
        super().__init__()
        self.extra_net = extra_net
        self.label_ind = label_ind
        self.sm = nn.Softmax(1)
        self.roc_auc = AUROC(task="multiclass", average='macro', num_classes=3)
    
    def forward(self, emb, y):
        out = self.extra_net(emb)
        probs = self.sm(out)
        return self.roc_auc(probs, y[self.label_ind])


class WithinSubsetCELoss(nn.Module):
    def __init__(self, extra_net, label_ind=0):
        super().__init__()
        self.extra_net = extra_net
        self.label_ind = label_ind
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, emb, y):
        y = y[self.label_ind]
        out = self.extra_net(emb)[y != 0, 1:]
        return self.loss(out, y[y != 0] - 1)


class OneVsAllCELoss(nn.Module):
    def __init__(self, extra_net, label_ind=0, singled_out_class=0):
        super().__init__()
        self.extra_net = extra_net
        self.label_ind = label_ind
        self.singled_out_class = singled_out_class
        self.sm = nn.Softmax(1)
        self.loss = nn.BCELoss()
    
    def forward(self, emb, y):
        y = y[self.label_ind]
        y = (y == self.singled_out_class).to(torch.float32)
        out = self.extra_net(emb)
        probs = self.sm(out)
        # class_of_interest = self.singled_out_class
        class_of_interest_scores = probs[:, self.singled_out_class]
        # non_class_of_interest_scores = sm[:, :class_of_interest].sum(1) + sm[:, class_of_interest + 1:].sum(1)
        # one_vs_all_scores = torch.stack([non_class_of_interest_scores, class_of_interest_scores], 1)
        return self.loss(class_of_interest_scores, y)


class MeanPerSensorTop1ArcFaceAcc(nn.Module):
    def __init__(self, meta_data_and_embeddings_df):
        super().__init__()
        self.meta_data_and_embeddings_df = meta_data_and_embeddings_df
    
    @staticmethod
    def l2_normalize(vecs, dim=1):
        return normalize(vecs, dim=dim)
    
    def get_class_est(self, emb):
        return (self.l2_normalize(emb) @ self.centroids).argmax(1)
    
    def get_per_sensor_acc(self):
        df = self.meta_data_and_embeddings_df
        subjects_and_centroids = df.groupby('subject').embedding.apply(lambda x: torch.stack(x).mean(0))
        subjects, centroids = zip(*[(k, v) for k, v in subjects_and_centroids.to_dict().items()])
        centroids = torch.stack(centroids).T
        centroids = self.l2_normalize(centroids, 0)
        y = torch.tensor(df['subject'].apply(lambda x: subjects.index(x)).to_list())
        emb = torch.tensor(torch.stack(df['embedding']))
        return (self.get_class_est(emb) == y).numpy().mean()
    

class Top1AccForArcFace(nn.Module):
    def __init__(self, centroids, label_ind=0):
        super().__init__()
        self.centroids = self.l2_normalize(centroids, 0)
        self.label_ind = label_ind
        
    @staticmethod
    def l2_normalize(vecs, dim=1):
        return normalize(vecs, dim=dim)
    
    def get_class_est(self, emb):
        centroids = self.centroids.to(emb.device)
        return (self.l2_normalize(emb) @ centroids).argmax(1)
    
    def get_top1_acc(self, model_out, y):
        return (self.get_class_est(model_out).argmax(1) == y[self.label_ind]).float().mean()
        
    def forward(self, emb, y):
        class_est = self.get_class_est(emb)
        return (class_est == y[self.label_ind]).float().mean()     
    

# class ElasticArcFace(nn.Module):
    # def __init__(self, num_classes, embedding_size, s=64.0, m=0.50, std=0.0125, plus=False):
    #     super().__init__()
    #     self.num_classes = num_classes
    #     self.embedding_size = embedding_size
    #     self.s = s
    #     self.m = m
    #     self.kernel = nn.Parameter(torch.FloatTensor(embedding_size, num_classes))
    #     nn.init.normal_(self.kernel, std=0.01)
    #     self.std = std
    #     self.plus = plus
     
    # @staticmethod   
    # def l2_norm(input, axis=1):
    #     norm = torch.norm(input, 2, axis, True)
    #     output = torch.div(input, norm)

    #     return output
        
    # def forward(self, emb, y):
    #     embbedings = self.l2_norm(emb, axis=1)
    #     kernel_norm = self.l2_norm(self.kernel, axis=0)
    #     cos_theta = torch.mm(embbedings, kernel_norm)
    #     cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
    #     index = torch.where(y != -1)[0]
    #     m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
    #     margin = torch.normal(mean=self.m, std=self.std, size=y[index, None].size(), device=cos_theta.device) # Fast converge .clamp(self.m-self.std, self.m+self.std)
    #     if self.plus:
    #         with torch.no_grad():
    #             distmat = cos_theta[index, y.view(-1)].detach().clone()
    #             _, idicate_cosie = torch.sort(distmat, dim=0, descending=True)
    #             margin, _ = torch.sort(margin, dim=0)
    #         m_hot.scatter_(1, y[index, None], margin[idicate_cosie])
    #     else:
    #         m_hot.scatter_(1, y[index, None], margin)
    #     cos_theta.acos_()
    #     cos_theta[index] += m_hot
    #     cos_theta.cos_().mul_(self.s)
    #     return cos_theta


class ArcFaceForUNPG(nn.Module):    
    def __init__(self, 
                 num_classes, embedding_size,
                 margin=0.1, scale=64.0,
                 init_centroids=True, train_centroids=False,
                 label_ind=0):
        super(ArcFaceForUNPG, self).__init__()
        self.in_feature = embedding_size
        self.out_feature = num_classes
        self.s = scale
        self.m = margin
        
        if init_centroids:
            W = centroids_init(num_classes, embedding_size).T
            self.weight = W
        else:
            self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_size))        
            nn.init.xavier_uniform_(self.weight)
        
        if not train_centroids:
            self.weight.requires_grad = False        
        
        self.weight = self.weight.to(dtype=torch.float32)
        self.label_ind = label_ind
        
    def forward(self, x, label):   
        x, weight = F.normalize(x), F.normalize(self.weight, dim=0)
        weight = weight.to(x.device)
        cosine = F.linear(x, weight.T)        
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
                
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_()
                
        return cosine


class UNPG(nn.Module):
    def __init__(self, num_classes, embedding_size, 
                 margin=28.6, scale=64, init_centroids=True, train_centroids=False, 
                 label_ind=0,    
                 wisk: float = 1.0):
        super().__init__()
        self.arcface = ArcFaceForUNPG(num_classes, embedding_size, margin, scale, 
                                      init_centroids, train_centroids)
        self.s = scale
        self.wisk = wisk
        self.ce = nn.CrossEntropyLoss()
        self.label_ind = label_ind
        
    @staticmethod
    def box_and_whisker_algorithm(similarities, wisk):        
        l = similarities.size(0)
        sorted_x = torch.sort(input=similarities, descending=False)[0]
        
        lower_quartile = sorted_x[int(0.25 * l)]
        upper_quartile = sorted_x[int(0.75 * l)]
                
        IQR = (upper_quartile - lower_quartile)        
        minimum = lower_quartile - wisk * IQR        
        maximum = upper_quartile + wisk * IQR
        mask = torch.logical_and(sorted_x <= maximum, sorted_x >= minimum)
        sn_prime = sorted_x[mask]
        return sn_prime
 
    @staticmethod
    def convert_label_to_similarity(normed_feature, label):
        similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
        label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

        positive_matrix = label_matrix.triu(diagonal=1)
        negative_matrix = label_matrix.logical_not().triu(diagonal=1)

        similarity_matrix = similarity_matrix.view(-1)
        positive_matrix = positive_matrix.view(-1)
        negative_matrix = negative_matrix.view(-1)
        return (similarity_matrix[positive_matrix], similarity_matrix[negative_matrix])
    
    def forward(self, x, label):
        label = label[self.label_ind]
        cosine = self.arcface(x, label) 
        norm_x = F.normalize(x)
        _, sn = self.convert_label_to_similarity(norm_x, label)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        
        aux_sn = self.box_and_whisker_algorithm(sn, wisk=self.wisk)                    
        aux_sn = torch.unsqueeze(aux_sn, 0)
        
        one = torch.ones(cosine.size(0), device=cosine.device).unsqueeze(1)        
        aux_sn = one * aux_sn        
        cosine = torch.cat([cosine, aux_sn], dim=1) 
        loss = self.ce(self.s * cosine, label)
        return loss
