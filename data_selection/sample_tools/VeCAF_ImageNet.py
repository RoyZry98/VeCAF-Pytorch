import json
import os
import numpy as np
import random
import argparse
import functools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import math
import utils
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from PIL import Image
from torchvision.utils import save_image
from accessory.demos.single_turn_mmnew import generate
import clip
import copy
import math
import utils
import torch.nn.functional as F
from scipy import linalg
import datetime
torch.autograd.set_detect_anomaly(True)
eps = 1e-10
infty = 1e10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class SampleModel(nn.Module):
    def __init__(self, args, features, sample_num, temperature, init, distance, balance, slice, batch_size):
        super(SampleModel, self).__init__()
        self.args = args
        self.features = features
        self.total_num = features.shape[0]
        self.temperature = temperature
        self.sample_num = sample_num
        self.balance = balance
        self.slice = slice
        if slice is None:
            self.slice = self.total_num
        self.batch_size = batch_size

        self.init = init
        self.distance = distance

        centroids = self.init_centroids()
        self.centroids = nn.Parameter(centroids).cuda()

    def init_centroids(self):
        if self.init == "random":
            sample_ids = list(range(self.total_num))
            sample_ids = random.sample(sample_ids, self.sample_num)
        elif self.init == "fps":
            dist_func = functools.partial(utils.get_distance, type=self.distance)
            sample_ids = utils.farthest_distance_sample(self.features, self.sample_num, dist_func)

        centroids = self.features[sample_ids].clone()
        return centroids

    def get_loss(self):
        if self.args.weight_dir != 'false':
            weight = torch.load(self.args.weight_dir).cuda().detach()
            
            if self.args.sigmoid == 'True':
                weight = torch.sigmoid(weight)
            if self.args.log == 'True':
                weight = torch.log(weight)
            if self.args.exp == 'True':
                weight = torch.exp(weight)
            if self.args.normalize == 'True':
                weight = F.normalize(weight, dim=0)
        
        centroids = F.normalize(self.centroids, dim=1)
        sample_ids = list(range(self.total_num))
        sample_ids = random.sample(sample_ids, self.batch_size)
        
        features = self.features[sample_ids]
        weight = weight[sample_ids]

        sample_slice_num = math.ceil(1.0 * self.sample_num / self.slice)
        batch_slice_num = math.ceil(1.0 * self.batch_size / self.slice)

        prod_exp_pos = []
        weight_pos = []
        
        pos_k = []
        for sid in range(batch_slice_num):
            start = sid * self.slice
            end = (sid + 1) * self.slice
            prod = torch.matmul(features[start: end], centroids.transpose(1, 0))  # (slice_num, k)
            prod = prod / self.temperature
            prod_exp = torch.exp(prod)
            prod_exp_pos_cur, pos_k_cur = torch.max(prod_exp, dim=1)  # (slice_num, )
            prod_exp_pos.append(prod_exp_pos_cur)
            pos_k.append(pos_k_cur)
            
            weight_cur = weight[start: end]
            weight_pos.append(weight_cur)
            
        pos_k = torch.cat(pos_k, dim=0)
        prod_exp_pos = torch.cat(prod_exp_pos, dim=0)
        weight_pos = torch.cat(weight_pos, dim=0)

        cent_prob_exp_sum = []
        for sid in range(sample_slice_num):
            start = sid * self.slice
            end = (sid + 1) * self.slice
            cent_prod = torch.matmul(centroids.detach(), centroids[start:end].transpose(1, 0))  # (k, slice_num)
            cent_prod = cent_prod / self.temperature
            cent_prod_exp = torch.exp(cent_prod)
            cent_prob_exp_sum_cur = torch.sum(cent_prod_exp, dim=0)  # (slice_num, )
            cent_prob_exp_sum.append(cent_prob_exp_sum_cur)
        cent_prob_exp_sum = torch.cat(cent_prob_exp_sum, dim=0)
        
        if self.args.weight_dir != 'false':
            prod_exp_pos = torch.mul(prod_exp_pos, weight_pos)

        J = torch.log(prod_exp_pos) - torch.log(prod_exp_pos + cent_prob_exp_sum[pos_k] * self.balance)
        J = -torch.mean(J)

        return J


def optimize_dist(features, sample_num, args):
    ensemble_num = 1
    g = 25623#int(args.percent*12811)25623
    centroids = torch.zeros(g, 768)
    centroids = nn.Parameter(centroids).cuda()
    global_delta = []
    centroids_sample = []
    for num in range(ensemble_num):
        print("sampling" + str(num))
        #  features: (n, c)
        sample_model = SampleModel(args, features, sample_num, args.temperature, args.init, args.distance, args.balance, args.slice, args.batch_size)
        sample_model = sample_model.cuda()

        optimizer = optim.Adam(sample_model.parameters(), lr=args.lr)
        if args.scheduler != "none":
            if args.scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_iter, eta_min=1e-6)
            else:
                raise NotImplementedError

        for i in range(args.max_iter):
            loss = sample_model.get_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.scheduler != "none":
                scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            print("Iter: %d, lr: %.6f, loss: %f" % (i, lr, loss.item()))

        centroid = copy.deepcopy(sample_model.centroids.detach())
        if num == 0:
            global_delta.append(centroid)
        else:
            centroids = torch.add(centroids, centroid)
            centroids_sample.append(centroid)

    # ensemble calculations
    centroids_mean = torch.div(centroids, ensemble_num-1)
    #global_delta = global_delta[0]
    global_delta = centroids_sample[0]
    delta = get_delta(centroids_sample, centroids_mean, global_delta)
    centroids = torch.mul(global_delta-centroids_mean,delta) + global_delta

    centroids = F.normalize(centroids, dim=1)
    slice = 100
    sample_slice_num = math.ceil(centroids.shape[0] / slice)
    sample_ids = set()
    # _, ids_sort = torch.sort(dist, dim=1, descending=True)
    for sid in range(sample_slice_num):
        start = sid * slice
        end = min((sid + 1) * slice, centroids.shape[0])
        dist = torch.matmul(centroids[start:end], features.transpose(1, 0))  # (slice_num, n)
        _, ids_sort = torch.sort(dist, dim=1, descending=True)
        for i in range(ids_sort.shape[0]):
            for j in range(ids_sort.shape[1]):
                if ids_sort[i, j].item() not in sample_ids:
                    sample_ids.add(ids_sort[i, j].item())
                    break

    print(len(sample_ids))
    sample_ids = list(sample_ids)
    return sample_ids

def get_delta(samples, means, glob_weight):
    dim = len(samples[0].shape)
    cov_matrix = None
    if dim == 1:  # no inverse
        return glob_weight-means
    for s in range(len(samples)):
        input_vec = samples[s]
        if dim == 3:
            a, b, c = input_vec.size()
            input_vec = input_vec.view(a, b)
            means = means.view(a, b)
            glob_weight = glob_weight.view(a, b)
        x = input_vec
        if cov_matrix == None:
            cov_matrix = torch.matmul(x.T, x)
        else:
            cov_matrix += torch.matmul(x.T, x)
    cov_matrix = torch.div(cov_matrix, len(samples))  # E(theta theta^T)
    # E(theta theta^T) - E(theta)^2
    cov_matrix -= torch.matmul(means.T, means)
    try:
        inverse_cov = torch.linalg.inv(cov_matrix)
    except: 
        inverse_cov = torch.linalg.pinv(cov_matrix)
        
    delta = torch.matmul(glob_weight-means, inverse_cov)
    return delta.reshape(samples[0].shape)

def aggregate_att(w_clients, w_server, stepsize, metric, dp):
    w_next = copy.deepcopy(w_server)
    w_clients = w_clients
    w_server = w_server
    att, att_mat = {}, {}
    for k in range(len(w_server)):
        w_next[k] = torch.zeros_like(w_server[k]).cpu()
        att[k] = torch.zeros(len(w_clients)).cpu()
    for k in range(len(w_server)):
        # for i in range(0, len(w_clients)):
        att[k] = torch.from_numpy(np.array(linalg.norm(w_server[k]-w_clients[k], ord=metric)))
    for k in range(len(w_server)):
        att[k] = F.softmax(att[k], dim=0)
    for k in range(len(w_server)):
        att_weight = torch.zeros_like(w_server[k])
        # for i in range(0, len(w_clients)):
        att_weight += torch.mul(w_server[k]-w_clients[k], att[k])
        w_next[k] = w_server[k] - torch.mul(att_weight, stepsize) + torch.mul(torch.randn(w_server[k].shape), dp)
    return w_next

def main(args):
    all_text_features = []
    q = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-L/14")
    input = np.load(args.feature_path)
    features, _ = input[:, :-1], input[:, -1]
    features = torch.Tensor(features).cuda()
    total_num = features.shape[0]
    sample_num = int(total_num)
    sample_num = int(total_num * args.percent * 0.01)
    print("start")
    if args.filename is None:
        name = args.feature_path.split("/")[-1]
        name = name[:-4]
        if args.balance != 1:
            args.filename = name + "_ActiveFT_%s_temp_%.2f_lr_%f_scheduler_%s_br_%.2f_iter_%d_sampleNum_%d.json" % (
                args.distance, args.temperature, args.lr, args.scheduler, args.balance, args.max_iter, sample_num)
        else:
            args.filename = name + "_ActiveFT_%s_temp_%.2f_lr_%f_scheduler_%s_iter_%d_sampleNum_%d_ensemble_boundary.json" % (
                args.distance, args.temperature, args.lr, args.scheduler, args.max_iter, sample_num)
                
    output_path = os.path.join(args.output_dir, args.filename) 

    # only when loop > 0, conduct the embedding augmentation
    if args.loop > 0:
        # with open("/data/home/zhangrongyu/code/ActiveFT/data_selection/features/ImageNet_dino_base_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_100_sampleNum_12811_ensemble_boundary.json") as file1:
        #   sample_ids = json.load(file1)
        start = 0
        end = 1281165
        sample_ids = range(1281167)[start:end]
        datasetzero = datasets.ImageFolder("/data/zqz/data/ImageNet/train/")
        dataset = torch.utils.data.Subset(datasetzero, sample_ids)
        print("load done")

        indexes = []######
        existed_ = []
        existed_captions = {}
        selected_id = []
        temp = []
        with open("/data/home/zhangrongyu/code/ActiveFT/data_selection/index_new.json") as q1:
            indexes = json.load(q1)
        indexset = set(indexes)
        with open("/data/home/zhangrongyu/code/ActiveFT/data_selection/together_loop.json") as q2:
            existed_ = json.load(q2)
        print(len(existed_))
        for caps in existed_:
            existed_captions[str(caps["index"])] = caps######
        for item in dataset:
            # if sample caption already exists in the json file 
            if sample_ids[q] in indexset:######
                print("caption existed!")
                caption = existed_captions[str(sample_ids[q])].get('caption')
                if len(caption.split()) > 50:
                    caption = ' '.join(caption.split()[:50])
                # project into clip embedding 
                text = clip.tokenize(caption).to(device)
                with torch.no_grad():
                    text_features = clip_model.encode_text(text)
                all_text_features.append(text_features[0].cpu())

                selected_id.append(sample_ids[q])
                q += 1
                continue######
            # if sample caption not exists, add it into the json file
            with open("/data/home/zhangrongyu/code/ActiveFT/data_selection/imagenet_class_index.json") as gr:
                imagelabel = json.load(gr)
                raw_image = item[0]
                imagenet_class = imagelabel[str(item[1])][1]
                raw_image.save(str("/data/home/zhangrongyu/code/ActiveFT/data_selection/sample_tools/temp_img/"+str(sample_ids[q])+".jpg"), quality=95)
                img_path = "/data/home/zhangrongyu/code/ActiveFT/data_selection/sample_tools/temp_img/" + str(sample_ids[q])+".jpg"
        
            # get image captions
            prompt = "Please describe the image within one sentence. The description must contain the keyword " + imagenet_class + "."
            # print(prompt)
            caption = generate(
                img_path=img_path,
                prompt=prompt,
                question_input=None,
                max_gen_len=70,
                gen_t=0.1, top_p=0.75)

            # project into clip embedding
            if len(caption.split()) > 70:
                caption = ' '.join(caption.split()[:70])
            text = clip.tokenize(caption).to(device)
            with torch.no_grad():
                text_features = clip_model.encode_text(text)
            all_text_features.append(text_features[0].cpu())

            new_o = {}
            new_o["caption"] = caption
            new_o["index"] = sample_ids[q]
            new_o["label"] = item[1]
            existed_.append(new_o)
            temp.append(new_o)
            indexes.append(sample_ids[q])
            selected_id.append(sample_ids[q])
            q += 1
            print(q)

        all_image_features = []
        # cross-attentive embedding augmentation
        for i in selected_id:
            all_image_features.append(features[i].cpu())
        aug_features = aggregate_att(all_text_features, all_image_features, 1.2, 2, 0.001)
        for i in range(len(selected_id)):
            features[selected_id[i]] = torch.Tensor(aug_features[i]).cuda()

    t1 = datetime.datetime.now()
    features = F.normalize(features, dim=1)
    sample_ids = optimize_dist(features, sample_num, args)
    sample_ids.sort()
    print("out normalize")
    t2 = datetime.datetime.now()
    print(t2 - t1)
    print(len(existed_))

    with open(output_path, "w") as file:
        json.dump(sample_ids, file)

    np.save('/data/home/zhangrongyu/code/ActiveFT/data_selection/features/text_feature.py', all_text_features)

    with open("/data/home/zhangrongyu/code/ActiveFT/data_selection/caption_complete.json", "w") as file:
        json.dump(existed_, file, cls=NpEncoder)
    with open("/data/home/zhangrongyu/code/ActiveFT/data_selection/together_indexes_loop.json", "w") as file:
        json.dump(indexes, file, cls=NpEncoder)
    with open("/data/home/zhangrongyu/code/ActiveFT/data_selection/captions/caption_together.json", "w") as file:
        json.dump(temp, file, cls=NpEncoder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize extracted features')
    parser.add_argument('--feature_path', default='/data/home/zhangrongyu/code/ActiveFT/data_selection/features/ImageNet_dino_base_train.npy', type=str, help='path of saved features')
    parser.add_argument('--output_dir', default='/data/home/zhangrongyu/code/ActiveFT/data_selection/new_features', type=str, help='dir to save the visualization')
    parser.add_argument('--filename', default=None, type=str, help='filename of the visualization')
    parser.add_argument('--temperature', default=0.07, type=float, help='temperature for softmax')
    parser.add_argument('--threshold', default=0.0001, type=float, help='convergence threshold')
    parser.add_argument('--max_iter', default=100, type=int, help='max iterations')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--percent', default=2, type=float, help='sample percent')
    parser.add_argument('--init', default='random', type=str, choices=['random', 'fps'])
    parser.add_argument('--distance', default='euclidean', type=str, help='euclidean or cosine')
    parser.add_argument('--scheduler', default='none', type=str, help='scheduler')
    parser.add_argument('--balance', default=1.0, type=float, help='balance ratio')
    parser.add_argument('--batch_size', default=1000, type=int, help='batch size for SGD')
    
    parser.add_argument('--weight_dir', default='/data/home/zhangrongyu/code/ActiveFT/output_boundary/ImageNet_output/loss.pt', type=str, help='dir to save the weights')
    parser.add_argument('--normalize', default='False', type=str, help='log or not')
    parser.add_argument('--exp', default='False', type=str, help='log or not')
    parser.add_argument('--log', default='False', type=str, help='log or not')
    parser.add_argument('--sigmoid', default='False', type=str, help='log or not')
    
    parser.add_argument('--slice', default=None, type=int, help='size of slice to save memory')
    parser.add_argument('--loop', default=1, type=int, help='loop number')
    args = parser.parse_args()
    main(args)