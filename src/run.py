#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import argparse
from pprint import pprint

import torch.optim as optim

try:
    from utils import *
    from models import *
    from Load import *
    from loss import *
except:
    from src.utils import *
    from src.models import *
    from src.Load import *
    from src.loss import *


def load_img_features(ent_num, file_dir):
    # load images features
    if "V1" in file_dir:
        split = "norm"
        img_vec_path = "data/pkls/dbpedia_wikidata_15k_norm_GA_id_img_feature_dict.pkl"
    elif "V2" in file_dir:
        split = "dense"
        img_vec_path = "data/pkls/dbpedia_wikidata_15k_dense_GA_id_img_feature_dict.pkl"
    elif "FB15K" in file_dir:
        filename = os.path.split(file_dir)[-1].upper()
        img_vec_path = "data/mmkb-datasets/" + filename + "/" + filename + "_id_img_feature_dict.pkl"
    else:
        split = file_dir.split("/")[-1]
        img_vec_path = "data/pkls/" + split + "_GA_id_img_feature_dict.pkl"

    img_features = load_img(ent_num, img_vec_path)
    return img_features


class MCLEA:

    def __init__(self):

        self.ent2id_dict = None
        self.ills = None
        self.triples = None
        self.r_hs = None
        self.r_ts = None
        self.ids = None
        self.left_ents = None
        self.right_ents = None

        self.img_features = None
        self.rel_features = None
        self.att_features = None
        self.char_features = None
        self.name_features = None
        self.ent_vec = None  # entity embedding

        self.left_non_train = None
        self.right_non_train = None
        self.ENT_NUM = None
        self.REL_NUM = None
        self.adj = None
        self.train_ill = None
        self.test_ill_ = None
        self.test_ill = None
        self.test_left = None
        self.test_right = None

        # model
        self.multimodal_encoder = None
        self.weight_raw = None
        self.rel_fc = None
        self.att_fc = None
        self.img_fc = None
        self.char_fc = None  # 字符embedding
        self.shared_fc = None

        self.gcn_pro = None
        self.rel_pro = None
        self.attr_pro = None
        self.img_pro = None
        self.input_dim = None
        self.entity_emb = None
        self.input_idx = None
        self.n_units = None
        self.n_heads = None
        self.cross_graph_model = None
        self.params = None
        self.optimizer = None

        self.criterion_cl = None
        self.criterion_align = None

        self.multi_loss_layer = None
        self.align_multi_loss_layer = None
        self.fusion = None  # fusion module

        self.parser = argparse.ArgumentParser()
        self.args = self.parse_options(self.parser)

        self.set_seed(self.args.seed, self.args.cuda)

        self.device = torch.device("cuda" if self.args.cuda and torch.cuda.is_available() else "cpu")

        # get data ids/features etc.
        self.init_data()

        # initialize model
        self.init_model()

        self.print_summary()

    @staticmethod
    def parse_options(parser):
        parser.add_argument("--file_dir", type=str, default="data/DBP15K/zh_en", required=False,
                            help="input dataset file directory, ('data/DBP15K/zh_en', 'data/DWY100K/dbp_wd')")
        parser.add_argument("--rate", type=float, default=0.3, help="training set rate")

        parser.add_argument("--cuda", action="store_true", default=True, help="whether to use cuda or not")
        parser.add_argument("--seed", type=int, default=2021, help="random seed")
        parser.add_argument("--epochs", type=int, default=1000, help="number of epochs to train")
        parser.add_argument("--check_point", type=int, default=100, help="check point")
        parser.add_argument("--hidden_units", type=str, default="128,128,128",
                            help="hidden units in each hidden layer(including in_dim and out_dim), splitted with comma")
        parser.add_argument("--heads", type=str, default="2,2", help="heads in each gat layer, splitted with comma")
        parser.add_argument("--instance_normalization", action="store_true", default=False,
                            help="enable instance normalization")
        parser.add_argument("--lr", type=float, default=0.005, help="initial learning rate")
        parser.add_argument("--weight_decay", type=float, default=0, help="weight decay (L2 loss on parameters)")
        parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate for layers")
        parser.add_argument("--attn_dropout", type=float, default=0.0, help="dropout rate for gat layers")
        parser.add_argument("--dist", type=int, default=2, help="L1 distance or L2 distance. ('1', '2')")
        parser.add_argument("--csls", action="store_true", default=False, help="use CSLS for inference")
        parser.add_argument("--csls_k", type=int, default=10, help="top k for csls")
        parser.add_argument("--il", action="store_true", default=False, help="Iterative learning?")
        parser.add_argument("--semi_learn_step", type=int, default=10, help="If IL, what's the update step?")
        parser.add_argument("--il_start", type=int, default=500, help="If Il, when to start?")
        parser.add_argument("--bsize", type=int, default=7500, help="batch size")
        parser.add_argument("--unsup", action="store_true", default=False)
        parser.add_argument("--unsup_mode", type=str, default="img", help="unsup mode")
        parser.add_argument("--unsup_k", type=int, default=1000, help="|visual seed|")
        # parser.add_argument("--long_tail_analysis", action="store_true", default=False)
        parser.add_argument("--lta_split", type=int, default=0, help="split in {0,1,2,3,|splits|-1}")
        parser.add_argument("--tau", type=float, default=0.1, help="the temperature factor of contrastive loss")
        parser.add_argument("--tau2", type=float, default=1, help="the temperature factor of alignment loss")
        parser.add_argument("--alpha", type=float, default=0.2, help="the margin of InfoMaxNCE loss")
        parser.add_argument("--with_weight", type=int, default=1, help="Whether to weight the fusion of different "
                                                                       "modal features")
        parser.add_argument("--structure_encoder", type=str, default="gat", help="the encoder of structure view, "
                                                                                 "[gcn|gat]")

        parser.add_argument("--ab_weight", type=float, default=0.5, help="the weight of NTXent Loss")

        parser.add_argument("--projection", action="store_true", default=False, help="add projection for model")

        parser.add_argument("--attr_dim", type=int, default=100, help="the hidden size of attr and rel features")
        parser.add_argument("--img_dim", type=int, default=100, help="the hidden size of img feature")
        parser.add_argument("--name_dim", type=int, default=100, help="the hidden size of name feature")
        parser.add_argument("--char_dim", type=int, default=100, help="the hidden size of char feature")

        parser.add_argument("--w_gcn", action="store_false", default=True, help="with gcn features")
        parser.add_argument("--w_rel", action="store_false", default=True, help="with rel features")
        parser.add_argument("--w_attr", action="store_false", default=True, help="with attr features")
        parser.add_argument("--w_name", action="store_false", default=True, help="with name features")
        parser.add_argument("--w_char", action="store_false", default=True, help="with char features")
        parser.add_argument("--w_img", action="store_false", default=True, help="with img features")

        # multi loss params
        parser.add_argument("--inner_view_num", type=int, default=6, help="the number of inner view")

        parser.add_argument("--word_embedding", type=str, default="glove", help="the type of word embedding, "
                                                                                "[glove|fasttext]")
        # projection head
        parser.add_argument("--use_project_head", action="store_true", default=False, help="use projection head")

        parser.add_argument("--zoom", type=float, default=0.1, help="narrow the range of losses")
        parser.add_argument("--reduction", type=str, default="mean", help="[sum|mean]")
        parser.add_argument("--save_path", type=str, default="save_pkl/test", help="save path")

        return parser.parse_args()

    @staticmethod
    def set_seed(seed, cuda=True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def visual_pivot_induction(self, mode="img"):
        # if unsupervised? use image to obtain links
        if mode == "char":
            l_img_f = self.char_features[self.left_ents]  # left images
            r_img_f = self.char_features[self.right_ents]  # right images
        elif mode == "name":
            l_img_f = self.name_features[self.left_ents]  # left images
            r_img_f = self.name_features[self.right_ents]  # right images
        else:
            l_img_f = self.img_features[self.left_ents]  # left images
            r_img_f = self.img_features[self.right_ents]  # right images

        img_sim = l_img_f.mm(r_img_f.t())  # t : transpose

        topk = self.args.unsup_k
        two_d_indices = get_topk_indices(img_sim, topk * 100)
        del l_img_f, r_img_f, img_sim

        visual_links = []
        used_inds = []
        count = 0
        for ind in two_d_indices:
            if self.left_ents[ind[0]] in used_inds:
                continue
            if self.right_ents[ind[1]] in used_inds:
                continue
            used_inds.append(self.left_ents[ind[0]])
            used_inds.append(self.right_ents[ind[1]])
            visual_links.append((self.left_ents[ind[0]], self.right_ents[ind[1]]))
            count += 1
            if count == topk:
                break

        count = 0.0
        for link in visual_links:
            if link in self.ills:
                count = count + 1
        print("%.2f%% in true links" % (count / len(visual_links) * 100))
        print("visual links length: %d" % (len(visual_links)))
        train_ill = np.array(visual_links, dtype=np.int32)
        return train_ill

    def print_summary(self):
        print("-----dataset summary-----")
        print("dataset:\t", self.args.file_dir)
        print("triple num:\t", len(self.triples))
        print("entity num:\t", self.ENT_NUM)
        print("relation num:\t", self.REL_NUM)
        print("train ill num:\t", self.train_ill.shape[0], "\ttest ill num:\t", self.test_ill.shape[0])
        print("-------------------------")

    def init_data(self):
        # Load data
        lang_list = [1, 2]
        file_dir = self.args.file_dir
        device = self.device

        self.ent2id_dict, self.ills, self.triples, self.r_hs, \
        self.r_ts, self.ids = read_raw_data(file_dir, lang_list)
        e1 = os.path.join(file_dir, 'ent_ids_1')
        e2 = os.path.join(file_dir, 'ent_ids_2')
        self.left_ents = get_ids(e1)
        self.right_ents = get_ids(e2)

        self.ENT_NUM = len(self.ent2id_dict)
        self.REL_NUM = len(self.r_hs)
        print("total ent num: {}, rel num: {}".format(self.ENT_NUM, self.REL_NUM))

        np.random.shuffle(self.ills)

        # load images features
        self.img_features = load_img_features(self.ENT_NUM, file_dir)
        self.img_features = F.normalize(torch.Tensor(self.img_features).to(device))
        print("image feature shape:", self.img_features.shape)

        # load name/char features (only for DBP15K datasets)
        data_dir, dataname = os.path.split(file_dir)
        if self.args.word_embedding == "glove":
            word2vec_path = "data/embedding/glove.6B.300d.txt"
        elif self.args.word_embedding == 'fasttext':
            pass
        else:
            raise Exception("error word embedding")

        if "DBP15K" in file_dir:
            name_path = os.path.join(data_dir, "translated_ent_name", "dbp_" + dataname + ".json")
            self.ent_vec, self.char_features = load_word_char_features(self.ENT_NUM, word2vec_path, name_path)
            self.name_features = F.normalize(torch.Tensor(self.ent_vec)).to(self.device)
            self.char_features = F.normalize(torch.Tensor(self.char_features).to(device))
            print("name feature shape:", self.name_features.shape)
            print("char feature shape:", self.char_features.shape)

        # train/val/test split
        if self.args.unsup:
            # if unsupervised? use image to obtain links
            self.train_ill = self.visual_pivot_induction(mode=self.args.unsup_mode)
        else:
            # if supervised
            self.train_ill = np.array(self.ills[:int(len(self.ills) // 1 * self.args.rate)], dtype=np.int32)

        self.test_ill_ = self.ills[int(len(self.ills) // 1 * self.args.rate):]
        self.test_ill = np.array(self.test_ill_, dtype=np.int32)

        self.test_left = torch.LongTensor(self.test_ill[:, 0].squeeze()).to(device)
        self.test_right = torch.LongTensor(self.test_ill[:, 1].squeeze()).to(device)

        self.left_non_train = list(set(self.left_ents) - set(self.train_ill[:, 0].tolist()))
        self.right_non_train = list(set(self.right_ents) - set(self.train_ill[:, 1].tolist()))
        # left_non_train = test_ill[:,0].tolist()
        # right_non_train = test_ill[:,1].tolist()
        print("#left entity : %d, #right entity: %d" % (len(self.left_ents), len(self.right_ents)))
        print("#left entity not in train set: %d, #right entity not in train set: %d"
              % (len(self.left_non_train), len(self.right_non_train)))

        # convert relations to numbers
        self.rel_features = load_relation(self.ENT_NUM, self.triples, 1000)
        self.rel_features = torch.Tensor(self.rel_features).to(device)
        print("relation feature shape:", self.rel_features.shape)

        # convert attributions to numbers
        a1 = os.path.join(file_dir, 'training_attrs_1')
        a2 = os.path.join(file_dir, 'training_attrs_2')
        self.att_features = load_attr([a1, a2], self.ENT_NUM, self.ent2id_dict, 1000)  # attr
        self.att_features = torch.Tensor(self.att_features).to(device)
        print("attribute feature shape:", self.att_features.shape)

        self.adj = get_adjr(self.ENT_NUM, self.triples, norm=True)  # getting a sparse tensor r_adj
        self.adj = self.adj.to(self.device)

    def init_model(self):
        img_dim = self.img_features.shape[1]
        char_dim = self.char_features.shape[1] if self.char_features is not None else 100
        self.multimodal_encoder = MultiModalEncoder(args=self.args,
                                                    ent_num=self.ENT_NUM,
                                                    img_feature_dim=img_dim,
                                                    char_feature_dim=char_dim,
                                                    use_project_head=self.args.use_project_head).to(self.device)

        self.multi_loss_layer = CustomMultiLossLayer(loss_num=self.args.inner_view_num).to(self.device)
        self.align_multi_loss_layer = CustomMultiLossLayer(loss_num=self.args.inner_view_num).to(self.device)

        self.params = [
            {"params":
                 list(self.multimodal_encoder.parameters()) +
                 list(self.multi_loss_layer.parameters()) +
                 list(self.align_multi_loss_layer.parameters())
             }]
        self.optimizer = optim.AdamW(
            self.params,
            lr=self.args.lr
        )
        total_params = sum(p.numel() for p in self.multimodal_encoder.parameters() if p.requires_grad)
        total_params += sum(p.numel() for p in self.multi_loss_layer.parameters() if p.requires_grad)
        total_params += sum(p.numel() for p in self.align_multi_loss_layer.parameters() if p.requires_grad)
        print("total params num", total_params)
        # {"params": [weight_raw], "lr":0.01, "weight_decay":0}],
        # optimizer = optim.AdamW(params, lr=args.lr)
        print("MCLEA model details:")
        print(self.multimodal_encoder.cross_graph_model)
        print("optimiser details:")
        print(self.optimizer)

        # contrastive loss
        self.criterion_cl = icl_loss(device=self.device, tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2)
        self.criterion_align = ial_loss(device=self.device, tau=self.args.tau2,
                                        ab_weight=self.args.ab_weight,
                                        zoom=self.args.zoom,
                                        reduction=self.args.reduction)

    def semi_supervised_learning(self):

        with torch.no_grad():
            gph_emb, img_emb, rel_emb, att_emb, \
            name_emb, char_emb, joint_emb = self.multimodal_encoder(self.input_idx,
                                                                    self.adj,
                                                                    self.img_features,
                                                                    self.rel_features,
                                                                    self.att_features,
                                                                    self.name_features,
                                                                    self.char_features)

            final_emb = F.normalize(joint_emb)

        distance_list = []
        for i in np.arange(0, len(self.left_non_train), 1000):
            d = pairwise_distances(final_emb[self.left_non_train[i:i + 1000]], final_emb[self.right_non_train])
            distance_list.append(d)
        distance = torch.cat(distance_list, dim=0)
        preds_l = torch.argmin(distance, dim=1).cpu().numpy().tolist()
        preds_r = torch.argmin(distance.t(), dim=1).cpu().numpy().tolist()
        del distance_list, distance, final_emb
        return preds_l, preds_r

    def inner_view_loss(self, gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, train_ill):

        loss_GCN = self.criterion_cl(gph_emb, train_ill) if gph_emb is not None else 0
        loss_rel = self.criterion_cl(rel_emb, train_ill) if rel_emb is not None else 0
        loss_att = self.criterion_cl(att_emb, train_ill) if att_emb is not None else 0
        loss_img = self.criterion_cl(img_emb, train_ill) if img_emb is not None else 0
        loss_name = self.criterion_cl(name_emb, train_ill) if name_emb is not None else 0
        loss_char = self.criterion_cl(char_emb, train_ill) if char_emb is not None else 0

        total_loss = self.multi_loss_layer([loss_GCN, loss_rel, loss_att, loss_img, loss_name, loss_char])
        return total_loss

    def kl_alignment_loss(self, joint_emb, gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, train_ill):

        zoom = self.args.zoom
        loss_GCN = self.criterion_align(gph_emb, joint_emb, train_ill) if gph_emb is not None else 0
        loss_rel = self.criterion_align(rel_emb, joint_emb, train_ill) if rel_emb is not None else 0
        loss_att = self.criterion_align(att_emb, joint_emb, train_ill) if att_emb is not None else 0
        loss_img = self.criterion_align(img_emb, joint_emb, train_ill) if img_emb is not None else 0
        loss_name = self.criterion_align(name_emb, joint_emb, train_ill) if name_emb is not None else 0
        loss_char = self.criterion_align(char_emb, joint_emb, train_ill) if char_emb is not None else 0

        total_loss = self.align_multi_loss_layer(
                [loss_GCN, loss_rel, loss_att, loss_img, loss_name, loss_char]) * zoom
        return total_loss

    def train(self):

        # print args
        pprint(self.args)

        # Train
        print("[start training...] ")
        t_total = time.time()
        new_links = []
        epoch_KE, epoch_CG = 0, 0

        bsize = self.args.bsize
        device = self.device

        self.input_idx = torch.LongTensor(np.arange(self.ENT_NUM)).to(device)

        for epoch in range(self.args.epochs):

            if epoch == epoch >= self.args.il_start:
                self.optimizer = optim.AdamW(self.params, lr=self.args.lr / 5)

            t_epoch = time.time()

            self.multimodal_encoder.train()
            self.multi_loss_layer.train()
            self.align_multi_loss_layer.train()
            self.optimizer.zero_grad()

            gph_emb, img_emb, rel_emb, att_emb, \
            name_emb, char_emb, joint_emb = self.multimodal_encoder(self.input_idx,
                                                                    self.adj,
                                                                    self.img_features,
                                                                    self.rel_features,
                                                                    self.att_features,
                                                                    self.name_features,
                                                                    self.char_features)

            loss_sum_gcn, loss_sum_rel, loss_sum_att, loss_sum_img, loss_sum_all = 0, 0, 0, 0, 0

            epoch_CG += 1

            # manual batching
            np.random.shuffle(self.train_ill)
            for si in np.arange(0, self.train_ill.shape[0], bsize):
                #  ICL loss for joint embedding
                loss_joi = self.criterion_cl(joint_emb, self.train_ill[si:si + bsize])

                # ICL loss for uni-modal embedding
                in_loss = self.inner_view_loss(gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb,
                                               self.train_ill[si:si + bsize])

                # IAL loss for uni-modal embedding
                align_loss = self.kl_alignment_loss(joint_emb, gph_emb, rel_emb, att_emb, img_emb, name_emb,
                                                    char_emb, self.train_ill[si:si + bsize])

                loss_all = loss_joi + in_loss + align_loss

                loss_all.backward(retain_graph=True)

                loss_sum_all = loss_sum_all + loss_all.item()

            self.optimizer.step()
            print("[epoch {:d}] loss_all: {:f}, time: {:.4f} s".format(epoch, loss_sum_all, time.time() - t_epoch))

            # semi-supervised learning
            if epoch >= self.args.il_start and (epoch + 1) % self.args.semi_learn_step == 0 and self.args.il:
                # predict links
                preds_l, preds_r = self.semi_supervised_learning()

                # if args.csls is True:
                #    distance = 1 - csls_sim(1 - distance, args.csls_k)
                # print (len(preds_l), len(preds_r))

                if (epoch + 1) % (self.args.semi_learn_step * 10) == self.args.semi_learn_step:
                    new_links = [(self.left_non_train[i], self.right_non_train[p]) for i, p in enumerate(preds_l)
                                 if preds_r[p] == i]  # Nearest neighbors
                else:
                    new_links = [(self.left_non_train[i], self.right_non_train[p]) for i, p in enumerate(preds_l)
                                 if (preds_r[p] == i)
                                 and ((self.left_non_train[i], self.right_non_train[p]) in new_links)]
                print("[epoch %d] #links in candidate set: %d" % (epoch, len(new_links)))

            if epoch >= self.args.il_start and (epoch + 1) % (self.args.semi_learn_step * 10) == 0 and len(
                    new_links) != 0 and self.args.il:
                # get similarity of newly linked pairs
                # if len(new_links) > 1000:
                #    left_inds = np.array(new_links, dtype=np.int32)[:,0]
                #    right_inds = np.array(new_links, dtype=np.int32)[:,1]
                #    emb_sim = final_emb[left_inds].mm(final_emb[right_inds].t())
                #    two_d_indices = get_topk_indices(emb_sim, K=1000)
                #    new_links_elect = []
                #    for inds in two_d_indices:
                #        new_links_elect.append((left_inds[inds[0]], right_inds[inds[1]]))
                # else:
                new_links_elect = new_links
                print("\n#new_links_elect:", len(new_links_elect))

                # if len(new_links) >= 5000: new_links = random.sample(new_links, 5000)
                self.train_ill = np.vstack((self.train_ill, np.array(new_links_elect)))
                print("train_ill.shape:", self.train_ill.shape)

                num_true = len([nl for nl in new_links_elect if nl in self.test_ill_])
                print("#true_links: %d" % num_true)
                print("true link ratio: %.1f%%" % (100 * num_true / len(new_links_elect)))

                # remove from left/right_non_train
                for nl in new_links_elect:
                    self.left_non_train.remove(nl[0])
                    self.right_non_train.remove(nl[1])
                print("#entity not in train set: %d (left) %d (right)" % (
                    len(self.left_non_train), len(self.right_non_train)))

                new_links = []

            if self.args.cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Test
            if (epoch + 1) % self.args.check_point == 0:
                print("\n[epoch {:d}] checkpoint!".format(epoch))
                self.test(epoch)

            if self.args.cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()

            del joint_emb, gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb

        print("[optimization finished!]")
        print("[total time elapsed: {:.4f} s]".format(time.time() - t_total))

    def test(self, epoch):
        with torch.no_grad():
            t_test = time.time()
            self.multimodal_encoder.eval()
            self.multi_loss_layer.eval()
            self.align_multi_loss_layer.eval()

            gph_emb, img_emb, rel_emb, att_emb, \
            name_emb, char_emb, joint_emb = self.multimodal_encoder(self.input_idx,
                                                                    self.adj,
                                                                    self.img_features,
                                                                    self.rel_features,
                                                                    self.att_features,
                                                                    self.name_features,
                                                                    self.char_features)

            w_normalized = F.softmax(self.multimodal_encoder.fusion.weight, dim=0)
            print("normalised weights:", w_normalized.data.squeeze())

            inner_view_weight = torch.exp(-self.multi_loss_layer.log_vars)
            print("inner-view loss weights:", inner_view_weight.data)
            align_weight = torch.exp(-self.align_multi_loss_layer.log_vars)
            print("align loss weights:", align_weight.data)

            final_emb = F.normalize(joint_emb)

            # top_k = [1, 5, 10, 50, 100]
            top_k = [1, 10, 50]
            if "100" in self.args.file_dir:
                Lvec = final_emb[self.test_left].cpu().data.numpy()
                Rvec = final_emb[self.test_right].cpu().data.numpy()
                acc_l2r, mean_l2r, mrr_l2r, acc_r2l, mean_r2l, mrr_r2l = multi_get_hits(Lvec, Rvec, top_k=top_k,
                                                                                        args=self.args)
                del final_emb
                gc.collect()
            else:
                acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
                acc_r2l = np.zeros((len(top_k)), dtype=np.float32)
                test_total, test_loss, mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0, 0., 0., 0., 0., 0.
                if self.args.dist == 2:
                    distance = pairwise_distances(final_emb[self.test_left], final_emb[self.test_right])
                elif self.args.dist == 1:
                    distance = torch.FloatTensor(scipy.spatial.distance.cdist(
                        final_emb[self.test_left].cpu().data.numpy(),
                        final_emb[self.test_right].cpu().data.numpy(), metric="cityblock"))
                else:
                    raise NotImplementedError

                if self.args.csls is True:
                    distance = 1 - csls_sim(1 - distance, self.args.csls_k)

                if epoch + 1 == self.args.epochs:
                    to_write = []
                    test_left_np = self.test_left.cpu().numpy()
                    test_right_np = self.test_right.cpu().numpy()
                    to_write.append(["idx", "rank", "query_id", "gt_id", "ret1", "ret2", "ret3"])

                for idx in range(self.test_left.shape[0]):
                    values, indices = torch.sort(distance[idx, :], descending=False)
                    rank = (indices == idx).nonzero().squeeze().item()
                    mean_l2r += (rank + 1)
                    mrr_l2r += 1.0 / (rank + 1)
                    for i in range(len(top_k)):
                        if rank < top_k[i]:
                            acc_l2r[i] += 1
                    # save idx, correct rank pos, and indices
                    if epoch + 1 == self.args.epochs:
                        indices = indices.cpu().numpy()
                        to_write.append(
                            [idx, rank, test_left_np[idx], test_right_np[idx], test_right_np[indices[0]],
                             test_right_np[indices[1]], test_right_np[indices[2]]])
                if epoch + 1 == self.args.epochs:
                    import csv
                    # with open("logs/pred.txt", "w") as f:
                    #     wr = csv.writer(f, dialect='excel')
                    #     wr.writerows(to_write)
                    save_path = self.args.save_path
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    with open(os.path.join(save_path, "pred.txt"), "w") as f:
                        wr = csv.writer(f, dialect='excel')
                        wr.writerows(to_write)

                for idx in range(self.test_right.shape[0]):
                    _, indices = torch.sort(distance[:, idx], descending=False)
                    rank = (indices == idx).nonzero().squeeze().item()
                    mean_r2l += (rank + 1)
                    mrr_r2l += 1.0 / (rank + 1)
                    for i in range(len(top_k)):
                        if rank < top_k[i]:
                            acc_r2l[i] += 1

                mean_l2r /= self.test_left.size(0)
                mean_r2l /= self.test_right.size(0)
                mrr_l2r /= self.test_left.size(0)
                mrr_r2l /= self.test_right.size(0)
                for i in range(len(top_k)):
                    acc_l2r[i] = round(acc_l2r[i] / self.test_left.size(0), 4)
                    acc_r2l[i] = round(acc_r2l[i] / self.test_right.size(0), 4)
                del distance, gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb
                gc.collect()
            print("l2r: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s ".format(top_k, acc_l2r,
                                                                                                mean_l2r, mrr_l2r,
                                                                                                time.time() - t_test))
            print("r2l: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s \n".format(top_k, acc_r2l,
                                                                                                  mean_r2l, mrr_r2l,
                                                                                                  time.time() - t_test))


if __name__ == "__main__":
    model = MCLEA()
    model.train()
