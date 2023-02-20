import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from data_handler.sampler import sampler

class Net(Dataset):
    # read dataset
    def __init__(self, net_path):
        self.vtx_to_id, self.id_to_vtx = {}, {}
        self.usr_to_id, self.id_to_usr = {}, {}
        self.itm_to_id, self.id_to_itm = {}, {}
        self.list_of_vtx = []
        self.v_num = 0
        self.usr_num, self.itm_num = 0, 0
        self.pos_lst = []
        graph = []
        self.sampler = None

        R = open(net_path, encoding="utf-8", errors='ignore')
        R_lines = R.readlines()
        R.close()

        for line in tqdm(R_lines):
            usr, itm, weight = line.split(" ")

            if usr not in self.vtx_to_id:
                self.vtx_to_id[usr] = self.v_num
                self.id_to_vtx[self.v_num] = usr
                self.usr_to_id[usr] = self.usr_num
                self.id_to_usr[self.usr_num] = usr
                self.v_num += 1
                self.usr_num += 1
            if itm not in self.vtx_to_id:
                self.vtx_to_id[itm] = self.v_num
                self.id_to_vtx[self.v_num] = itm
                self.itm_to_id[itm] = self.itm_num
                self.id_to_itm[self.itm_num] = itm
                self.v_num += 1
                self.itm_num += 1

            self.pos_lst.append((usr, itm))
            graph.append((usr, itm, float(weight)))

        # build rating matrix
        self.rating_matrix = torch.zeros(self.usr_num, self.itm_num)
        for usr, itm in self.pos_lst:
            self.rating_matrix[self.usr_to_id[usr]][self.itm_to_id[itm]] = 1

        self.sampler = sampler(graph, info=True)
        self.list_of_vtx = list(self.vtx_to_id)

    def __item__(self, vtx):
        if "usr" in vtx:
            return self.usr_to_id[vtx]
        else:
            return self.itm_to_id[vtx]

    def get_usr_itm_num(self):
        return self.usr_num, self.itm_num

    def get_rating_matrix(self):
        return self.rating_matrix

    def sample_pos(self):
        usr, itm = random.choice(self.pos_lst)
        return usr, itm

    def sample_neg(self, usr, neg_num=1):
        return self.sampler.sample(usr, neg_num)
