import os
import h5py
import pickle
import numpy as np

data_path = "../data/mmkb-datasets"


def handle_attr_triples(path, save_root, ids=1):
    ent2attr_dict = dict()
    with open(path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            items = line.strip().split()
            assert len(items) == 3 or len(items) == 4
            items = items[:3]
            ent, attr, attr_val = items
            if ent not in ent2attr_dict:
                ent2attr_dict[ent] = set()
            ent2attr_dict[ent].add(attr)
    savefile = os.path.join(save_root, "training_attrs_" + str(ids))
    with open(savefile, 'w', encoding="utf-8") as f_w:
        for key, val in ent2attr_dict.items():
            f_w.write(key + "\t" + "\t".join(val) + "\n")


def save_rel_triples(data_root, save_root, ent2ids, rel2ids, ids=1):
    rel_triple_file = os.path.join(data_root, os.path.split(data_root)[-1] + "_EntityTriples.txt")
    if ids == 2:
        ill_file = os.path.join(data_root, os.path.split(data_root)[-1] + "_SameAsLink.txt")
    else:
        ill_file = None
    triples = list()
    ent_ids = dict()
    with open(rel_triple_file, "r", encoding="utf-8") as f_in:
        for line in f_in:
            items = line.strip().split()
            assert len(items) == 4
            head, rel, tail = items[:3]
            head_id = str(ent2ids[head])
            rel_id = str(rel2ids[rel])
            tail_id = str(ent2ids[tail])
            triples.append((head_id, rel_id, tail_id))
            if head_id not in ent2ids:
                ent_ids[head_id] = head
            if tail_id not in ent2ids:
                ent_ids[tail_id] = tail
    savefile = os.path.join(save_root, "triples_" + str(ids))
    with open(savefile, 'w', encoding="utf-8") as f_w:
        for triple in triples:
            f_w.write("\t".join(triple) + "\n")
    savefile = os.path.join(save_root, "ent_ids_" + str(ids))
    with open(savefile, 'w', encoding="utf-8") as f_w:
        for key, val in ent_ids.items():
            f_w.write(str(key) + "\t" + val + "\n")

    ills = list()
    ent_sets = set()
    if ill_file:
        with open(ill_file, "r", encoding="utf-8") as f_in:
            for line in f_in:
                items = line.strip().split()
                assert len(items) == 4
                a, b = items[0], items[2]
                a_id = str(ent2ids[a])
                b_id = str(ent2ids[b])
                ills.append((a_id, b_id))
                if a not in ent_sets:
                    ent_sets.add(a)
                else:
                    print(a)
                if b not in ent_sets:
                    ent_sets.add(b)
                else:
                    print(b)
        savefile = os.path.join(save_root, "ill_ent_ids")
        with open(savefile, 'w', encoding="utf-8") as f_w:
            for ill in ills:
                f_w.write("\t".join(ill) + "\n")
        print("total ill link {}".format(len(ills)))


def handle_rel_triples(data_paths, save_root):
    ent2ids = dict()
    rel2ids = dict()
    for path in data_paths:
        with open(path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                items = line.strip().split()
                assert len(items) == 4
                head, rel, tail = items[:3]
                if head not in ent2ids:
                    ent2ids[head] = len(ent2ids)
                if tail not in ent2ids:
                    ent2ids[tail] = len(ent2ids)
                if rel not in rel2ids:
                    rel2ids[rel] = len(rel2ids)

    for i in range(len(data_paths)):
        data_root = os.path.dirname(data_paths[i])
        save_rel_triples(data_root, save_root, ent2ids, rel2ids, ids=i + 1)
    print("total entities {}, rels {}".format(len(ent2ids), len(rel2ids)))
    return ent2ids, rel2ids


def handle_img_features(data_paths, save_root, ent2ids):
    ent2feats = dict()

    def load_img_index(path):
        img2ids = dict()
        with open(path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                items = line.strip().split()
                assert len(items) == 2
                img2ids[items[1]] = items[0]
        return img2ids
    repeat_num = 0
    for path in data_paths:
        img_id_path = os.path.join(os.path.dirname(path), path.split("/")[-2]+"_ImageIndex.txt")
        img2ids = load_img_index(img_id_path)
        f = h5py.File(path, 'r')
        for key, val in f.items():
            key_ = img2ids[key]
            id_ = ent2ids[key_]
            val = val[:].reshape(-1,)
            if id_ not in ent2feats:
                ent2feats[id_] = val
            else:
                repeat_num += 1
                print(key, key_, id_, val.shape)
                # raise Exception("load error.")
    savefile = os.path.join(save_root, os.path.split(save_root)[-1] + "_id_img_feature_dict.pkl")
    with open(savefile, 'wb') as f_w:
        pickle.dump(ent2feats, f_w)
    print("repeat num {}".format(repeat_num))


def convert_mmkb_to_dbpStyle(dataset_name, data1, data2):
    # attr triples
    data_root = os.path.join(data_path, dataset_name)
    if not os.path.exists(data_root):
        os.mkdir(data_root)
    handle_attr_triples(os.path.join(data1, os.path.split(data1)[-1] + "_NumericalTriples.txt"), save_root=data_root,
                        ids=1)
    handle_attr_triples(os.path.join(data2, os.path.split(data2)[-1] + "_NumericalTriples.txt"), save_root=data_root,
                        ids=2)
    # rel triples
    data1_rel_path = os.path.join(data1, os.path.split(data1)[-1] + "_EntityTriples.txt")
    data2_rel_path = os.path.join(data2, os.path.split(data2)[-1] + "_EntityTriples.txt")
    data_paths = [data1_rel_path, data2_rel_path]
    ent2ids, rel2ids = handle_rel_triples(data_paths, data_root)

    # image features (based on VGG16)
    data1_img_path = os.path.join(data1, os.path.split(data1)[-1] + "_ImageData.h5")
    data2_img_path = os.path.join(data2, os.path.split(data2)[-1] + "_ImageData.h5")
    data_paths = [data1_img_path, data2_img_path]
    handle_img_features(data_paths, data_root, ent2ids)


if __name__ == "__main__":
    # dataset_name = "FB15K_DB15K"
    # data1 = os.path.join(data_path, "FB15K")
    # data2 = os.path.join(data_path, "DB15K")
    # convert_mmkb_to_dbpStyle(dataset_name, data1, data2)

    dataset_name = "FB15K_YAGO15K"
    data1 = os.path.join(data_path, "FB15K")
    data2 = os.path.join(data_path, "YAGO15K")
    convert_mmkb_to_dbpStyle(dataset_name, data1, data2)
