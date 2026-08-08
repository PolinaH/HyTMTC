import os, random, json, csv
import ipaddress, pickle
import pudb
import warnings

warnings.filterwarnings("ignore")


# generate random ipv4 address
def random_ipv4():
    IPV4_MAX = ipaddress.IPv4Address._ALL_ONES
    ip_int = random.randint(0, IPV4_MAX)
    ip_str = ipaddress.IPv4Address._string_from_ip_int(ip_int)
    return ip_str


# generate random ipv6 address
def random_ipv6():
    IPV6_MAX = ipaddress.IPv6Address._ALL_ONES
    ip_int = random.randint(0, IPV6_MAX)
    ip_str = ipaddress.IPv6Address._string_from_ip_int(ip_int)
    return ip_str


def random_field(bits):
    field_max = 2 ** bits - 1
    field_int = random.randint(0, field_max)
    return field_int


def convert_pcapng_2_pcap(pcapng_path, pcapng_file, output_path):
    pcap_file = output_path + pcapng_file.replace('pcapng', 'pcap')
    cmd = "editcap -F pcap %s %s"
    command = cmd % (pcapng_path + pcapng_file, pcap_file)
    os.system(command)
    return 0


def convert_pcapng_to_pcap(logging, pcapng_path, pcap_path, folder_is_pcap_label):
    # (1) pcapng转为pcap
    # (2) 变成每个文件夹是一类
    logging.info("Begin converting pcapng to pcap...")
    if not os.listdir(pcap_path):
        if not folder_is_pcap_label:  # pcapng_path文件夹中一个pcapng文件是一类
            for file in os.listdir(pcapng_path):
                if file[-7:] == ".pcapng":
                    if not os.path.exists(f"{pcap_path}/{file[:-7]}"):
                        os.makedirs(f"{pcap_path}/{file[:-7]}")
                    # 将pcapng转为pcap
                    cmd = f"editcap -F pcap {pcapng_path}/{file} {pcapng_path}/{file[:-7]}.pcap"
                    os.system(cmd)
                    os.system(f"rm {pcapng_path}/{file}")
                    cmd = f"mv {pcapng_path}/{file[:-7]}.pcap {pcap_path}/{file[:-7]}/"
                    os.system(cmd)
                else:
                    if not os.path.exists(f"{pcap_path}/{file[:-5]}"):
                        os.makedirs(f"{pcap_path}/{file[:-5]}")
                    cmd = f"mv {pcapng_path}/{file[:-5]}.pcap {pcap_path}/{file[:-5]}/"
                    os.system(cmd)
        else:  # pcapng_path文件夹中还包含一层文件夹，一个文件夹是一类，每个文件夹中包含多个pcapng文件
            for c in os.listdir(pcapng_path):
                if not os.path.exists(f"{pcap_path}/{c}"):
                    os.makedirs(f"{pcap_path}/{c}")
                for file in os.listdir(f"{pcapng_path}/{c}"):
                    if file[-7:] == ".pcapng":
                        # 将pcapng转为pcap
                        cmd = f"editcap -F pcap {pcapng_path}/{c}/{file} {pcapng_path}/{c}/{file[:-7]}.pcap"
                        os.system(cmd)
                        os.system(f"rm {pcapng_path}/{c}/{file}")
                        cmd = f"mv {pcapng_path}/{c}/{file[:-7]}.pcap {pcap_path}/{c}/"
                        os.system(cmd)
                    else:
                        cmd = f"mv {pcapng_path}/{c}/{file[:-5]}.pcap {pcap_path}/{c}/"
                        os.system(cmd)
    else:
        if not folder_is_pcap_label:
            for file in os.listdir(pcap_path):
                if not os.path.exists(f"{pcap_path}/{file[:-5]}"):
                    os.makedirs(f"{pcap_path}/{file[:-5]}")
                cmd = f"mv {pcap_path}/{file} {pcap_path}/{file[:-5]}/"
                os.system(cmd)


def split_cap(logging, pcap_split_path, pcap_file_path, pcap_name, pcap_label, split_way='bidirection'):
    # f"{pcap_split_path}/splitcap/{pcap_label}/{pcap_name}" is output
    # f"{pcap_file_path}/{pcap_name}" is input
    logging.info("Begin splitting pcap as session flows...")

    # pudb.set_trace()
    if not os.path.exists(f"{pcap_split_path}/splitcap/{pcap_label}"):
        os.makedirs(f"{pcap_split_path}/splitcap/{pcap_label}")
    # if not os.path.exists(f"{pcap_split_path}/splitcap/{pcap_label}/{pcap_name}"):
    #     os.makedirs(f"{pcap_split_path}/splitcap/{pcap_label}/{pcap_name}")
    file_path = f"{pcap_file_path}/{pcap_name}"
    output_path = f"{pcap_split_path}/splitcap/{pcap_label}"  # + "/" + pcap_name
    # 'bidirection':按照双向流切分 'flow':按照单向流切分
    split_way = "session" if split_way == 'bidirection' else "flow"

    logging.info(f"file path: {file_path}")
    logging.info(f"output path: {output_path}")
    cmd = f"mono ./data_generation/SplitCap.exe -r {file_path} -s {split_way} -o {output_path}"
    # print(cmd)
    os.system(cmd)
    return output_path


def cut(obj, sec):
    result = [obj[i:i + sec] for i in range(0, len(obj), sec)]
    try:
        remanent_count = len(result[0]) % 4
    except Exception as e:
        remanent_count = 0
        print("cut datagram error!")
    if remanent_count == 0:
        pass
    else:
        result = [obj[i:i + sec + remanent_count] for i in range(0, len(obj), sec + remanent_count)]
    return result


def bigram_generation(packet_datagram, token_len=64, flag=True):
    result = ''
    generated_datagram = cut(packet_datagram, 1)
    token_count = 0
    for sub_string_index in range(len(generated_datagram)):
        if sub_string_index != (len(generated_datagram) - 1):
            token_count += 1
            if token_count > token_len:
                break
            else:
                merge_word_bigram = generated_datagram[sub_string_index] + generated_datagram[sub_string_index + 1]
        else:
            break
        result += merge_word_bigram
        result += ' '

    return result


def gram_generation(packet_datagram):
    result = ''
    generated_datagram = cut(packet_datagram, 2)
    for sub_string_index in range(len(generated_datagram)):
        merge_word_bigram = generated_datagram[sub_string_index]
        result += merge_word_bigram
        result += ' '

    return result


def size_format(size):
    # 'KB'
    file_size = '%.3f' % float(size / 1000)
    return file_size


def read_data_from_json(json_data, features):
    X, Y = [], []
    for feature_index in range(len(features)):
        x = []
        for label in json_data.keys():
            x_label = []
            for sample_index in json_data[label][features[feature_index]].keys():
                x_label.append(json_data[label][features[feature_index]][sample_index])
            x.append(x_label)
            y = [label] * len(x_label)
            Y.append(y)
        X.append(x)
    return X, Y


def obtain_data(features, dataset_save_path, json_data=None):
    if json_data:
        X, Y = read_data_from_json(json_data, features)
    else:
        print("read dataset from json file.")
        with open(f"{dataset_save_path}/dataset.json", "r") as f:
            dataset = json.load(f)
        X, Y = read_data_from_json(dataset, features)

    for index in range(len(X)):
        if len(X[index]) != len(Y):
            print("data and labels are not properly associated.")
            print("x:%s\ty:%s" % (len(X[index]), len(Y)))
            return -1
    return X, Y


def write_dataset_tsv_jamba(datagram, tokens_list, coarse_semantic_list, fine_semantic_list, seg_list, label,
                            file_folder, file_prefix, file_suffix=''):
    dataset_file = [["label", "text_a", "tokens_list", "coarse_semantic_list", "fine_semantic_list",
                     "seg_list"]]
    for index in range(len(label)):
        dataset_file.append([label[index], datagram[index], tokens_list[index],
                             coarse_semantic_list[index], fine_semantic_list[index], seg_list[index]])
    if file_suffix == '':
        with open(f"{file_folder}/{file_prefix}_dataset.tsv", 'w', newline='') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            tsv_w.writerows(dataset_file)
    else:
        with open(f"{file_folder}/{file_prefix}_dataset_{file_suffix}.tsv", 'w', newline='') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            tsv_w.writerows(dataset_file)
    return 0


def write_dataset_tsv(data, label, file_folder, file_prefix):
    dataset_file = [["label", "text_a"]]
    for index in range(len(label)):
        dataset_file.append([label[index], data[index]])
    with open(f"{file_folder}/{file_prefix}_dataset.tsv", 'w', newline='') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        tsv_w.writerows(dataset_file)
    return 0


def write_dataset_tsv_twoc(data1, data2, label, file_dir, type):
    dataset_file = [["label", "text_a", "text_b"]]
    for index in range(len(label)):
        dataset_file.append([label[index], data1[index], data2[index]])
    with open(f"{file_dir}/{type}_dataset.tsv", 'w', newline='') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        tsv_w.writerows(dataset_file)
    return 0


def unlabel_data(label_data):
    nolabel_data = ""
    with open(label_data, newline='') as f:
        data = csv.reader(f, delimiter='\t')
        for row in data:
            nolabel_data += row[1] + '\n'
    nolabel_file = label_data.replace("test_dataset", "nolabel_test_dataset")
    # nolabel_file = label_data.replace("train_dataset", "nolabel_train_dataset")
    with open(nolabel_file, 'w', newline='') as f:
        f.write(nolabel_data)
    return 0


# print(gram_generation("a86bad1f9bcdc49a025996f808004500003c0cc540004006b3b90a2a00d31f0d"))

def get_instance_number(file):
    count = 0
    with open(file, "rb") as f:
        try:
            while True:
                intsance = pickle.load(f)
                count += 1
                if count % 1000000 == 0:
                    print(count)
        except EOFError:
            print(count)


def typicalsamling(group, typicalNDict):
    name = group.name
    n = typicalNDict[name]
    return group.sample(n=n)
