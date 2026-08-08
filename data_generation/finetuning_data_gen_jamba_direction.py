import os
import random
import shutil
import binascii
import scapy.all as scapy
import functools
from functools import reduce
from flowcontainer.extractor import extract
from utils import *
import json, operator
from tqdm import tqdm
import multiprocessing
import copy
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from sklearn.model_selection import train_test_split
import string
# scapy.load_layer("tls")
from scapy.layers.tls.handshake import TLSClientHello, TLSServerHello
from scapy.layers.tls.extensions import TLS_Ext_ServerName
import pudb
import warnings

warnings.filterwarnings("ignore")


def random_tcp_ts_option(packets):
    src_ts = None
    dst_ts = None
    random_src_ts = random_field(32)
    random_dst_ts = random_field(32)
    src_port = None
    for packet in packets:
        if packet.haslayer(scapy.TCP):
            tcp_options = [list(option) for option in packet['TCP'].options]
            for option in tcp_options:
                if option[0] == 'Timestamp':
                    if src_port == None:
                        src_port = packet['TCP'].sport
                        src_ts = option[1][0]
                        if option[1][1] != 0:
                            dst_ts = option[1][1]
                    if packet['TCP'].sport == src_port:
                        if option[1][1] != 0:
                            option[1] = (random_src_ts + option[1][0] - src_ts, random_dst_ts + option[1][1] - dst_ts)
                        else:
                            option[1] = (random_src_ts + option[1][0] - src_ts, 0)
                    else:
                        if dst_ts == None:
                            dst_ts = option[1][0]
                        if option[1][1] != 0:
                            option[1] = (random_dst_ts + option[1][0] - dst_ts, random_src_ts + option[1][1] - src_ts)
                        else:
                            option[1] = (random_dst_ts + option[1][0] - dst_ts, 0)

            packet['TCP'].options = [tuple(option) for option in tcp_options]
    return packets


def random_tls_sni(logging):
    # We're using few bytes and won't get to the sni field
    # server name(sni) is a variable-length field, modifying it triggers a series of length modifications, ranging from tls extension length, tls length, IP length, and even packet ack.
    scapy.load_layer("tls")

    packets = scapy.rdpcap("1.pcap")
    count = 0
    for packet in packets:
        # change server name
        logging.info(f"{count}")
        count += 1
        if packet.haslayer(TLSClientHello):
            tls_client_hello = packet[TLSClientHello]
            tls_client_hello.show()
            logging.info(packet['IP'].len, packet[TLS].len, tls_client_hello.msglen, tls_client_hello.extlen)
            add = 0
            if tls_client_hello.ext == None:
                continue
            for ext in tls_client_hello.ext:
                # Check for Server Name extension
                if isinstance(ext, TLS_Ext_ServerName):
                    for server_name in ext.servernames:
                        random_length = random.randint(10, 25)  # random length
                        # random_length = server_name.namelen #keep length
                        logging.info(f"{str(random_length)}")
                        # ramdom server name
                        random_server_name = ''.join(
                            random.choices(string.ascii_lowercase + string.digits, k=random_length))
                        logging.info(f"{random_server_name}")
                        server_name.servername = random_server_name.encode("UTF-8")
                        add += random_length - server_name.namelen
                        server_name.namelen = random_length
                        server_name.nametype = 0

                    logging.info(f"add: {str(add)}")
                    ext.len += add
                    ext.servernameslen += add

            tls_client_hello.extlen += add
            tls_client_hello.msglen += add
            packet[TLS].len += add
            packet['IP'].len += add

        # change tls1.2 timestamps
        if packet.haslayer(TLSClientHello):
            packet[TLSClientHello].gmt_unix_time = random_field(32)
        if packet.haslayer(TLSServerHello):
            packet[TLSServerHello].gmt_unix_time = random_field(32)

    scapy.wrpcap("random_sni.pcap", packets)


def random_ip_port(packets: scapy.PacketList):
    if len(packets) > 0:
        if scapy.IP in packets[0]:
            client_ip = packets[0][scapy.IP].src
            server_ip = packets[0][scapy.IP].dst
            r_client_ip = random_ipv4()
            r_server_ip = random_ipv4()
        elif scapy.IPv6 in packets[0]:
            client_ip = packets[0][scapy.IPv6].src
            server_ip = packets[0][scapy.IPv6].dst
            r_client_ip = random_ipv6()
            r_server_ip = random_ipv6()

        else:
            print("other L3 protocol")
        r_client_port = random_field(16)
        r_server_port = random_field(16)
    else:
        print("No packet in the pcap file")
        return -1

    for packet in packets:
        if scapy.IP in packet:
            is_client = packet[scapy.IP].src == client_ip
            packet[scapy.IP].src = r_client_ip if is_client else r_server_ip
            packet[scapy.IP].dst = r_server_ip if is_client else r_client_ip
            if scapy.TCP in packet:
                packet[scapy.TCP].sport = r_client_port if is_client else r_server_port
                packet[scapy.TCP].dport = r_server_port if is_client else r_client_port
            if scapy.UDP in packet:
                packet[scapy.UDP].sport = r_client_port if is_client else r_server_port
                packet[scapy.UDP].dport = r_server_port if is_client else r_client_port
        if scapy.IPv6 in packet:
            is_client = packet[scapy.IPv6].src == client_ip
            packet[scapy.IPv6].src = r_client_ip if is_client else r_server_ip
            packet[scapy.IPv6].dst = r_server_ip if is_client else r_client_ip
            if scapy.TCP in packet:
                packet[scapy.TCP].sport = r_client_port if is_client else r_server_port
                packet[scapy.TCP].dport = r_server_port if is_client else r_client_port
            if scapy.UDP in packet:
                packet[scapy.UDP].sport = r_client_port if is_client else r_server_port
                packet[scapy.UDP].dport = r_server_port if is_client else r_client_port

    return packets


def random_ip(packets: scapy.PacketList):
    if len(packets) > 0:
        if scapy.IP in packets[0]:
            client_ip = packets[0][scapy.IP].src
            server_ip = packets[0][scapy.IP].dst
            r_client_ip = random_ipv4()
            r_server_ip = random_ipv4()
        elif scapy.IPv6 in packets[0]:
            client_ip = packets[0][scapy.IPv6].src
            server_ip = packets[0][scapy.IPv6].dst
            r_client_ip = random_ipv6()
            r_server_ip = random_ipv6()

        else:
            print("other L3 protocol")

    else:
        print("No packet in the pcap file")
        return -1

    for packet in packets:
        if scapy.IP in packet:
            is_client = packet[scapy.IP].src == client_ip
            packet[scapy.IP].src = r_client_ip if is_client else r_server_ip
            packet[scapy.IP].dst = r_server_ip if is_client else r_client_ip

        if scapy.IPv6 in packet:
            is_client = packet[scapy.IPv6].src == client_ip
            packet[scapy.IPv6].src = r_client_ip if is_client else r_server_ip
            packet[scapy.IPv6].dst = r_server_ip if is_client else r_client_ip

    return packets


def random_tls_randomtime(packets: scapy.PacketList):
    for packet in packets:
        if packet.haslayer(TLSClientHello):
            packet[TLSClientHello].gmt_unix_time = random_field(32)
        if packet.haslayer(TLSServerHello):
            packet[TLSServerHello].gmt_unix_time = random_field(32)
    return packets


# 生成coarse semantic embedding和fine semantic embedding
def jamba_semantic_embedding(tokens, select_packet_len):
    error_flag = False  # 若返回时error_flag==True，则舍弃这个数据包

    tokens_len = len(tokens)
    # 先默认数据包均包含worker_args['select_packet_len']个tokens，即均包含64字节
    coarse_tokens = [5] * int(select_packet_len/2)  # 除1-4编号外的全是payload
    fine_tokens = [12] * int(select_packet_len/2)  # 11个字段之外的packet字段赋值12

    # Step 1: IP header
    ihl_token = tokens[0]
    try:
        ihl = int(ihl_token[1], 16)
    except (IndexError, ValueError):
        error_flag = True
        return coarse_tokens, fine_tokens, error_flag

    ip_header_len_tokens = (ihl << 2) >> 1  # ihl * 4 / 2
    if ip_header_len_tokens != 10:
        error_flag = True
        return coarse_tokens, fine_tokens, error_flag
    end_ip = ip_header_len_tokens

    coarse_tokens[0:end_ip] = [1] * end_ip
    fine_tokens[1] = 1
    fine_tokens[4] = 2
    fine_tokens[6:8] = [3]*2
    fine_tokens[8:10] = [4]*2

    # Step 2: L4 protocol header
    try:
        protocol = int(tokens[4][2:], 16)  # 获取协议十进制编码tcp：6，udp：17
    except:
        error_flag = True
        return coarse_tokens, fine_tokens, error_flag

    l4_start = end_ip
    if protocol == 6:  # TCP
        l4_label = 2
        l4_len_tokens = 10  # 只考虑tcp标准情况
        fine_tokens[0 + l4_start] = 5
        fine_tokens[1 + l4_start] = 6
        fine_tokens[2 + l4_start:4 + l4_start] = [7] * 2
        fine_tokens[4 + l4_start:6 + l4_start] = [8] * 2
        fine_tokens[6 + l4_start] = 9
    elif protocol == 17:  # UDP
        l4_label = 3
        l4_len_tokens = 4  # 8 bytes / 2
        fine_tokens[0 + l4_start] = 10
        fine_tokens[1 + l4_start] = 11
    else:
        l4_label = 4
        l4_len_tokens = 4  # 8 bytes / 2

    end_l4 = l4_start + l4_len_tokens
    coarse_tokens[l4_start:end_l4] = [l4_label] * l4_len_tokens

    coarse_tokens = coarse_tokens[:tokens_len]
    fine_tokens = fine_tokens[:tokens_len]

    coarse_tokens = [ctx+2 for ctx in coarse_tokens]  # 从3开始编码，给PAD SEP CLS保留编码号
    fine_tokens = [ftx+2 for ftx in fine_tokens]

    return coarse_tokens, fine_tokens, error_flag


def get_feature_flow(label_pcap, tokenizer, select_packet_len, packets_num, start_index=76, add_sep=True):
    # pudb.set_trace()
    feature_data = []
    packets = scapy.rdpcap(label_pcap)

    No_ether = False
    if not hasattr(packets[0], 'type'):  # no ether header
        No_ether = True
    if (not No_ether and packets[0].type == 0x86dd) or (No_ether and packets[0].version == 6):  # do not handle IPV6
        return -1

    feature_result = extract(label_pcap)

    if len(packets) < 2:
        # print("preprocess tcp flow %s but this flow has less than 2 packets." % label_pcap)
        return -1

    if feature_result == {}:
        return -1

    packet_length = []
    packet_time = []
    packet_direction_length = []
    packet_direction = []
    packet_message_type = []

    feature_result_lens = len(feature_result.keys())
    for key in feature_result.keys():
        value = feature_result[key]  # value解析pcap中所有包，value.ip_lengths获取每个包的长度和方向eg.[61,-373]
        packet_direction_length = [x for x in value.ip_lengths]
        packet_direction = [x//abs(x) for x in value.ip_lengths]
        #packet_length.append(value.ip_lengths)
        #packet_time.append(value.ip_timestamps)

    if len(packet_direction) != len(packets):
        return -1

    #packets = random_ip_port(packets)
    packets = random_ip(packets)
    #packets = random_tcp_ts_option(packets)
    packets = random_tls_randomtime(packets)

    packet_count = 0
    flow_data_string = ''
    flow_data_tokens = []
    flow_coarse_semantic = []
    flow_fine_semantic = []
    flow_seg = []

    packet_gram_list = []  # 流列表，里面包含这条流的前packets_num个数据包的string表示
    packet_tokens_ids_list = []  # 流列表，里面包含这条流的前packets_num个数据包的tokens ids表示
    coarse_semantic_list = []  # 粗粒度语义编码列表，保存这条流中前packets_num个数据包的粗粒度语义编码
    fine_semantic_list = []  # 细粒度语义编码列表，保存这条流中前packets_num个数据包的细粒度语义编码
    packet_index = 0
    for packet in packets:
        packet_data = packet.copy()
        data = (binascii.hexlify(bytes(packet_data)))
        if No_ether:
            packet_string = data.decode()
            if packet_direction[packet_index] == 1:
                packet_string = "c49a025996f8e46f13e2e3ae0800" + packet_string
            else:
                packet_string = "e46f13e2e3aec49a025996f80800" + packet_string
            packet_string = packet_string[start_index * 2:2 * (start_index + select_packet_len)]
        else:
            packet_string = data.decode()[start_index * 2:2 * (start_index + select_packet_len)]

        packet_gram = gram_generation(packet_string.strip())
        packet_tokens = packet_gram.split(' ')
        if len(packet_tokens[-1]) != 4:
            packet_tokens = packet_tokens[:-1]
        coarse_tokens, fine_tokens, error_flag = \
            jamba_semantic_embedding(packet_tokens, select_packet_len)
        if error_flag:
            continue
        packet_gram_list.append(packet_gram)
        packet_tokens_ids_list.append(tokenizer.convert_tokens_to_ids(packet_tokens))
        coarse_semantic_list.append(coarse_tokens)
        fine_semantic_list.append(fine_tokens)

        packet_count += 1
        if packet_count == packets_num:  # 取一条流的前packets_num个数据包
            break

    tmp_flow_len = 0
    for idx_pkt in range(min(packet_count, packets_num)):
        packet_len = len(packet_tokens_ids_list[idx_pkt])
        if add_sep:
            packet_len += 1
            if idx_pkt != 0:
                flow_data_string += '[SEP] '  # [SEP] ids: 1
                flow_data_tokens.extend([1])
                flow_coarse_semantic.extend([1])
                flow_fine_semantic.extend([1])
            else:
                flow_data_string += '[CLS] '  # [CLS] ids: 2
                flow_data_tokens.extend([2])
                flow_coarse_semantic.extend([2])
                flow_fine_semantic.extend([2])
        flow_data_string += packet_gram_list[idx_pkt]
        flow_data_tokens.extend(packet_tokens_ids_list[idx_pkt])
        flow_coarse_semantic.extend(coarse_semantic_list[idx_pkt])
        flow_fine_semantic.extend(fine_semantic_list[idx_pkt])
        tmp_flow_len += packet_len
        flow_seg.append(tmp_flow_len)

    packet_direction_length = packet_direction_length[:20]
    for idx_item, item in enumerate(packet_direction_length):
        if item > 1500:
            packet_direction_length[idx_item] = 1500
        elif item < -1500:
            packet_direction_length[idx_item] = -1500
        else:
            pass
    packet_direction = packet_direction[:20]

    for len_idx, item in enumerate(packet_direction_length):
        if item < 0:
            packet_direction_length[len_idx] = str(item)
        else:
            packet_direction_length[len_idx] = '+' + str(item)

    for len_idx, item in enumerate(packet_direction):
        if item < 0:
            packet_direction[len_idx] = str(item)
        else:
            packet_direction[len_idx] = '+' + str(item)

    flow_data_tokens.extend([1])
    flow_data_tokens.extend(tokenizer.convert_tokens_to_ids(packet_direction_length))
    flow_coarse_semantic.extend([1])
    flow_coarse_semantic.extend([8]*len(packet_direction_length))
    flow_fine_semantic.extend([1])
    flow_fine_semantic.extend([15]*len(packet_direction_length))
    flow_seg.append(flow_seg[-1]+len(packet_direction_length)+1)

    feature_data.append(flow_data_string)  # flow string
    feature_data.append(packet_direction_length)  # packet_direction_length
    feature_data.append(packet_direction)  # packet_direction
    feature_data.append(flow_data_tokens)  # flow tokens
    feature_data.append(flow_coarse_semantic)  # flow coarse
    feature_data.append(flow_fine_semantic)  # flow fine
    feature_data.append(flow_seg)  # flow seg

    return feature_data


def get_feature_flow_1(label_pcap, save_path):
    # pudb.set_trace()
    packets = scapy.rdpcap(label_pcap)

    No_ether = False
    if not hasattr(packets[0], 'type'):  # no ether header
        No_ether = True
    if (not No_ether and packets[0].type == 0x86dd) or (No_ether and packets[0].version == 6):  # do not handle IPV6
        return -1

    feature_result = extract(label_pcap)

    if len(packets) < 2:
        # print("preprocess tcp flow %s but this flow has less than 2 packets." % label_pcap)
        return -1

    if feature_result == {}:
        return -1

    packet_direction = []

    feature_result_lens = len(feature_result.keys())
    for key in feature_result.keys():
        value = feature_result[key]  # value解析pcap中所有包，value.ip_lengths获取每个包的长度和方向eg.[61,-373]
        packet_direction_length = [x for x in value.ip_lengths]
        packet_direction = [x//abs(x) for x in value.ip_lengths]
        #packet_length.append(value.ip_lengths)
        #packet_time.append(value.ip_timestamps)

    if len(packet_direction) != len(packets):
        return -1

    os.system(f"cp {label_pcap} {save_path}")



def init(counter):
    global global_counter
    global_counter = counter


def process_one_label_2(r_f, src_path, dst_path):
    # pudb.set_trace()
    os.system(f"mv {src_path}/{r_f} {dst_path}")
    return 0


def process_one_label_1(r_f, save_path):
    # pudb.set_trace()
    try:
        feature_data = get_feature_flow_1(r_f, save_path)
    except:
        feature_data = -1
    return feature_data


def process_one_label(r_f, tokenizer, payload_length, payload_packet_num, start_index=76):
    # pudb.set_trace()
    try:
        feature_data = get_feature_flow(r_f, tokenizer=tokenizer, select_packet_len=payload_length,
                                        packets_num=payload_packet_num, start_index=start_index)
    except:
        feature_data = -1
    return feature_data


def feature_datas_processing(feature_dataset_path, key, feature_datas):
    result = {
        "samples": 0,
        "datagram": {},
        "datagram_packet_direction_length": {},
        "datagram_packet_direction": {},
        "tokens_list": {},
        "coarse_semantic_list": {},
        "fine_semantic_list": {},
        "seg_list": {}
    }

    for feature_data in feature_datas:
        result["samples"] += 1
        if len(result["datagram"].keys()) > 0:
            result["datagram"][str(result["samples"])] = feature_data[0]
            result["datagram_packet_direction_length"][str(result["samples"])] = feature_data[1]
            result["datagram_packet_direction"][str(result["samples"])] = feature_data[2]
            result["tokens_list"][str(result["samples"])] = feature_data[3]
            result["coarse_semantic_list"][str(result["samples"])] = feature_data[4]
            result["fine_semantic_list"][str(result["samples"])] = feature_data[5]
            result["seg_list"][str(result["samples"])] = feature_data[6]
        else:
            result["datagram"]["1"] = feature_data[0]
            result["datagram_packet_direction_length"]["1"] = feature_data[1]
            result["datagram_packet_direction"]["1"] = feature_data[2]
            result["tokens_list"]["1"] = feature_data[3]
            result["coarse_semantic_list"]["1"] = feature_data[4]
            result["fine_semantic_list"]["1"] = feature_data[5]
            result["seg_list"]["1"] = feature_data[6]

    # pudb.set_trace()

    with open(f"{feature_dataset_path}/{key}.json", 'w') as f:
        json.dump(result, f)


def generation_multiP(logging, tokenizer, split_pcap_path, process_samples_num, max_samples_num,
                      dataset_save_path, feature_dataset_path, payload_length=64, payload_packet_num=5,
                      start_index=76, max_process=24):
    # dataset_name: dataset name
    # pcap_path: the path of splited pcap. In the pacp path, each dir is one class. In each dir, each pcap is one flow.
    # samples: samples * _category, samples is the maximum samples of each class, _category is the number of class.
    # dataset_save_path: generated dataset path
    # payload_length: the used bytes of data
    # payload_packet_num: the used packets
    # start_index: the index of start byte (the number of byte, i.e., If start from IP header, start_index = 14)
    dataset = {}
    label_name_list = []
    session_pcap_path = {}

    for parent, dirs, files in os.walk(split_pcap_path):
        if label_name_list == []:
            label_name_list.extend(dirs)

        for dir in label_name_list:
            session_pcap_path[dir] = f"{split_pcap_path}/{dir}"
        break
    logging.info(f"label number: {str(len(label_name_list))}")

    label_id = {}
    for index in range(len(label_name_list)):
        label_id[label_name_list[index]] = index
    # logging.info(f"{str(label_id['Benign'])}")
    for key in label_id.keys():
        logging.info(f"category: {key}, label id: {str(label_id[key])}")

    # 按类别顺序执行，每个类别里多进程
    for key in session_pcap_path.keys():
        if os.path.exists(f"{feature_dataset_path}/{key}.json"):
            logging.info(f"{feature_dataset_path}/{key}.json already exists, skip generation")
        else:
            # 构造任务
            #tasks = []
            target_all_files = [x[0] + "/" + y for x in [(p, f) for p, d, f in os.walk(session_pcap_path[key])] for y in
                                x[1]]
            # for f in target_all_files:
            #     file_size = float(size_format(os.path.getsize(pcap_split_path+"splitcap" + "/" + dir + "/" + f)))
            #     if file_size>

            random.shuffle(target_all_files)
            if len(target_all_files) > process_samples_num:
                r_files = random.sample(target_all_files, process_samples_num)
            else:
                r_files = target_all_files

            logging.info(f"generating {feature_dataset_path}/{key}.json...")
            feature_datas = []
            feature_datas_num = 0
            if max_process != 0:  # max_process不为0时使用多进程
                # 计算进程池大小
                pool = multiprocessing.Pool(min(max_process, len(r_files)))
                pbar = tqdm(total=len(r_files), desc=f'{key} generate features')
                process_func = functools.partial(process_one_label, tokenizer=tokenizer,
                                                 payload_length=payload_length,
                                                 payload_packet_num=payload_packet_num,
                                                 start_index=start_index)  # 将不变的参数绑定
                # 使用 imap_unordered 处理任务，imap_unordered只能给函数传入1个参数
                for feature_data in pool.imap_unordered(process_func, r_files):
                    if feature_data != -1:
                        feature_datas.append(feature_data)  # 结果可能是乱序的
                    pbar.update()  # 更新进度条
                pool.close()
                pool.join()
                pbar.close()  # 关闭进度条

            else:  # 不使用多进程
                pbar = tqdm(total=len(r_files), desc=f'{key} generate features')
                for r_f in r_files:
                    feature_data = \
                        process_one_label(r_f, tokenizer, payload_length, payload_packet_num, start_index)
                    if feature_data != -1:
                        feature_datas.append(feature_data)  # 结果可能是乱序的
                        feature_datas_num += 1
                        if feature_datas_num == max_samples_num:
                            break
                    pbar.update()  # 更新进度条
                pbar.close()  # 关闭进度条
            logging.info(f"samples num in category {key}: {len(feature_datas)}")
            feature_datas_processing(feature_dataset_path, key, feature_datas)

    for jsonfile in os.listdir(feature_dataset_path):
        logging.info(f"reading {feature_dataset_path}/{jsonfile}...")
        with open(f"{feature_dataset_path}/{jsonfile}", 'r') as f:
            result = json.load(f)
        dataset[label_id[jsonfile.split('.')[-2]]] = result

    # pudb.set_trace()
    all_data_number = 0
    for index in range(len(label_name_list)):
        # logging.info("%s\t%s\t%d"%(label_id[label_name_list[index]], label_name_list[index], dataset[label_id[label_name_list[index]]]["samples"]))
        all_data_number += dataset[label_id[label_name_list[index]]]["samples"]
    logging.info(f"all samples num: {str(all_data_number)}")

    with open(f"{dataset_save_path}/dataset.json", "w") as f:
        json.dump(dataset, fp=f, ensure_ascii=False, indent=4)


def create_train_val_test_split_datasets(logging, split_pcap_path, split_train_val_test_path,
                                         process_samples_num, max_process=24):
    # dataset_name: dataset name
    # pcap_path: the path of splited pcap. In the pacp path, each dir is one class. In each dir, each pcap is one flow.
    # samples: samples * _category, samples is the maximum samples of each class, _category is the number of class.
    # dataset_save_path: generated dataset path
    # payload_length: the used bytes of data
    # payload_packet_num: the used packets
    # start_index: the index of start byte (the number of byte, i.e., If start from IP header, start_index = 14)
    dataset = {}
    label_name_list = []
    session_pcap_path = {}

    for parent, dirs, files in os.walk(split_pcap_path):
        if label_name_list == []:
            label_name_list.extend(dirs)

        for dir in label_name_list:
            session_pcap_path[dir] = f"{split_pcap_path}/{dir}"
        break
    logging.info(f"label number: {str(len(label_name_list))}")


    # 按类别顺序执行，每个类别里多进程
    for key in session_pcap_path.keys():
        if os.path.exists(f"{split_train_val_test_path}/{key}"):
            logging.info(f"{split_train_val_test_path}/{key} already exists, skip generation")
        else:
            os.makedirs(f"{split_train_val_test_path}/{key}")
            # 构造任务
            #tasks = []
            target_all_files = [x[0] + "/" + y for x in [(p, f) for p, d, f in os.walk(session_pcap_path[key])] for y in
                                x[1]]
            # for f in target_all_files:
            #     file_size = float(size_format(os.path.getsize(pcap_split_path+"splitcap" + "/" + dir + "/" + f)))
            #     if file_size>

            random.shuffle(target_all_files)
            if len(target_all_files) > process_samples_num:
                r_files = random.sample(target_all_files, process_samples_num)
            else:
                r_files = target_all_files

            logging.info(f"generating {split_train_val_test_path}/{key}...")

            # 计算进程池大小
            pool = multiprocessing.Pool(min(max_process, len(r_files)))
            pbar = tqdm(total=len(r_files), desc=f'{key} generate features')
            process_func = functools.partial(process_one_label_1,
                                             save_path=f"{split_train_val_test_path}/{key}")  # 将不变的参数绑定
            # 使用 imap_unordered 处理任务，imap_unordered只能给函数传入1个参数
            for feature_data in pool.imap_unordered(process_func, r_files):
                if feature_data != -1:
                    pass
                pbar.update()  # 更新进度条
            pool.close()
            pool.join()
            pbar.close()  # 关闭进度条

        pcap_list = os.listdir(f"{split_train_val_test_path}/{key}")
        pcap_num = len(os.listdir(f"{split_train_val_test_path}/{key}"))

        train_num = int(0.8*pcap_num)
        val_num = int(0.1*pcap_num)
        test_num = pcap_num - train_num - val_num

        random.shuffle(pcap_list)

        os.makedirs(f"{split_train_val_test_path}/{key}/train")
        os.makedirs(f"{split_train_val_test_path}/{key}/valid")
        os.makedirs(f"{split_train_val_test_path}/{key}/test")

        # 计算进程池大小
        pool = multiprocessing.Pool(min(max_process, train_num))
        pbar = tqdm(total=train_num, desc=f'{key} move pcap')
        process_func = functools.partial(process_one_label_2,
                                         src_path=f"{split_train_val_test_path}/{key}",
                                         dst_path=f"{split_train_val_test_path}/{key}/train")  # 将不变的参数绑定
        # 使用 imap_unordered 处理任务，imap_unordered只能给函数传入1个参数
        for feature_data in pool.imap_unordered(process_func, pcap_list[0:train_num]):
            if feature_data != -1:
                pass
            pbar.update()  # 更新进度条
        pool.close()
        pool.join()
        pbar.close()  # 关闭进度条


        # 计算进程池大小
        pool = multiprocessing.Pool(min(max_process, val_num))
        pbar = tqdm(total=train_num, desc=f'{key} move pcap')
        process_func = functools.partial(process_one_label_2,
                                         src_path=f"{split_train_val_test_path}/{key}",
                                         dst_path=f"{split_train_val_test_path}/{key}/valid")  # 将不变的参数绑定
        # 使用 imap_unordered 处理任务，imap_unordered只能给函数传入1个参数
        for feature_data in pool.imap_unordered(process_func, pcap_list[train_num:train_num+val_num]):
            if feature_data != -1:
                pass
            pbar.update()  # 更新进度条
        pool.close()
        pool.join()
        pbar.close()  # 关闭进度条


        # 计算进程池大小
        pool = multiprocessing.Pool(min(max_process, test_num))
        process_func = functools.partial(process_one_label_2,
                                         src_path=f"{split_train_val_test_path}/{key}",
                                         dst_path=f"{split_train_val_test_path}/{key}/test")  # 将不变的参数绑定
        # 使用 imap_unordered 处理任务，imap_unordered只能给函数传入1个参数
        for feature_data in pool.imap_unordered(process_func, pcap_list[train_num+val_num:pcap_num]):
            if feature_data != -1:
                pass
            pbar.update()  # 更新进度条
        pool.close()
        pool.join()
        pbar.close()  # 关闭进度条


def convert_splitcap(logging, pcapng_path, pcap_path, pcap_split_path, folder_is_pcap_label=False):
    # pcapng_path: the path of pcapng files (if the traffic is the pacp type, pcapng_path = pcap_path)
    # pcap_path: the path of pcap files
    # pcap_split_path: the path of splited pcap files
    # pcapng to pcap

    # pudb.set_trace()

    # (1) pcapng转为pcap
    # (2) 变成每个文件夹是一类
    convert_pcapng_to_pcap(logging, pcapng_path, pcap_path, folder_is_pcap_label)

    label_name_list = []  # 类别名称列表,每个类别名称是一个文件夹,包含多个pcap文件
    for c in os.listdir(pcap_path):
        label_name_list.append(c)

    # pudb.set_trace()

    logging.info(f"label num: {str(len(label_name_list))}")
    logging.info(f"label: {label_name_list}")

    # pudb.set_trace()
    # split pcap
    for c in os.listdir(pcap_path):
        if not os.path.exists(f"{pcap_split_path}/splitcap/{c}"):
            for file in os.listdir(f"{pcap_path}/{c}"):
                split_cap(logging, pcap_split_path, f"{pcap_path}/{c}", file, pcap_label=c)
        else:
            logging.info(f"{pcap_split_path}/splitcap/{c} already exists, skipping splitting process")

    # remove small pcap and split again big pcap
    logging.info("Begin removing small pcap and splitting again big pcap...")

    for c in os.listdir(f"{pcap_split_path}/splitcap"):
        class_path = f"{pcap_split_path}/splitcap/{c}"
        for f in os.listdir(class_path):
            file_size = float(size_format(os.path.getsize(f"{pcap_split_path}/splitcap/{c}/{f}")))
            # 1KB
            # if file_size < 1:  # remove small pcap
            #    os.remove(f"{pcap_split_path}/splitcap/{c}/{f}")
            # logging.info("remove sample: %s for its size is less than 2 KB." % (f"{pcap_split_path}/splitcap/{dir}/{f}"))
            if file_size > 10240:  # 10MB  split again big pcap
                logging.info(f"{pcap_split_path}/splitcap/{c}/{f} is bigger than 10MB")
                # -i 10 每隔10秒切分
                cmd = "editcap -i 10 {} {}".format(f"{pcap_split_path}/splitcap/{c}/{f}",
                                                   f"{pcap_split_path}/splitcap/{c}/{f}")
                os.system(cmd)
                os.system("rm {}".format(f"{pcap_split_path}/splitcap/{c}/{f}"))

    # remove class that has less flow
    all_flows = []

    for c in os.listdir(f"{pcap_split_path}/splitcap"):
        file_num = len(os.listdir(f"{pcap_split_path}/splitcap/{c}"))
        logging.info(f"{c}: {file_num}")
        if file_num < 10:
            shutil.rmtree(f"{pcap_split_path}/splitcap/{c}")
            logging.info(f"remove class: {pcap_split_path}/splitcap/{c} for its flow size is less than 10.")
        else:
            all_flows.append(file_num)

    logging.info(f"all flows: {str(sum(all_flows))}")



def dataset_extract(logging, dataset_save_path, num_category, features):
    logging.info("read dataset from json file")
    with open(f"{dataset_save_path}/dataset.json", "r") as f:
        dataset = json.load(f)

    dataset_statistic = [0] * num_category

    data_train, data_val, data_test = None, None, None
    for label_idx, app_label in enumerate(dataset.keys()):
        tmp_data_all = []
        for index_sample in range(dataset[app_label]["samples"]):
            x = []
            for feature in features:
                x.append(dataset[app_label][feature][str(index_sample + 1)])
            x.append(int(app_label))
            dataset_statistic[int(app_label)] += 1
            tmp_data_all.append(x)
        tmp_data = pd.DataFrame(tmp_data_all, columns=features + ['label'])

        # split train set and test set
        tmp_data_train, tmp_data_test = train_test_split(tmp_data, test_size=0.2, random_state=41, stratify=tmp_data["label"])
        # split validate set and test set
        tmp_data_val, tmp_data_test = train_test_split(tmp_data_test, test_size=0.5, random_state=42, stratify=tmp_data_test["label"])

        logging.info(f"{app_label} sample num of train: {len(tmp_data_train)}")
        logging.info(f"{app_label} sample num of val: {len(tmp_data_val)}")
        logging.info(f"{app_label} sample num of test: {len(tmp_data_test)}")

        if label_idx == 0:
            data_train = copy.deepcopy(tmp_data_train)
            data_val = copy.deepcopy(tmp_data_val)
            data_test = copy.deepcopy(tmp_data_test)
        else:
            data_train = pd.concat([data_train, tmp_data_train], ignore_index=True)
            data_val = pd.concat([data_val, tmp_data_val], ignore_index=True)
            data_test = pd.concat([data_test, tmp_data_test], ignore_index=True)

    logging.info("category flow")
    for index in range(len(dataset_statistic)):
        logging.info(f"catogery {index}: {str(dataset_statistic[index])}")
    logging.info(f"all samples num: {str(sum(dataset_statistic))}")

    logging.info(f"label num of train: {len(data_train['label'].value_counts())}")
    logging.info(f"label num of val: {len(data_val['label'].value_counts())}")
    logging.info(f"label num of test: {len(data_test['label'].value_counts())}")

    logging.info(f"Total sample num of train: {len(data_train)}")
    logging.info(f"Total sample num of val: {len(data_val)}")
    logging.info(f"Total sample num of test: {len(data_test)}")

    data_train = data_train.sample(frac=1)
    data_val = data_val.sample(frac=1)
    data_test = data_test.sample(frac=1)

    data_train = data_train.reset_index(drop=True)
    data_val = data_val.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)

    if not os.path.exists(f"{dataset_save_path}/dataset"):
        os.mkdir(f"{dataset_save_path}/dataset")

    # save features to .pkl
    # with open(f"{dataset_save_path}/dataset/train.pkl","wb") as f:
    #     pickle.dump(data_train,f)
    # with open(f"{dataset_save_path}/dataset/test.pkl","wb") as f:
    #     pickle.dump(data_test,f)
    # with open(f"{dataset_save_path}/dataset/valid.pkl","wb") as f:
    #     pickle.dump(data_val,f)

    # save bytes to tsv
    write_dataset_tsv_jamba(data_train['datagram'], data_train['tokens_list'],
                            data_train['coarse_semantic_list'], data_train['fine_semantic_list'],
                            data_train['seg_list'], data_train['label'],
                            f"{dataset_save_path}/dataset", file_prefix="train")
    write_dataset_tsv_jamba(data_test['datagram'], data_test['tokens_list'],
                            data_test['coarse_semantic_list'], data_test['fine_semantic_list'],
                            data_test['seg_list'], data_test['label'],
                            f"{dataset_save_path}/dataset", file_prefix="test")
    write_dataset_tsv_jamba(data_val['datagram'], data_val['tokens_list'],
                            data_val['coarse_semantic_list'], data_val['fine_semantic_list'],
                            data_val['seg_list'], data_val['label'],
                            f"{dataset_save_path}/dataset", file_prefix="valid")
    logging.info(f"finish generating fine_tuning's datagram dataset.\nPlease check in {dataset_save_path}/dataset")


def small_dataset_extract(logging, dataset_save_path, num_category, selected_sample_num, features):
    logging.info("read dataset from json file")
    with open(f"{dataset_save_path}/dataset.json", "r") as f:
        dataset = json.load(f)

    dataset_statistic = [0] * num_category

    data_train, data_val, data_test = None, None, None
    for label_idx, app_label in enumerate(dataset.keys()):
        tmp_data_all = []
        sample_num = min(dataset[app_label]["samples"], selected_sample_num)
        selected_numbers = sorted(random.sample(range(dataset[app_label]["samples"]), sample_num))
        for index_sample in selected_numbers:
            x = []
            for feature in features:
                x.append(dataset[app_label][feature][str(index_sample + 1)])
            x.append(int(app_label))
            dataset_statistic[int(app_label)] += 1
            tmp_data_all.append(x)
        tmp_data = pd.DataFrame(tmp_data_all, columns=features + ['label'])

        # split train set and test set
        tmp_data_train, tmp_data_test = train_test_split(tmp_data, test_size=0.2, random_state=41, stratify=tmp_data["label"])
        # split validate set and test set
        tmp_data_val, tmp_data_test = train_test_split(tmp_data_test, test_size=0.5, random_state=42,
                                                       stratify=tmp_data_test["label"])

        logging.info(f"{app_label} sample num of train: {len(tmp_data_train)}")
        logging.info(f"{app_label} sample num of val: {len(tmp_data_val)}")
        logging.info(f"{app_label} sample num of test: {len(tmp_data_test)}")

        if label_idx == 0:
            data_train = copy.deepcopy(tmp_data_train)
            data_val = copy.deepcopy(tmp_data_val)
            data_test = copy.deepcopy(tmp_data_test)
        else:
            data_train = pd.concat([data_train, tmp_data_train], ignore_index=True)
            data_val = pd.concat([data_val, tmp_data_val], ignore_index=True)
            data_test = pd.concat([data_test, tmp_data_test], ignore_index=True)

    logging.info("category flow")
    for index in range(len(dataset_statistic)):
        logging.info(f"catogery {index}: {str(dataset_statistic[index])}")
    logging.info(f"all samples num: {str(sum(dataset_statistic))}")

    logging.info(f"label num of train: {len(data_train['label'].value_counts())}")
    logging.info(f"label num of val: {len(data_val['label'].value_counts())}")
    logging.info(f"label num of test: {len(data_test['label'].value_counts())}")

    logging.info(f"Total sample num of train: {len(data_train)}")
    logging.info(f"Total sample num of val: {len(data_val)}")
    logging.info(f"Total sample num of test: {len(data_test)}")

    data_train = data_train.sample(frac=1)
    data_val = data_val.sample(frac=1)
    data_test = data_test.sample(frac=1)

    data_train = data_train.reset_index(drop=True)
    data_val = data_val.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)

    # save bytes to tsv
    write_dataset_tsv_jamba(data_train['datagram'], data_train['tokens_list'],
                            data_train['coarse_semantic_list'], data_train['fine_semantic_list'],
                            data_train['seg_list'], data_train['label'],
                            f"{dataset_save_path}/dataset", file_prefix="train",
                            file_suffix=int(selected_sample_num*0.8))
    write_dataset_tsv_jamba(data_test['datagram'], data_test['tokens_list'],
                            data_test['coarse_semantic_list'], data_test['fine_semantic_list'],
                            data_test['seg_list'], data_test['label'],
                            f"{dataset_save_path}/dataset", file_prefix="test",
                            file_suffix=int(selected_sample_num*0.1))
    write_dataset_tsv_jamba(data_val['datagram'], data_val['tokens_list'],
                            data_val['coarse_semantic_list'], data_val['fine_semantic_list'],
                            data_val['seg_list'], data_val['label'],
                            f"{dataset_save_path}/dataset", file_prefix="valid",
                            file_suffix=int(selected_sample_num*0.1))
    logging.info(f"finish generating fine_tuning's datagram dataset.\nPlease check in {dataset_save_path}/dataset")


def mini_dataset_extract(logging, tsv_save_path, train_tsv, num_category, selected_sample_num):
    logging.info(f"read dataset from {tsv_save_path}/{train_tsv}")

    # 读取TSV文件
    df = pd.read_csv(f"{tsv_save_path}/{train_tsv}", sep='\t')

    # 设置每种label最多抽取的样本数
    sample_size_per_label = [1, 5, 10, 20, 30, 50, 100, 200, 300]
    for s_size in sample_size_per_label:
        # 对每个label分组并随机抽样
        sampled_df = df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(min(len(x), s_size), random_state=42)  # 设置random_state保证可重复性
        ).reset_index(drop=True)

        # 保存到新的TSV文件
        sampled_df.to_csv(f'{tsv_save_path}/train_dataset_{s_size}.tsv', index=False, sep='\t')

    logging.info(f"finish generating fine_tuning's mini train dataset.\nPlease check in {tsv_save_path}")


def enhance_based_tsv(logging, file_folder, enhance_file, enhance_factor=1):
    # file_folder: the tsv folder
    # enhance_file: 'train' or 'valid' or 'test' file to enhance
    # enhance_factor: augmentation factor
    dataset = []
    columns = {}
    # 待增强的tsv文件
    filename = f"{file_folder}/{enhance_file}_dataset.tsv"
    # enhanced file的前缀
    file_prefix = f"enhance_{str(enhance_factor)}_{enhance_file}"

    logging.info(f"begin enhance: {filename}")
    logging.info(f"enhanced file: {file_folder}/{file_prefix}_dataset.tsv")

    if not os.path.exists(filename):
        logging.info(f"{filename} doesnt exist, please generate first")
        raise FileNotFoundError(f"{filename} doesnt exist, please generate first")
    if os.path.exists(f"{file_folder}/{file_prefix}_dataset.tsv"):
        logging.info(f"enhanced file {file_folder}/{file_prefix}_dataset.tsv already exists")
        raise FileExistsError(f"enhanced file {file_folder}/{file_prefix}_dataset.tsv already exists")

    with open(filename, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line[:-1].split("\t")
            tgt = int(line[columns["label"]])
            text_a = line[columns["text_a"]]
            text_list = text_a.split("[SEP]")[1:]
            for _ in range(enhance_factor):
                # IPID:4, src:12, dst: 16, sport:20, dport:22, seq:24, ack:28
                IP, proto, first_forward_datagrams, first_backward_datagrams = None, None, None, None
                datagramss = []
                for i in range(len(text_list)):
                    pac = text_list[i]
                    datagrams = pac.split(" ")[1:-1]
                    datagramss.append(datagrams)
                    if i == 0:
                        if datagrams[0][0] == "4":
                            IP = 4
                            if datagrams[9][:2] == "06":
                                proto = 6
                            elif datagrams[9][:2] == "11":
                                proto = 17
                        elif datagrams[0][0] == "6":
                            IP = 6
                            if datagrams[6][:2] == "06":
                                proto = 6
                            elif datagrams[6][:2] == "11":
                                proto = 17
                        src = datagrams[12:16]
                        first_forward_datagrams = datagrams
                    if datagrams[12:16] != src and first_backward_datagrams is None:
                        first_backward_datagrams = datagrams

                if IP == None or proto == None:
                    logging.info(line)
                    return

                if IP == 4:
                    rsrc = random_field(32)
                    rdst = random_field(32)
                    rsrcid = random_field(16)
                    rdstid = random_field(16)
                elif IP == 6:
                    logging.info("IPV6 is waiting to process")
                    continue
                if proto == 6:
                    rsrcp = random_field(16)
                    rdstp = random_field(16)
                    rsrcseq = random_field(32)
                    rdstseq = random_field(32)
                elif proto == 17:
                    rsrcp = random_field(16)
                    rdstp = random_field(16)

                forward_4tstr = hex(rsrc)[2:].zfill(8) + hex(rdst)[2:].zfill(8) + hex(rsrcp)[2:].zfill(4) + hex(rdstp)[
                                                                                                            2:].zfill(4)
                backward_4tstr = hex(rdst)[2:].zfill(8) + hex(rsrc)[2:].zfill(8) + hex(rdstp)[2:].zfill(4) + hex(rsrcp)[
                                                                                                             2:].zfill(
                    4)
                srcipid = int(first_forward_datagrams[4], 16)
                if first_backward_datagrams is not None:
                    dstipid = int(first_backward_datagrams[4], 16)
                srcseq = int(first_forward_datagrams[24] + first_forward_datagrams[26], 16)
                srcack = int(first_forward_datagrams[28] + first_forward_datagrams[30], 16)

                dstseq = False
                if first_backward_datagrams is not None:
                    dstseq = int(first_backward_datagrams[24] + first_backward_datagrams[26], 16)
                elif srcack != 0:
                    dstseq = srcack
                # else:
                #    logging.info("cant process dstseq...")
                # logging.info(hex(srcseq),hex(dstseq),hex(rsrcseq),hex(rdstseq))

                # logging.info(f"{forward_4tstr}, {backward_4tstr}")
                for i in range(len(datagramss)):
                    # logging.info("------")
                    # logging.info(datagramss[i])
                    if datagramss[i][12:16] == src:  # forward
                        datagramss[i][11] = datagramss[i][11][:2] + forward_4tstr[:2]
                        cc = 12
                        for elm in bigram_generation(forward_4tstr, token_len=len(forward_4tstr) / 2).split(" ")[:-1]:
                            datagramss[i][cc] = elm
                            cc += 1
                        datagramss[i][cc] = forward_4tstr[-2:] + datagramss[i][cc][2:4]
                        # handle IPID
                        if IP == 4:
                            if srcipid != 0:
                                temp = hex((int(datagramss[i][4], 16) - srcipid + rsrcid) % (2 ** 16))[2:].zfill(4)
                                datagramss[i][4] = temp
                                datagramss[i][3] = datagramss[i][3][:2] + temp[:2]
                                datagramss[i][5] = temp[2:] + datagramss[i][5][2:]
                        # handle seq
                        if proto == 6:
                            tempsrcseq = hex(
                                (rsrcseq + int(datagramss[i][24] + datagramss[i][26], 16) - srcseq) % (2 ** 32))[
                                         2:].zfill(8)
                            datagramss[i][23] = datagramss[i][23][:2] + tempsrcseq[:2]
                            cc = 24
                            for elm in bigram_generation(tempsrcseq, len(tempsrcseq) / 2).split(" ")[:-1]:
                                datagramss[i][cc] = elm
                                cc += 1
                            datagramss[i][cc] = tempsrcseq[-2:] + datagramss[i][cc][2:4]
                            # handle ack
                            if int(datagramss[i][28] + datagramss[i][30], 16) != 0 and dstseq:
                                tempsrcack = hex(
                                    (rdstseq + int(datagramss[i][28] + datagramss[i][30], 16) - dstseq) % (2 ** 32))[
                                             2:].zfill(8)
                                datagramss[i][27] = datagramss[i][27][:2] + tempsrcack[:2]
                                cc = 28
                                for elm in bigram_generation(tempsrcack, len(tempsrcack) / 2).split(" ")[:-1]:
                                    datagramss[i][cc] = elm
                                    cc += 1
                                datagramss[i][cc] = tempsrcack[-2:] + datagramss[i][cc][2:4]

                    else:
                        datagramss[i][11] = datagramss[i][11][:2] + backward_4tstr[:2]
                        cc = 12
                        for elm in bigram_generation(backward_4tstr, token_len=len(backward_4tstr) / 2).split(" ")[:-1]:
                            datagramss[i][cc] = elm
                            cc += 1
                        datagramss[i][cc] = backward_4tstr[-2:] + datagramss[i][cc][2:4]
                        # handle IPID
                        if IP == 4:
                            if dstipid != 0:
                                temp = hex((int(datagramss[i][4], 16) - dstipid + rdstid) % (2 ** 16))[2:].zfill(4)
                                datagramss[i][4] = temp
                                datagramss[i][3] = datagramss[i][3][:2] + temp[:2]
                                datagramss[i][5] = temp[2:] + datagramss[i][5][2:]
                        # handle seq
                        if proto == 6:
                            if dstseq:
                                tempdstseq = hex(
                                    (rdstseq + int(datagramss[i][24] + datagramss[i][26], 16) - dstseq) % (2 ** 32))[
                                             2:].zfill(8)
                                datagramss[i][23] = datagramss[i][23][:2] + tempdstseq[:2]
                                cc = 24
                                for elm in bigram_generation(tempdstseq, len(tempdstseq) / 2).split(" ")[:-1]:
                                    datagramss[i][cc] = elm
                                    cc += 1
                                datagramss[i][cc] = tempdstseq[-2:] + datagramss[i][cc][2:4]
                            # handle ack
                            if int(datagramss[i][28] + datagramss[i][30], 16) != 0:
                                tempdstack = hex(
                                    (rsrcseq + int(datagramss[i][28] + datagramss[i][30], 16) - srcseq) % (2 ** 32))[
                                             2:].zfill(8)
                                datagramss[i][27] = datagramss[i][27][:2] + tempdstack[:2]
                                cc = 28
                                for elm in bigram_generation(tempdstack, len(tempdstack) / 2).split(" ")[:-1]:
                                    datagramss[i][cc] = elm
                                    cc += 1
                                datagramss[i][cc] = tempdstack[-2:] + datagramss[i][cc][2:4]
                    # logging.info(datagramss[i])

                newtext_a = ''
                for i in range(len(datagramss)):
                    if newtext_a != '':  # 2024.4.23 add
                        newtext_a += ' '
                    newtext_a += '[SEP]'
                    for j in range(len(datagramss[i])):
                        if newtext_a != '':
                            newtext_a += ' '
                        newtext_a += datagramss[i][j]

                dataset.append([newtext_a, tgt])

    dataset = pd.DataFrame(dataset, columns=['datagram', 'label'])
    dataset = dataset.sample(frac=1)
    # logging.info(dataset.head())

    logging.info(f"saving {file_folder}/{file_prefix}_dataset.tsv...")
    write_dataset_tsv(dataset['datagram'], dataset['label'], file_folder, file_prefix)
