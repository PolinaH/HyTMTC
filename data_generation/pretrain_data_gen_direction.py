import os,sys
import random
import shutil
import binascii
import scapy.all as scapy
from functools import reduce
from flowcontainer.extractor import extract
from utils import *
from data_generation.loss_retransmit_outoforder import *
from tqdm import tqdm
import multiprocessing as mp
import traceback
import numpy as np
import pudb


def user_excepthook(tp, val, tb):
    # print the exception to standard error
    traceback.print_exc()

# Semantic lossless enhancement
def enhancement(packets, is_addr=True, is_port=True,):
    # IP6 FLOWLABEL,src,dst
    # IP：IPID，src, dst
    # TCP: seq, ack, sport, dport, 
    # UDP: sport, dport,
    if not hasattr(packets[0],'type'): # some datasets do not have ether header
        new_packets = scapy.PacketList()
        for i in range(len(packets)):
            if packets[i].src == packets[0].src: # ICMP
                # create an ether header
                ether = scapy.Ether(src="00:00:00:00:00:00", dst="ff:ff:ff:ff:ff:ff", type=0x0800)
            else:
                ether = scapy.Ether(src="ff:ff:ff:ff:ff:ff", dst="00:00:00:00:00:00", type=0x0800)
            packet = ether/packets[i]
            new_packets.append(packet)
    packets = new_packets
    first_forward_packet = packets[0].copy()
    for packet_index in range(len(packets)):
        if packets[packet_index].src != first_forward_packet.src:
            first_backward_packet = packets[packet_index].copy()
            break
    if first_forward_packet.type == 0x0800:
        replace_src = random_ipv4()
        replace_dst = random_ipv4()
        replace_src_id = random_field(16)
        replace_dst_id = random_field(16)
        if first_forward_packet.payload.proto == 6:
            replace_sport = random_field(16)
            replace_dport = random_field(16)
            replace_src_seq = random_field(32)
            replace_dst_seq = random_field(32)
        elif first_forward_packet.payload.proto == 17:
            replace_sport = random_field(16)
            replace_dport = random_field(16)

    elif first_forward_packet.type == 0x86dd:
        replace_src = random_ipv6()
        replace_dst = random_ipv6()
        replace_src_flowlabel = random_field(20)
        replace_dst_flowlabel = random_field(20)
        if first_forward_packet.payload.nh == 6:
            replace_sport = random_field(16)
            replace_dport = random_field(16)
            replace_src_seq = random_field(32)
            replace_dst_seq = random_field(32)
        elif first_forward_packet.payload.nh == 17:
            replace_sport = random_field(16)
            replace_dport = random_field(16)

    for packet_index in range(len(packets)):
        if packets[packet_index].src == first_forward_packet.src: #forward
            if is_addr:
                packets[packet_index].payload.src = replace_src
                packets[packet_index].payload.dst = replace_dst
            if packets[packet_index].type == 0x0800:
                if first_forward_packet.payload.id!=0: 
                    packets[packet_index].payload.id =  replace_src_id + (packets[packet_index].payload.id - first_forward_packet.payload.id)
                    packets[packet_index].payload.id %= 2**16
                #print("forward: ",packets[packet_index].payload.id)
                if packets[packet_index].payload.proto == 6:
                    if is_port:
                        packets[packet_index].payload.payload.sport = replace_sport
                        packets[packet_index].payload.payload.dport = replace_dport
                    packets[packet_index].payload.payload.seq = replace_src_seq + (packets[packet_index].payload.payload.seq - first_forward_packet.payload.payload.seq)
                    if not ("S" in packets[packet_index].payload.payload.flags and packets[packet_index].payload.payload.ack==0):
                        packets[packet_index].payload.payload.ack = replace_dst_seq + (packets[packet_index].payload.payload.ack - first_backward_packet.payload.payload.seq)
                    packets[packet_index].payload.payload.seq %= 2**32
                    packets[packet_index].payload.payload.ack %= 2**32
                elif packets[packet_index].payload.proto == 17:
                    if is_port:
                        packets[packet_index].payload.payload.sport = replace_sport
                        packets[packet_index].payload.payload.dport = replace_dport

            elif packets[packet_index].type == 0x86dd:
                packets[packet_index].payload.fl =  replace_src_flowlabel + (packets[packet_index].payload.fl - first_forward_packet.payload.fl)
                packets[packet_index].payload.fl %= 2**20
                if packets[packet_index].payload.nh == 6:
                    if is_port:
                        packets[packet_index].payload.payload.sport = replace_sport
                        packets[packet_index].payload.payload.dport = replace_dport
                    packets[packet_index].payload.payload.seq = replace_src_seq + (packets[packet_index].payload.payload.seq - first_forward_packet.payload.payload.seq)
                    if not ("S" in packets[packet_index].payload.payload.flags and packets[packet_index].payload.payload.ack==0):
                        packets[packet_index].payload.payload.ack = replace_dst_seq + (packets[packet_index].payload.payload.ack - first_backward_packet.payload.payload.seq)
                    packets[packet_index].payload.payload.seq %= 2**32
                    packets[packet_index].payload.payload.ack %= 2**32
                elif packets[packet_index].payload.nh == 17:
                    if is_port:
                        packets[packet_index].payload.payload.sport = replace_sport
                        packets[packet_index].payload.payload.dport = replace_dport
        else: # backward
            if is_addr:
                packets[packet_index].payload.src = replace_dst
                packets[packet_index].payload.dst = replace_src
            if packets[packet_index].type == 0x0800:
                packets[packet_index].payload.id =  replace_dst_id + (packets[packet_index].payload.id - first_backward_packet.payload.id)
                packets[packet_index].payload.id %= 2**16
                #print("backward: ",replace_dst_id, packets[packet_index].payload.id)
                if packets[packet_index].payload.proto == 6:
                    if is_port:
                        packets[packet_index].payload.payload.sport = replace_dport
                        packets[packet_index].payload.payload.dport = replace_sport
                    packets[packet_index].payload.payload.seq = replace_dst_seq + (packets[packet_index].payload.payload.seq - first_backward_packet.payload.payload.seq)
                    if not ("S" in packets[packet_index].payload.payload.flags and packets[packet_index].payload.payload.ack==0):
                        packets[packet_index].payload.payload.ack = replace_src_seq + (packets[packet_index].payload.payload.ack - first_forward_packet.payload.payload.seq)
                    packets[packet_index].payload.payload.seq %= 2**32
                    packets[packet_index].payload.payload.ack %= 2**32
                elif packets[packet_index].payload.proto == 17:
                    if is_port:
                        packets[packet_index].payload.payload.sport = replace_dport
                        packets[packet_index].payload.payload.dport = replace_sport
            elif packets[packet_index].type == 0x86dd:
                packets[packet_index].payload.fl =  replace_dst_flowlabel + (packets[packet_index].payload.fl - first_backward_packet.payload.fl)
                packets[packet_index].payload.fl %= 2**20
                if packets[packet_index].payload.nh == 6:
                    if is_port:
                        packets[packet_index].payload.payload.sport = replace_sport
                        packets[packet_index].payload.payload.dport = replace_dport
                    packets[packet_index].payload.payload.seq = replace_src_seq + (packets[packet_index].payload.payload.seq - first_forward_packet.payload.payload.seq)
                    if not ("S" in packets[packet_index].payload.payload.flags and packets[packet_index].payload.payload.ack==0):
                        packets[packet_index].payload.payload.ack = replace_src_seq + (packets[packet_index].payload.payload.ack - first_forward_packet.payload.payload.seq)
                    packets[packet_index].payload.payload.seq %= 2**32
                    packets[packet_index].payload.payload.ack %= 2**32
                elif packets[packet_index].payload.nh == 17:
                    if is_port:
                        packets[packet_index].payload.payload.sport = replace_sport
                        packets[packet_index].payload.payload.dport = replace_dport
    return packets


def change_directionlist_to_str(alist):
    astring = ''
    for item in alist:
        if item >= 0:
            astring += '+'
            astring += str(item)
        else:
            astring += str(item)
        astring += ' '
    return astring


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


def get_bursts(label_pcap, dataset_name, select_packet_len, corpora_file_folder, start_index=0, add_network_err="False", is_multi=False):
    packets = scapy.rdpcap(label_pcap)

    if add_network_err == "True":
        random_pro = np.random.uniform(0, 1)
        if 0 < random_pro < 0.1:  # loss
            packets = loss_pkt(packets)
        elif 0.1 <= random_pro < 0.2:  # retransmit
            packets = retransmit(packets)
        elif 0.2 <= random_pro < 0.3:  # out of order
            packets = outoforder(packets)
        else:
            pass

    No_ether = False
    if not hasattr(packets[0],'type'): #no ether header
        No_ether = True

    if (not No_ether and packets[0].type == 0x86dd) or (No_ether and packets[0].version == 6): #not handle ipv6
        return 0
    if len(packets) < 2:
        return 0

    packets = random_ip(packets)  # 随机src ip,dst ip，但是客户端服务端相对关系不变

    packet_direction_length = []
    packet_direction = []
    feature_result = extract(label_pcap)
    for key in feature_result.keys():
        value = feature_result[key]  # value解析pcap中所有包，value.ip_lengths获取每个包的长度和方向eg.[61,-373]
        packet_direction_length = [x for x in value.ip_lengths]
        packet_direction = [x//abs(x) for x in value.ip_lengths]

    if len(packet_direction) != len(packets):
        return 0
    # burst_txt = ''

    # 最多100个数据包1个scapy.PacketList()，拆分了数据包数量超过100的packets
    # 切分时只能在packet_directionss中方向变化的位置切，所以1个scapy.PacketList()中的数据包数量可能少于100个
    # packetss和packet_directionss中存储拆分后的数据
    packetss = []
    packet_direction_lengthss = []
    packet_directionss = []
    packet_index = 0
    packets_cnt = len(packets)
    while packet_index < packets_cnt:
        end = min(packet_index + 100, packets_cnt)
        # 如果切割点正好在一段相同值中间，就往前找一个“值变了”的位置切
        while end < packets_cnt and packet_direction[end - 1] == packet_direction[end]:
            end -= 1
            if end == packet_index:
                # 极端情况：100 个值完全一样，无法切分，强行切
                end = packet_index + 100
                break
        packetss.append(packets[packet_index:end])
        packet_direction_lengthss.append(packet_direction_length[packet_index:end])
        packet_directionss.append(packet_direction[packet_index:end])
        packet_index = end


    # packets中存储的对象是使用scapy中格式存储的数据包，1个packets中包含最多100个数据包
    # packetss中存储的对象是scapy.PacketList()，1个packetss中包含多个packets
    # 逐一处理packetss中的scapy.PacketList()
    burst_txt = ''
    for pp in range(len(packetss)):
        packets = packetss[pp]
        packet_direction_length = packet_direction_lengthss[pp]
        packet_direction = packet_directionss[pp]

        tmp_burst_txt = ''
        direction_length_string = change_directionlist_to_str(packet_direction_length)
        for packet_index in range(len(packets)):
            packet_data = packets[packet_index].copy()  # 一个数据包
            # 数据包转16进制,type(data)=bytes,eg. b'45000...'
            data = (binascii.hexlify(bytes(packet_data)))

            if No_ether:  # 为没有ethernet layer的数据包手动添加14个字节的ethernet layer
                # 将字节解码成字符串,type(packet_string)=str,eg. '45000...'
                packet_string = data.decode()
                if packet_direction[packet_index] == 1:  # 该数据包方向为+
                    packet_string = "c49a025996f8e46f13e2e3ae0800" + packet_string
                else:  # 该数据包方向为-
                    packet_string = "e46f13e2e3aec49a025996f80800" + packet_string
                packet_string = packet_string[2*start_index:2*(start_index+select_packet_len)]
            else:
                packet_string = data.decode()[2*start_index:2*(start_index+select_packet_len)]

            # 如果不按照最大100个数据包进行拆分，则1个label_pcap文件对应的packetss中只有1个packets，也就只有1条新流
            if packet_index == 0:
                tmp_burst_txt += '\n'
                direction_length_string = "||" + direction_length_string #a new flow
                tmp_burst_txt += direction_length_string
                tmp_burst_txt += '\n'
                tmp_burst_txt += packet_string
                tmp_burst_txt += '\n'
            else:
                if packet_direction[packet_index] != packet_direction[packet_index - 1]:
                    tmp_burst_txt += '\n'
                tmp_burst_txt += packet_string
                tmp_burst_txt += '\n'

        burst_txt += tmp_burst_txt
        #with open(f"{corpora_path[:-4]}_{pp}.txt",'a') as f1:
        #    f1.write(tmp_burst_txt)

    if is_multi:
        pid = os.getpid()
        with open(f"{corpora_file_folder}/corpora_files_{dataset_name}_{pid}.txt",'a') as f:
            f.write(burst_txt)
    else:
        with open(f"{corpora_file_folder}/corpora_files_{dataset_name}.txt",'a') as f:
            f.write(burst_txt)

    return 0

def get_consecutive_packets(label_pcap, select_packet_len, corpora_path,start_index = 0):
    packets = scapy.rdpcap(label_pcap)

    if not hasattr(packets[0],'type'):
        print("No ethernet...")
        return 0
        
    packet_direction = []
    feature_result = extract(label_pcap)
    for key in feature_result.keys():
        value = feature_result[key]
        packet_direction = [x // abs(x) for x in value.ip_lengths]

    if len(packet_direction) == len(packets):
        burst_txt = ''
        burst_direction = ''
        for packet_index in range(len(packets)):
            packet_data = packets[packet_index].copy()
            data = (binascii.hexlify(bytes(packet_data)))
            
            packet_string = data.decode()[2*start_index:2*(start_index+select_packet_len)]
            
            if packet_index == 0:
                burst_txt += packet_string
                burst_txt += '\n'
            else:
                burst_txt += packet_string
                burst_txt += '\n'
                burst_txt += '\n'
                burst_txt += packet_string
                if packet_direction[packet_index] != packet_direction[packet_index - 1]:
                    burst_direction += '0'
                else:
                    burst_direction += '1'
        
        with open(corpora_path,'a') as f:
            f.write(burst_txt)
        with open(corpora_path[:-4]+"_extra.txt",'a') as f: 
            f.write(burst_direction)
    return 0


def merge(path):
    with open(f"{path}_all.txt",'w') as fw1:
        for idx_file, file in enumerate(os.listdir(path)):
            if idx_file != 0:
                fw1.write('\n')
            with open(f"{path}/{file}",'r') as fr:
                while True:
                    line = fr.readline()
                    if not line:
                        break
                    fw1.write(line)
            
def pretrain_dataset_generation(logging,dataset_name,pcapng_path,pcap_path,pcap_split_path,
                                corpora_file_folder,folder_is_pcap_label=False,start_index=0,
                                select_packet_len=64,add_network_err="False", max_process=False):
    # pcapng_path: the path of pcapng files (if the traffic is the pacp type, pcapng_path = pcap_output_path)
    # pcap_output_path: the path of pcap files
    # pcap_split_path: the path of splited pcap files
    # start_index: the index of start byte (i.e., If start from IP header, start_index = 14)
    # select_packet_len: the bytes of used
    # enhance_factor: enhance factor, enhance_factor = 1 represents do not enhance the pretrain data
    # max_process: how many multi process to use. 0: no use
    # folder_is_pcap_label: whether a folder refers to a class

    # (1) pcapng转为pcap
    # (2) 变成每个文件夹是一类
    if len(os.listdir(pcap_path)) == 0:
        convert_pcapng_to_pcap(logging,pcapng_path, pcap_path, folder_is_pcap_label)

    # split pcap
    for c in os.listdir(pcap_path):
        if not os.path.exists(f"{pcap_split_path}/splitcap/{c}"):
            for file in os.listdir(f"{pcap_path}/{c}"):
                split_cap(logging, pcap_split_path, f"{pcap_path}/{c}", file, pcap_label=c)
        else:
            logging.info(f"{pcap_split_path}/splitcap/{c} already exists, skipping splitting process")

    logging.info("Begin generating corpora file (burst dataset)...")
    if not os.path.exists(corpora_file_folder):
        os.makedirs(corpora_file_folder)
    if max_process != 0:
        all_files = []
        for c in os.listdir(f"{pcap_split_path}/splitcap"):
            for file in os.listdir(f"{pcap_split_path}/splitcap/{c}"):
                all_files.append(f"{pcap_split_path}/splitcap/{c}/{file}")

        pbar = tqdm(total=len(all_files))
        pbar.set_description('get bursts')
        update = lambda *args: pbar.update()

        pool = mp.Pool(max_process)

        for file in all_files:
            pool.apply_async(get_bursts, (file, dataset_name, select_packet_len, corpora_file_folder,
                                          start_index, add_network_err, True), callback=update)
        pool.close()
        pool.join()

    else:  # max_process == 0
        all_files = []
        for c in os.listdir(f"{pcap_split_path}/splitcap"):
            for file in os.listdir(f"{pcap_split_path}/splitcap/{c}"):
                all_files.append(f"{pcap_split_path}/splitcap/{c}/{file}")
        for file in tqdm(all_files):
            get_bursts(file,dataset_name,select_packet_len=select_packet_len,
                       corpora_file_folder=corpora_file_folder,start_index=start_index,
                       add_network_err=add_network_err, is_multi=False)

    return 0


def corpora_to_bigram(corpora_path,corpora_bigram_path):
    with open(corpora_bigram_path,'w') as fw:
        with open(corpora_path,'r') as fr:
            line = fr.readline()
            while True:
                line = fr.readline()
                if not line:
                    break
                if not line.strip():
                    fw.write(line)
                else:
                    newline = bigram_generation(line.strip(), token_len=len(line.strip()))
                    if newline[:2] == "||":
                        newline = "||"+newline[5:]
                    fw.write(newline+"\n")       


def corpora_to_gram(corpora_path,corpora_gram_path):
    with open(corpora_gram_path,'w') as fw:
        with open(corpora_path,'r') as fr:
            line = fr.readline()
            while True:
                line = fr.readline()
                if not line:
                    break
                if not line.strip():
                    fw.write(line)
                else:
                    if line[:2] == "||":
                        fw.write(line)
                        line = fr.readline()
                        newline = gram_generation(line.strip())
                    else:
                        newline = gram_generation(line.strip())
                    fw.write(newline+"\n")         

def read_flows(path):
    print("process ",path)
    file1 = []
    with open(path,'r') as fr:
        flow = []
        while True:
            line = fr.readline()
            if not line:
                break
            if line[:2] == "||":
                if len(flow)>0:
                    file1.append(flow)
                flow = []
                flow.append(line)
            else:
                flow.append(line)
        if len(flow)>0:
            file1.append(flow)
    return file1

def merge_txts():
    corpora_path1 = "corpora1.txt"
    corpora_path2 = "corpora2.txt"
    corpora_path3 = "corpora3.txt"
    corpora_path = "corpora.txt"
    file1 = read_flows(corpora_path1)
    for flow in file1:
        if len(flow)>100000:
            print(len(flow))

    file2 = read_flows(corpora_path2)
    for flow in file2:
        if len(flow)>100000:
            print(len(flow))
    file3 = read_flows(corpora_path3)
    for flow in file3:
        if len(flow)>100000:
            print(len(flow))
    files = file1 + file2 + file3
    random.shuffle(files)
    return
