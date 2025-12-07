import r2pipe
import re
import os
import pickle
import networkx as nx
import subprocess

from tqdm import tqdm
from queue import Queue


RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"

class AddressTaken:
    def __init__(self, r2, proc):
        self.r2 = r2
        self.proc = proc
        self.addr2name = {} # {addr -> func_name} 
        self.rela_info = None
        self.addr_set = set() # set (func)

        self.filter_func_set = set(['entry0', 'main', 'sym.__libc_csu_fini', 'sym.__libc_csu_init', 'sym.frame_dummy', 'sym.__do_global_dtors_aux'])

        self.filter_func_addr = set()

        self.func_quene = Queue() 
        self.func_resolved = set() 
        self.ref_as_data = {} 
        self.g = nx.DiGraph() 

        self.call_graph_save_path = f'{os.path.basename(proc)}_call_graph.pkl'
        self.address_taken_save_path = f'address_taken_cache/{os.path.basename(proc)}_address_taken.pkl' # (addr_set, addr2name, ref_as_data) -> file

        if not os.path.exists('address_taken_cache'):
            os.makedirs('address_taken_cache')

    def init(self):
        # init rela info
        self.rela_info = self.get_rela_info_readelf_r()

        # init filter func addr
        for fn in self.filter_func_set:
            func_info = self.r2.cmdj(f'afij @ {fn}')
            if len(func_info) < 1: continue
            self.filter_func_addr.add(func_info[0]['offset'])

        self.analyze()
        
    """
    rela info in radare2:
    ('0x00003da0', '0x00002da0', 'ADD_64', '0x000011a0')
    ('0x00003da8', '0x00002da8', 'ADD_64', '0x00001160')
    ('0x00003fb8', '0x00002fb8', 'SET_64', 'free')
    ('0x00003fc0', '0x00002fc0', 'SET_64', 'puts')
    
    """
    def get_rela_info_r2(self):
        rela_pattern = re.compile(r"^(0x[0-9a-fA-F]+)\s+(0x[0-9a-fA-F]+)\s+(\S+)\s+(.+)$", re.MULTILINE) 
        s = self.r2.cmd('ir')
        rela_info = rela_pattern.findall(s)
        return rela_info
    
    """
    rela info in readelf -r
    ('0000001de140', '000000000008', 'R_X86_64_RELATIVE', '1893b9')
    ('0000001de148', '000000000008', 'R_X86_64_RELATIVE', '1893d5')
    ('0000001de150', '000000000008', 'R_X86_64_RELATIVE', '1d5400')
    ('0000001de160', '000000000008', 'R_X86_64_RELATIVE', '1d55f8')
    ('0000001de168', '000000000008', 'R_X86_64_RELATIVE', '1d55a8')
    """
    def get_rela_info_readelf_r(self):
        rela_pattern = re.compile(r'^([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+(R_X86_64_RELATIVE)\s+(.+)', re.MULTILINE)
        result = subprocess.run(['readelf', '-r', self.proc], capture_output=True, text=True)
        data = result.stdout
        rela_info = rela_pattern.findall(data)
        new_rela_info = [(r[0], r[1], r[2], '0x' + r[3]) for r in rela_info]

        return new_rela_info

    def insert_func_addr(self, addr):
        self.addr_set.add(addr)

    def filter_func(self, addr):
        if addr in self.filter_func_addr:
            return False
        if addr in self.addr2name:
            func_name = self.addr2name[addr]
        else:
            func_info = self.r2.cmdj(f'afij @ {addr}')
            if len(func_info) == 0:
                return False
            func_name = func_info[0]['name']
        if 'sym.imp' in func_name:
            return False
        return True

    def analyze_rela_dyn(self):
        rela_info = self.rela_info
        for rela in tqdm(rela_info):
            # if rela[2] != 'ADD_64': continue # from radare2
            if rela[2] != 'R_X86_64_RELATIVE': continue # from readelf -r
            try_addr = rela[3]
            func_info = self.r2.cmdj(f'afij @ {try_addr}')

            if len(func_info) > 0: 
                addr = func_info[0]['offset']
                if self.filter_func(addr):
                    # print(f'{func_name} in rela')
                    self.insert_func_addr(addr)                

    def analyze_func_axt(self, addr):
        refs = self.r2.cmdj(f'axtj @ {addr}')
        for ref in refs:
            if ref['type'] == 'DATA':
                if addr not in self.ref_as_data.keys(): self.ref_as_data[addr] = set()
                if ref.get('fcn_addr') is not None:
                    self.ref_as_data[addr].add(ref['fcn_addr'])
        
        if addr in self.ref_as_data:
            self.insert_func_addr(addr)
    
    def parse_call_in_func(self, func):
        r2 = self.r2
        s = r2.cmd(f'pdsf @ {func}')
        direct_call_set = set()
        include_indirect_call = False

        pattern_fcn = r"^0x\w+.*call fcn\..*\n"
        func_matches = re.findall(pattern_fcn, s, re.MULTILINE)
        for row in func_matches:
            target = row.split(' ')[2]
            callee_name = target
            callee_addr = int(callee_name.split('.')[1].strip(), 16)

            direct_call_set.add(callee_addr)

        pattern_rest = r"^0x\w+\s+(?!.*fcn\.).*\n"
        rows = re.findall(pattern_rest, s, re.MULTILINE) 

        for row in rows:
            addr = int(row.split(' ')[0], base=16) 
            t = row.split(' ')[1] 

            if row.split(' ', 1)[1].strip() == 'call':
                syscall_info = r2.cmdj(f'pdj 1 @ {addr}')[0]['disasm']
                print(BLUE + f'{syscall_info} in {func}' + RESET)
                continue

            if t == 'call':
                target = row.split(' ')[2]
                instr_info = r2.cmdj(f'pdj 1 @{addr}')[0]
                ref = instr_info.get('jump')
                if ref != None:
                    callee_info = r2.cmdj(f'afij @{ref}')
                    if len(callee_info) == 0:
                        print(RED + f'{hex(instr_info["offset"])} : {instr_info["disasm"]}; Target address can not be resolved as a function by radare2.' + RESET)
                        continue

                    callee_info = callee_info[0]
                    callee_name = callee_info['name']

                    direct_call_set.add(ref)

                else:
                    include_indirect_call = True

        return direct_call_set, include_indirect_call
    
    def call_graph_construct(self):
        main_info = self.r2.cmdj('afij @ main')
        if len(main_info) < 1:
            return
        
        main_addr = main_info[0]['offset']
        self.func_quene.put(main_addr)
        self.func_resolved.add(main_addr)

        while not self.func_quene.empty():
            print(f'function resolve quene length is : {self.func_quene.qsize()}')
            func = self.func_quene.get()
            direct_call_set, include_indirect_call = self.parse_call_in_func(func)
            for dc in direct_call_set:
                if self.filter_func(dc): 
                    self.g.add_edge(func, dc)
            
            dif = direct_call_set - self.func_resolved
            self.func_resolved |= dif
            for dc in dif: self.func_quene.put(dc)

            if include_indirect_call:
                for idc in self.addr_set: self.g.add_edge(func, idc)

                dif = self.addr_set - self.func_resolved
                self.func_resolved |= dif
                for idc in dif: self.func_quene.put(idc)
        
        self.g.graph['address_taken'] = self.addr_set
        self.g.graph['ref_as_data'] = self.ref_as_data
        self.g.graph['addr2name'] = self.addr2name

        print(f'call graph construct done, save to {self.call_graph_save_path}')
        with open(f'{self.call_graph_save_path}', 'wb') as fd:
            pickle.dump(self.g, fd)
    
    def analyze(self):
        self.analyze_rela_dyn()
        
        func_list = self.r2.cmdj('aflj')
        for func in tqdm(func_list):
            func_name = func['name']
            addr = func['offset']
            self.addr2name[addr] = func_name

            if self.filter_func(addr):
                # print(f'{func_name} in function axt')
                self.analyze_func_axt(addr)

        with open(self.address_taken_save_path, 'wb') as fd:
            pickle.dump((self.addr_set, self.addr2name, self.ref_as_data), fd)

        print(f'address taken saves to {self.address_taken_save_path}')
    
        # self.call_graph_construct()

    def prune_at_in_callgraph(self, at, main_addr):
        
        if not self.g.has_node(at): return

        prune_at_set = set([at])
        self.g.remove_node(at)

        zero_indegree_nodes = set([node for node, ind in self.g.in_degree() if ind == 0])

        while len(zero_indegree_nodes) > 1:
            zero_indegree_nodes.remove(main_addr)
            for nd in zero_indegree_nodes:
                self.g.remove_node(nd)
                print(BLUE + f'delete address taken {self.addr2name[nd]}' + RESET)
                self.addr_set.remove(nd)
            
            zero_indegree_nodes = set([node for node, ind in self.g.in_degree() if ind == 0])
    
    def prune(self):
        if not os.path.exists(self.call_graph_save_path):
            print(RED + 'Please analyze binary first !!!' + RESET)
            exit(-1)
        
        with open(f'{self.call_graph_save_path}', 'rb') as fd:
            self.g:nx.DiGraph = pickle.load(fd)
            self.addr_set = self.g.graph['address_taken']
            self.ref_as_data = self.g.graph['ref_as_data']
            self.addr2name = self.g.graph['addr2name']

        zero_indegree_nodes = [node for node, ind in self.g.in_degree() if ind == 0]
        if len(zero_indegree_nodes) > 1:
            print(RED + 'More than one function has 0 indegree' + RESET)
            exit(-1)
        
        main_addr = zero_indegree_nodes[0]
        is_change = True

        while is_change:
            is_change = False
            node_set = set(self.g.nodes)
            for at, ref_pos in self.ref_as_data.items():
                # print(len(ref_pos), len(ref_pos & node_set))
                if len(ref_pos & node_set) == 0:
                    is_change = True
                    self.prune_at_in_callgraph(at, main_addr)
                    node_set = set(self.g.nodes)
                    
    def get_address_taken(self):
        return self.addr_set
    
def resume_node_label(g:nx.DiGraph):
    addr2name = g.graph['addr2name']
    ref_as_data = g.graph['ref_as_data']
    addr_set = g.graph['address_taken']

    def relabel_nodeid(addr):
        return addr2name[addr]
    
    g = nx.relabel_nodes(g, relabel_nodeid)
    g.graph['address_taken'] = set(map(relabel_nodeid, addr_set))
    new_ref_as_data = {relabel_nodeid(k): set([relabel_nodeid(v) for v in vset]) for k, vset in ref_as_data.items()}
    g.graph['ref_as_data'] = new_ref_as_data
    
    return g


if __name__ == '__main__':

    # binary_path = '/usr/local/nginx/sbin/nginx'
    binary_path = '/usr/lib/x86_64-linux-gnu/libc-2.31.so'
    # binary_path = '/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.28'
    # binary_path = '/root/access_control_program_level/test/indirect_jump'

    r2 = r2pipe.open(binary_path)
    r2.cmd('aaa')

    at = AddressTaken(r2, binary_path)
    at.init() 

    # at.prune()
    
    at_sz = len(at.get_address_taken())
    at_set = [at.addr2name[addr] for addr in at.get_address_taken()]
    print(f'address taken size is : {at_sz}')
    if at_sz < 100:
        print(at_set)

    print(at_set)

    # with open(f'{binary_path}_call_graph.pkl', 'rb') as fd:
    #     g:nx.DiGraph = pickle.load(fd)

    # g = resume_node_label(at.g)

    # print(len(g.edges))
    # print(' ============================= ')
    # print(len(g.nodes))
    # print(' ======================== ')
    # print(g.graph['ref_as_data'])
    # print(' ======================== ')
    # print(g.graph['address_taken'])


