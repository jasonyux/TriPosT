import re
from utils.constants import CHARACTERS


class dotdict(dict):
	def __init__(self, *args, **kwargs):
		super(dotdict, self).__init__(*args, **kwargs)
		self.__dict__ = self
		# recursively convert all dicts to dotdicts
		for key, value in self.items():
			if isinstance(value, dict):
				self[key] = dotdict(value)
			elif isinstance(value, list):
				self[key] = [dotdict(v) if isinstance(v, dict) else v for v in value]
		return

	def __getattr__(self, name):
		return self[name]


def format_multiple_choice(choices: list):
    choices_w_prefix = [f"({CHARACTERS[i]}) {choice}" for i, choice in enumerate(choices)]
    return "\n".join(choices_w_prefix).strip()


def find_sub_list(sublist, l):
    sll=len(sublist)
    for ind in (i for i,e in enumerate(l) if e==sublist[0]):
        if l[ind:ind+sll]==sublist:
            return ind, ind + sll -1
	

def findall(p:str, s:str):
    '''Yields all the positions of
    the pattern p in the string s.'''
    i = s.find(p)
    while i != -1:
        yield i
        i = s.find(p, i+1)


def extract_all_numbers(text:str):
    text = text.replace(',', '').replace('$', '').replace('%', '')
    return re.findall(r'-?\d*\.?\d+', text)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def print_red(*strs):
    s = ' '.join([str(s) for s in strs])
    print(bcolors.WARNING + s + bcolors.ENDC)
    
def print_blue(*strs):
	s = ' '.join([str(s) for s in strs])
	print(bcolors.OKBLUE + s + bcolors.ENDC)

def print_green(*strs):
	s = ' '.join([str(s) for s in strs])
	print(bcolors.OKGREEN + s + bcolors.ENDC)