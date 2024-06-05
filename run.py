import os


os.system('python prot_bert_advanced.py -p ./data/train.txt -m single -n prot_xlnet ')
os.system('python prot_bert_advanced.py -p ./data/test.txt -m single -n prot_xlnet ')
os.system('python prot_bert_advanced.py -p ./data/test.txt -m single -n prot_albert')
os.system('python prot_bert_advanced.py -p ./data/train.txt -m single -n prot_albert')
