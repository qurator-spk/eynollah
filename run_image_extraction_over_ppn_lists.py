import os
import sys

dir_ppn = '/home/vahid/Documents/eynollah/ppn_list.txt'


with open(dir_ppn) as f:
    ppn_list = f.readlines()
    
    
ppn_list = [ind.split('\n')[0] for ind in ppn_list]

url_main = 'https://content.staatsbibliothek-berlin.de/dc/download/zip?ppn=PPN'

out_result = './new_results_ppns2'


for ppn_ind in ppn_list:
    url = url_main + ppn_ind
    #curl -o ./ppn.zip "https://content.staatsbibliothek-berlin.de/dc/download/zip?ppn=PPN1762638355"
    os.system("curl -o "+"./PPN_"+ppn_ind+".zip"+" "+url)
    os.system("unzip "+"PPN_"+ppn_ind+".zip"+ " -d "+"PPN_"+ppn_ind)
    os.system("rm -rf "+"PPN_"+ppn_ind+"/*.txt")
    
    os.system("mkdir "+out_result+'/'+"PPN_"+ppn_ind+"_out")
    os.system("mkdir "+out_result+'/'+"PPN_"+ppn_ind+"_out_images")
    command_eynollah = "eynollah -m /home/vahid/Downloads/models_eynollah_renamed_savedmodel -di "+"PPN_"+ppn_ind+" "+"-o "+out_result+'/'+"PPN_"+ppn_ind+"_out "+"-eoi "+"-ep -si "+out_result+'/'+"PPN_"+ppn_ind+"_out_images"
    os.system(command_eynollah)
    
    os.system("rm -rf "+"PPN_"+ppn_ind+".zip")
    os.system("rm -rf "+"PPN_"+ppn_ind)
    #sys.exit()
    
