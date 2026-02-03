import os
import numpy as np
import json
import click
import logging



def run_character_list_update(dir_labels, out, current_character_list):
    ls_labels = os.listdir(dir_labels)
    ls_labels = [ind for ind in ls_labels if ind.endswith('.txt')]
    
    if current_character_list:
        with open(current_character_list, 'r') as f_name:
            characters = json.load(f_name)
            
        characters = set(characters)
    else:
        characters = set()


    for ind in ls_labels:
        label  = open(os.path.join(dir_labels,ind),'r').read().split('\n')[0]
            
        for char in label:
            characters.add(char)
                
                
    characters = sorted(list(set(characters)))

    with open(out, 'w') as f_name:
        json.dump(characters, f_name)
        

@click.command()
@click.option(
    "--dir_labels",
    "-dl",
    help="directory of labels which are txt files",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--current_character_list",
    "-ccl",
    help="current exsiting character list which is txt file and wished to be updated with a set of labels",
    type=click.Path(exists=True, file_okay=True),
    required=False,
)
@click.option(
    "--out",
    "-o",
    help="output file which is a txt file where generated or updated character list will be written",
    type=click.Path(exists=False, file_okay=True),
)

def main(dir_labels, out, current_character_list):
    run_character_list_update(dir_labels, out, current_character_list)
    
