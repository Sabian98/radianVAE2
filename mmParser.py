
import numpy as np
import os
from Bio.PDB.PDBParser import PDBParser
p = PDBParser(PERMISSIVE=1)
import math
from Bio.PDB.Polypeptide import PPBuilder
import Bio.PDB


file=open("/scratch/trahman2/list.txt")

f=file.readlines()
file.close()

line="ALA,ARG,ASN,ASP,CYS,GLN,GLU,GLY,HIS,ILE,LEU,LYS,MET,PHE,PRO,SER,THR,TRP,TYR,VAL"
fr=line.split(",")
atom_dict={}
val=0
for elem in fr:
    atom_dict[elem]=val
    val+=1


for elem in f:

    
    arr=[]
    cont=elem.strip("\n")
    structure_id = cont[:4]
    filename = "/scratch/trahman2/ext_structures/"+cont#folder containing unzipped pdb files
    structure = p.get_structure(structure_id, filename)#get structure
    model = structure[0]#get the first model only
    f_arr=[]
    out_folder="/scratch/trahman2/maps/dihedral_coords/"
    out=out_folder+cont
    for chain in model:
        if chain.get_id()=="A":#search for a particular chain ID
            polypeptides = Bio.PDB.PPBuilder().build_peptides(chain)
            for poly_index, poly in enumerate(polypeptides) :

                phi_psi = poly.get_phi_psi_list()
                for res_index, residue in enumerate(poly) :
                    arr=[]

                   
                    if residue.resname in atom_dict:
                        res=residue.resname
                        a=phi_psi[res_index][0]
                        b=phi_psi[res_index][1]
                        if (a!= None):
                            a=math.degrees(phi_psi[res_index][0])
                        else:
                            a=0

                        if (b!= None):
                            b=math.degrees(phi_psi[res_index][1])
                        else:
                            b=0
                        arr.append(round(a,2))
                        arr.append(round(b,2))
                    f_arr.append(arr)
 


    
    print(len(f_arr))
    np.savetxt(out,f_arr)








