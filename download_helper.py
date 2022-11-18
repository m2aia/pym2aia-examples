''' 
# Publicly available data set

The imzML files (each ~3.3 GB) contains continuous profile spectra and can be found in the Metabolights [2] repository with accession number MTBLS2639.

# References

[1] Geier,B. et al. (2021) Connecting structure and function from organisms to molecules in small-animal symbioses through chemo-histo-tomography. Proc Natl Acad Sci USA, 118, e2023773118
    https://www.pnas.org/doi/full/10.1073/pnas.2023773118
[2] https://www.ebi.ac.uk/metabolights/MTBLS2639
[3] http://teem.sourceforge.net/nrrd/format.html

'''

from typing import List
import os
from unicodedata import name
import wget
import zipfile
import pathlib



def DownloadMTBLS2639(selection: List[int] = None) -> List[str]:
    '''
    selection: List[int]
    1: "150429_ew_section1_pos.imzML"
    2: "150429_ew_section2_pos.imzML"
    3: "150505_ew_section3_pos.imzML"
    4: "150417_ew_section4_pos.imzML"
    '''
    example_imzML_files = []
    names = ["150429_ew_section1_pos","150429_ew_section2_pos", "150505_ew_section3_pos", "150417_ew_section4_pos"]
    if selection:
        names = [names[s-1] for s in selection]

    path = pathlib.Path('data')
    path.mkdir(parents=True, exist_ok=True)

    for k, i in zip(selection,names):
        if not os.path.exists(path.joinpath(f"{i}.ibd")):
            try:
                print("Download files @https://data.jtfc.de")
                wget.download(f"https://data.jtfc.de/MTBLS2639_{k}.zip", out=path)
                with zipfile.ZipFile(f"MTBLS2639_{k}.zip", 'r') as zip_ref:
                    zip_ref.extractall(path)
            except:
                print("Alternative url not found!")
        
        
        if not os.path.exists(path.joinpath(f"{i}.imzML")):
            print("Start download", f"{i}.imzML")
            wget.download(f"https://www.ebi.ac.uk/metabolights/ws/studies/MTBLS2639/download/b79faad9-e66d-4b17-bccf-bd7196f69d90?file={i}.imzML", out=path)
        
        if not os.path.exists(path.joinpath(f"{i}.ibd")):
            print("Start download", f"{i}.ibd")
            wget.download(f"https://www.ebi.ac.uk/metabolights/ws/studies/MTBLS2639/download/b79faad9-e66d-4b17-bccf-bd7196f69d90?file={i}.ibd", out=path)
        example_imzML_files.append(str(path.joinpath(f"{i}.imzML")))
        
    return example_imzML_files
