import glob
import numpy as np
import joblib
import mdtraj as md
import sys
from itertools import combinations
from biopandas.pdb import PandasPdb
import openmm as mm
from openmm import *
from openmm.app import *
from openmm.unit import *
import os
import urllib.request
import MDAnalysis as mda
import py3Dmol
import pytraj as pt
import platform
import pandas as pd
import pickle
import scipy.cluster.hierarchy
from scipy.spatial.distance import squareform
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
#from simtk.unit import Vec3
import seaborn as sb
from statistics import mean, stdev
from pytraj import matrix
from matplotlib import colors
from IPython.display import set_matplotlib_formats
import warnings
warnings.filterwarnings("ignore")
import mdtraj as md
import numpy as np
from itertools import combinations

class FrameAnalyzer:
    def __init__(self):
        self.column_names = None

    def analyze_trajectory(self, frame_file, res_indx):
        self.column_names = ['Name', 'Frame', 'ASN_Angle', 'PRO_Angle', 'TYR_Angle'] + [f"Residue Pair {idx}" for idx, _ in enumerate(combinations(res_indx, 2))]
        self.res_pair = list(combinations(res_indx, 2))

        trj = md.load(frame_file, stride=None, atom_indices=None, frame=None)
        features = []

        for frame in range(len(trj)):
            cont_feat = []

            for p in range(len(self.res_pair)):
                cont = md.compute_contacts(trj[frame], contacts=[self.res_pair[p]], scheme='ca')
                cont_feat.append(cont[0][0][0])

            topo = trj[frame].topology
            res = [rs for rs in topo.residues]
            atm = [t for t in topo.atoms]
            res_npxxy = []

            for r in range(len(res)):
                if str(res[r])[:3] == 'ASN' and str(res[r+1])[:3] == 'PRO' and str(res[r+4])[:3] == 'TYR':
                    d = (str(res[r])[3:], str(res[r+1])[3:], str(res[r+4])[3:])
                    res_npxxy.append(d)

            atom_asn, atom_pro, atom_tye = None, None, None

            for ma in range(len(atm)):
                if str(atm[ma]) == 'ASN' + res_npxxy[0][0] + '-' + 'O':
                    asn_o = ma
                if str(atm[ma]) == 'ASN' + res_npxxy[0][0] + '-' + 'C':
                    asn_c = ma
                if str(atm[ma]) == 'ASN' + res_npxxy[0][0] + '-' + 'N':
                    asn_n = ma

            atom_asn = (asn_o, asn_c, asn_n)
            asn_indx = np.array(atom_asn).reshape(3)
            asn_ = md.compute_angles(trj[frame], [asn_indx], periodic=True, opt=True)
            asn_ang = asn_[0][0]

            for mp in range(len(atm)):
                if str(atm[mp]) == 'PRO' + res_npxxy[0][1] + '-' + 'O':
                    pro_o = mp
                if str(atm[mp]) == 'PRO' + res_npxxy[0][1] + '-' + 'C':
                    pro_c = mp
                if str(atm[mp]) == 'PRO' + res_npxxy[0][1] + '-' + 'N':
                    pro_n = mp

            atom_pro = (pro_o, pro_c, pro_n)
            pro_indx = np.array(atom_pro).reshape(3)
            pro_ = md.compute_angles(trj[frame], [pro_indx], periodic=True, opt=True)
            pro_ang = pro_[0][0]

            for mt in range(len(atm)):
                if str(atm[mt]) == 'TYR' + res_npxxy[0][2] + '-' + 'O':
                    tyr_o = mt
                if str(atm[mt]) == 'TYR' + res_npxxy[0][2] + '-' + 'C':
                    tyr_c = mt
                if str(atm[mt]) == 'TYR' + res_npxxy[0][2] + '-' + 'N':
                    tyr_n = mt

            atom_tye = (tyr_o, tyr_c, tyr_n)
            tyr_indx = np.array(atom_tye).reshape(3)
            tyr_ = md.compute_angles(trj[frame], [tyr_indx], periodic=True, opt=True)
            tyr_ang = tyr_[0][0]

            g = (frame_file, frame, asn_ang, pro_ang, tyr_ang, *cont_feat)
            features.append(g)

        return features


import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

class DeltaActivityCalculator:
  def __init__(self, model_path):
    self.model = self.load_model(model_path)
    self.frame_analyzer = FrameAnalyzer()

  def load_model(self, model_path):
    with open(model_path,'rb') as file:
      model = pickle.load(file)
    return model

  def calculate_delta_activity(self, reference_frame, current_frame):
    res_indx = [(20), (48), (82), (89), (207), (215), (244), (247), (248), (251), (255)]
    reference_features = self.frame_analyzer.analyze_trajectory(reference_frame, res_indx)
    current_features = self.frame_analyzer.analyze_trajectory(current_frame,res_indx)
    print(reference_features)
    print(current_features)

    reference_features = np.array(reference_features)
    current_features = np.array(current_features)

    reference_features = reference_features[:, 2:]
    current_features = current_features[:, 2:]

    reference_features = reference_features.astype(float)
    current_features = current_features.astype(float)

    delta_activity = np.abs(current_features - reference_features)
    prediction = self.model.predict(delta_activity)

    return prediction

def save_frame_as_pdb(self, frame, output_directory):
  os.makedirs(output_directory, exist_ok=True)
  output_path = os.path.join(output_directory, 'frame.pdb')
  frame.save(output_path)
  return output_path

from selenium.webdriver.common.by import By
import os
import time
from selenium import webdriver
import tarfile
import openmm as mm
from openmm import *
from openmm.app import *
from openmm.unit import *
import pytraj as pt

from sys import stdout, exit, stderr
import os, math, fnmatch
import warnings
warnings.filterwarnings('ignore')
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

chromium_path = os.path.expanduser('~/apps/chromium/chrome-linux/chrome')

options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.binary_location = chromium_path
import shutil


# Set downloads to current directory
download_dir = os.getcwd()
prefs = {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "directory_upgrade": True
}
options.add_experimental_option("prefs", prefs)


replaced_root = os.path.abspath("../Replaced")

# Loop over all folders inside ../Replaced
for folder_name in os.listdir(replaced_root):
    folder_path = os.path.join(replaced_root, folder_name)

    if not os.path.isdir(folder_path):
        continue  # Skip non-folders

    print(f"🔍 Processing folder: {folder_name}")
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if not os.path.isfile(file_path):
            continue  # Skip subfolders

        print(f"📄 Found file: {filename}")
        equil_file = "equil_" + filename
      #below few lines of code have been automated to get the data directly from the internet. the codes open the CHARMM GUI to download all the files necessary for the equilibration
      

        wd = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        wd.get("https://www.charmm-gui.org/?doc=input/solution")
        wd.find_element("id","email").send_keys(" ")#enter your email id within the quotes, that you use on the CHARMM GUI. Create account if needed
        wd.find_element("id","password").send_keys(" ")#enter password in the quotes that you have used with your account
        wd.find_element("id","password").submit()
        wd.get("https://www.charmm-gui.org/?doc=input/solution")
        wd.find_element("xpath","//input[@type='file']").send_keys(file_path)

        wd.find_element("name","file").submit()
        # elem = WebDriverWait(wd, 10).until(
        # EC.presence_of_element_located(("id","nextBtn")))
        # time.sleep(10)
        wd.find_element("id","nextBtn").click()
        # elem = WebDriverWait(wd, 10).until(

        # EC.presence_of_element_located((By.ID, "nextBtn")))
        time.sleep(5)
        wd.find_element("id","nextBtn").click()
        time.sleep(5)
        download_button = wd.find_element(By.CSS_SELECTOR, 'a.download')
        download_button.click()
        print("✅ Download button clicked. Waiting for file to appear...")
        # wd.quit()
        # time.sleep(40)
        # open file
        download_path = os.path.join(download_dir, "charmm-gui.tgz")

        # Wait for download
        timeout = 60
        while not os.path.exists(download_path) and timeout > 0:
            time.sleep(1)
            timeout -= 1

        if not os.path.exists(download_path):
            print("Error: Download failed or took too long.")
            continue

        # Open and extract
        with tarfile.open(download_path) as file:
            file.extractall(path=folder_path)  # or wherever you want to unpack

       
        Google_Drive_Path=""
        print(file_path)

        # Specify the directory path

        # Specify the keyword to search for in the file names
        keyword = 'charmm'

        files = os.listdir(folder_path)
        matching_folders = [f for f in files if keyword in f]

        if not matching_folders:
            print(f"No CHARMM folder found in {folder_path}")
            continue

        Google_Drive_Path = os.path.join(folder_path, matching_folders[0])


        # # Specify the directory where you want to start searching
        # starting_directory = file_path
        # # Iterate through all the directories and files in the starting directory
        # for root, dirs, files in os.walk(starting_directory):
        #     # for dir_name in dirs:
        #     for file_name in files:
        #         if 'charmm' in  file_name.lower():
        #         # Found a folder with "charmm" in its name
        #           folder_path = os.path.join(root, file_name)
        #           print(f"Found 'charmm' folder: {folder_path}")
        #           Google_Drive_Path = folder_path
        PSF_filename = Google_Drive_Path+ '/step1_pdbreader.psf'
        CRD_filename = Google_Drive_Path+'/step1_pdbreader.crd'
        PDB_filename = Google_Drive_Path+'/step1_pdbreader.pdb'
        toppar_filename = Google_Drive_Path+'/toppar.str'

        # Google_Drive_Path = '/content/charmm-gui-9297636063'
        workDir = Google_Drive_Path

        top = os.path.join(workDir, str(PSF_filename))
        toppar = os.path.join(workDir, str(toppar_filename))
        crd = os.path.join(workDir, str(CRD_filename))
        pdb = os.path.join(workDir, str(PDB_filename))

        pdb_check = os.path.exists(pdb)
        crd_check = os.path.exists(crd)
        top_check = os.path.exists(top)
        toppar_check = os.path.exists(toppar)

        if pdb_check == True and crd_check == True and top_check == True and toppar_check == True:
            print("Files loaded succesfully! ;-)")
       
        import warnings
        warnings.filterwarnings('ignore')
        import py3Dmol

        color = "rainbow"
        show_sidechains = True
        show_mainchains = True
        #show_box = True
        #box_opacity = 0.6


        def show_pdb(show_sidechains=True, show_mainchains = True, show_box = False, color = 'rainbow'):
          view = py3Dmol.view(js= 'https://3dmol.org/build/3Dmol.js')
          view.addModel(open(pdb,'r').read(),'pdb')


          if show_sidechains:
            BB = ['C','O','N']
            view.addStyle({'and':[{'resn':["GLY","PRO"],'invert':True},{'atom':BB,'invert':True}]},
                        {'NewCartoon':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
            view.addStyle({'and':[{'resn':"GLY"},{'atom':'CA'}]},
                        {'NewCartoon':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
            view.addStyle({'and':[{'resn':"PRO"},{'atom':['C','O'],'invert':True}]},
                        {'NewCartoon':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
          if show_mainchains:
            BB = ['C','O','N','CA']
            view.addStyle({'atom':BB},{'Ribbon':{'colorscheme':f"WhiteCarbon",'radius':0.3}})

          #if show_box:
            #view.addSurface(py3Dmol.SAS, {'opacity': box_opacity, 'color':'white'})


          view.zoomTo()
          return view


        # show_pdb(show_sidechains, show_mainchains, color).show()
        Jobname = equil_file[:-4]
        Jobname = os.path.join(workDir,Jobname)

        Minimization_steps = '20000000'

        Time = "0.4"
        stride_time_eq = 0.1
        Integration_timestep = "1"
        dt_eq = Integration_timestep

        Temperature = 301.3
        temperature_eq = Temperature
        Pressure = 1
        pressure_eq = Pressure

        Force_constant = 500

        Write_the_trajectory = "20"
        Write_the_trajectory_eq = Write_the_trajectory

        Write_the_log = "10"
        Write_the_log_eq = Write_the_log
        def backup_old_log(pattern, string):
          result = []
          for root, dirs, files in os.walk("./"):
            for name in files:
              if fnmatch.fnmatch(name, pattern):

                try:
                  number = int(name[-2])
                  avail = isinstance(number, int)
                  #print(name,avail)
                  if avail == True:
                    result.append(number)
                except:
                  pass

          if len(result) > 0:
            maxnumber = max(result)
          else:
            maxnumber = 0

          backup_file = "\#" + string + "." + str(maxnumber + 1) + "#"
          os.system("mv " + string + " " + backup_file)
          return backup_file

        def gen_box(system, psf, crd):
            coords = crd.positions

            min_crds = [coords[0][0], coords[0][1], coords[0][2]]
            max_crds = [coords[0][0], coords[0][1], coords[0][2]]

            box_vectors = system.getDefaultPeriodicBoxVectors()
            boxlx = box_vectors[0][0].value_in_unit(nanometers)
            boxly = box_vectors[1][1].value_in_unit(nanometers)
            boxlz = box_vectors[2][2].value_in_unit(nanometers)
            print("boxlx:", boxlx)
            print("boxly:", boxly)
            print("boxlz:", boxlz)

            box_center = [boxlx / 2, boxly / 2, boxlz / 2]  # Calculate the center of the simulation box

            for coord in coords:
                dx = coord[0] - min_crds[0]
                dy = coord[1] - min_crds[1]
                dz = coord[2] - min_crds[2]

                # Adjust the coordinates based on the image convention
                if boxlx != 0.0:
                    dx_value = dx.value_in_unit(nanometers)
                    dx -= boxlx * round((dx_value - box_center[0]) / boxlx) * nanometers

                if boxly != 0.0:
                    dy_value = dy.value_in_unit(nanometers)
                    dy -= boxly * round((dy_value - box_center[1]) / boxly) * nanometers

                if boxlz != 0.0:
                    dz_value = dz.value_in_unit(nanometers)
                    dz -= boxlz * round((dz_value - box_center[2]) / boxlz) * nanometers

                min_crds[0] = min(min_crds[0], coord[0] + dx)
                min_crds[1] = min(min_crds[1], coord[1] + dy)
                min_crds[2] = min(min_crds[2], coord[2] + dz)
                max_crds[0] = max(max_crds[0], coord[0] + dx)
                max_crds[1] = max(max_crds[1], coord[1] + dy)
                max_crds[2] = max(max_crds[2], coord[2] + dz)

            min_crds = Vec3(min_crds[0], min_crds[1], min_crds[2])
            max_crds = Vec3(max_crds[0], max_crds[1], max_crds[2])
            box_vectors = [Vec3(boxlx, 0, 0), Vec3(0, boxly, 0), Vec3(0, 0, boxlz)]

            system.setDefaultPeriodicBoxVectors(box_vectors[0], box_vectors[1], box_vectors[2])
            psf.setBox(boxlx, boxly, boxlz)
            return system

        def read_toppar(filename,path):
          extlist = ['rtf', 'prm', 'str']
          parFiles = ()
          for line in open(filename, 'r'):
            # print("filename",filename)
            if '!' in line: line = line.split('!')[0]
            # print("line",line)
            index_of_toppar = line.find("toppar")
            # Check if "toppar" is found in the string
            if index_of_toppar != -1:
            # Extract the portion of the string after "toppar"
              parfile = line[index_of_toppar:]
            else:
            # "toppar" not found, return the original string
              parfile = line

            # print("parfile ",parfile)
            parfile = parfile.strip()
            if len(parfile) != 0:
              ext = parfile.lower().split('.')[-1]
              # print("ext",ext)
              if not ext in extlist: continue
              # print("path",path)
              # print("parfile  ",parfile)
              parFiles += ( path + "/"+parfile, )
              # print(parFiles)
          params = CharmmParameterSet( *parFiles )
          return params, parFiles
        #from simtk.unit import Vec3

        def restraints(system, crd, fc, restraint_array):
            box_vectors = system.getDefaultPeriodicBoxVectors()
            boxlx = box_vectors[0][0].value_in_unit(nanometers)
            boxly = box_vectors[1][1].value_in_unit(nanometers)
            boxlz = box_vectors[2][2].value_in_unit(nanometers)

            print("boxlx:", boxlx)
            print("boxly:", boxly)
            print("boxlz:", boxlz)

            if fc > 0:
                # positional restraints for all heavy-atoms
                posresPROT = CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2;')
                posresPROT.addPerParticleParameter('k')
                posresPROT.addPerParticleParameter('x0')
                posresPROT.addPerParticleParameter('y0')
                posresPROT.addPerParticleParameter('z0')

                for atom1 in restraint_array:
                    atom1 = int(atom1)

                    position = crd.positions[atom1]
                    x0 = position[0].value_in_unit(nanometers)
                    y0 = position[1].value_in_unit(nanometers)
                    z0 = position[2].value_in_unit(nanometers)

                    x0 = x0 - boxlx * round((x0 - boxlx/2) / boxlx)
                    y0 = y0 - boxly * round((y0 - boxly/2) / boxly)
                    z0 = z0 - boxlz * round((z0 - boxlz/2) / boxlz)

                    posresPROT.addParticle(atom1, [fc, x0, y0, z0])

                system.addForce(posresPROT)

            return system


        top = os.path.join(workDir, str(PSF_filename))
        toppar = os.path.join(workDir, str(toppar_filename))
        crd = os.path.join(workDir, str(CRD_filename))
        pdb = os.path.join(workDir, str(PDB_filename))

        toppar = toppar
        coordinatefile = crd
        pdbfile = pdb
        topologyfile = top

        delta_threshold = 0.01


        time_ps = float(Time)*1000
        simulation_time = float(time_ps)*picosecond
        dt = int(dt_eq)*femtosecond
        temperature = float(temperature_eq)*kelvin
        savcrd_freq = int(Write_the_trajectory_eq)*picosecond
        print_freq  = int(Write_the_log_eq)*picosecond

        pressure  = float(pressure_eq)*bar

        restraint_fc = int(Force_constant) # kJ/mol

        nsteps  = int(simulation_time.value_in_unit(picosecond)/dt.value_in_unit(picosecond))
        nprint  = int(print_freq.value_in_unit(picosecond)/dt.value_in_unit(picosecond))
        nsavcrd = int(savcrd_freq.value_in_unit(picosecond)/dt.value_in_unit(picosecond))

        charmm_params, toppar_files = read_toppar(toppar,workDir)

        print("\n> Simulation details:\n")
        print("\tJob name = " + Jobname)
        print("\tCoordinate file = " + str(coordinatefile))
        print("\tPDB file = " + str(pdbfile))
        print("\tTopology file = " + str(topologyfile))
        print("\tForce Field files = " + str(toppar_files))

        print("\n\tSimulation_time = " + str(simulation_time))
        print("\tIntegration timestep = " + str(dt))
        print("\tTotal number of steps = " +  str(nsteps))

        print("\n\tSave coordinates each " + str(savcrd_freq))
        print("\tPrint in log file each " + str(print_freq))

        print("\n\tTemperature = " + str(temperature))
        print("\tPressure = " + str(pressure))

        print("\n> Setting the system:\n")

        print("\t- Creating system and setting parameters...")
        forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

        pdb = app.PDBFile(pdb)


        print("\t- Reading topology and structure file...")
        psf = CharmmPsfFile(topologyfile)
        crd = CharmmCrdFile(coordinatefile)

        print("\t- Creating system and setting parameters...")
        system = psf.createSystem(charmm_params, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
        psf = gen_box(system, psf, crd)
        print("\t- Applying restraints. Force Constant = " + str(Force_constant) + "kJ/mol")
        pt_system = pt.iterload(coordinatefile, topologyfile)
        pt_topology = pt_system.top
        restraint_array = pt.select_atoms('!(:H*) & !(:WAT) & !(:Na+) & !(:Cl-) & !(:Mg+) & !(:K+)', pt_topology)
        system = restraints(system, crd, restraint_fc, restraint_array)

        print("\t- Setting barostat...")
        system.addForce(MonteCarloBarostat(pressure, temperature))

        print("\t- Setting integrator...")
        integrator = LangevinIntegrator(temperature, 1/picosecond, dt)
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(crd.positions)

        print("\t- Energy minimization: " + str(Minimization_steps) + " steps")
        simulation.minimizeEnergy(tolerance=10*kilojoule/(nanometer*mole), maxIterations=int(Minimization_steps))
        print("\t-> Potential Energy = " + str(simulation.context.getState(getEnergy=True).getPotentialEnergy()))

        print("\t- Setting initial velocities...")
        simulation.context.setVelocitiesToTemperature(temperature)


        dcd_file = Jobname + ".dcd"
        log_file = Jobname + ".log"
        rst_file = Jobname + ".rst"
        prv_rst_file = Jobname + ".rst"
        pdb_file = Jobname + ".pdb"

        if os.path.exists(rst_file):
            print("\n\n> Equilibration finished (" + rst_file + " present). Nothing to do here... <")
            exit
            sys.exit()

        else:
          dcd = DCDReporter(dcd_file, nsavcrd)
          firstdcdstep = (nsteps) + nsavcrd
          dcd._dcd = DCDFile(dcd._out, simulation.topology, simulation.integrator.getStepSize(), firstdcdstep, nsavcrd) # charmm doesn't like first step to be 0

          simulation.reporters.append(dcd)
          simulation.reporters.append(StateDataReporter(stdout, nprint, step=True, speed=True, progress=True, totalSteps=nsteps, remainingTime=True, separator='\t\t'))
          simulation.reporters.append(StateDataReporter(log_file, nprint, step=True, kineticEnergy=True, potentialEnergy=True, totalEnergy=True, temperature=True, volume=True, speed=True))

          print("\n> Simulating " + str(nsteps) + " steps...")

          try:
            simulation.step(nsteps)
          except ValueError as ve:
            # Handle the specific ValueError related to NaN energy
            print(f"ValueError occurred while processing {Jobname}: {str(ve)}")
            # Continue to the next file
            continue

          except Exception as e:
            # Handle other exceptions if needed
            print(f"An error occurred while processing {Jobname}: {str(e)}")
            # Continue to the next file
            continue

          simulation.reporters.clear()

          print("\n> Writing state file (" + str(rst_file) + ")...")
          state = simulation.context.getState( getPositions=True, getVelocities=True )
          with open(rst_file, 'w') as f:
            f.write(XmlSerializer.serialize(state))

          last_frame = int(nsteps/nsavcrd)
          print("> Writing coordinate file (" + str(pdb_file) + ", frame = " + str(last_frame) + ")...")
          positions = simulation.context.getState(getPositions=True).getPositions()
          PDBFile.writeFile(simulation.topology, positions, open(pdb_file, 'w'))

          source_file_path = pdb_file # Change to your generated file's path
          destination_folder = f'../minimized_inactive/{folder_name}'
          os.makedirs(destination_folder, exist_ok=True)
          shutil.move(pdb_file, os.path.join(destination_folder, os.path.basename(pdb_file)))
          time.sleep(30)

          # Check if the file has been moved successfully
          if os.path.exists(os.path.join(destination_folder, pdb_file)):
              print("File moved successfully.")
          else:
              print("File move failed.")

          print("Simulation completed.")
        # time.sleep(60)
        def delete_charmm_files_and_folders(directory):
            for root, dirs, files in os.walk(directory, topdown=False):
                for name in dirs:
                    if "charmm" in name.lower():
                        dir_path = os.path.join(root, name)
                        print(f"Deleting folder: {dir_path}")
                        shutil.rmtree(dir_path)  # Delete the folder and its contents

                for name in files:
                    if "charmm" in name.lower():
                        file_path = os.path.join(root, name)
                        print(f"Deleting file: {file_path}")
                        os.remove(file_path)  # Delete the file

        # Specify the directory where you want to start searching and deleting
        start_directory = file_path

        # Call the function to delete files and folders with "charmm" in their names
        delete_charmm_files_and_folders(start_directory)