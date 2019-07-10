import pymatgen as mg
import numpy as np
from sympy import *
import settings
import json
import feature_miner_functions.FeatureMinerHelper.CalculationHelpers as ch
import feature_miner_functions.FeatureMinerHelper.ShannonHelpers as sh
import copy
import pymatgen.analysis.bond_valence as pabv;
import pymatgen.symmetry.analyzer as psa
import os
import label_miner_functions.ClassifierCreation.CrystalSystem as cs

structureDir = os.path.join(settings.ROOT_DIR, 'structure_database')
ShannonBase = os.path.join(settings.ROOT_DIR, 'Shannon_Radii')
ShannonData = json.load(open(ShannonBase+'\\ShannonRadiiDictionary.json', 'r'));


'''
collective group of all final features used in the paper
each function returns a dictionary with the respective keys as the proper feature labels
feature_names
present
[ 'Forces', 'Coordination',
       'numberDensity', 'li-ion fittability', 'maxForce',
       'oxidation flexibility', 'ShannonRadii', 'positiveoxidationpop',
       'avgIonicRadVol', 'SpaceGroup', 'volume flexibility of cell',
       'deltaShannon', 'volumeshannonflex', 'ShannonRatio',
       'oxidation flexibility Std', 'deltaShannonstd', 'deltashannonmin',
       'deltashannonmax', 'chargedensityvoldens', 'positiveoxpopdens',
       'oxstateflexmin', 'oxstateflexmax', 'deltacrystal1', 'deltacrystal2',
       'deltacrystal3', 'deltacrystal4', 'coordination std',
       'solid electronegativity', 'mass moment of inertia','charge moment of inertia',
       'negativeoxdensity', 'averageNumNN',
       'STDnumNN' ]; #33 but we only have 30 on deck
       
not present
'ShannonMeanForce', 'ShannonStdForce',
'ShannonMaxForce',
'maxCentralDistance', 'minCentralDistance'
'density', 'energy_per_atom', 'nsites', 'nelements', 'bandgap',
       'volume', 'Unit Cell Mass', 'cap_grav_Li', 'energy',
       'formationenergy_pa', 'total_magnetization', 'energysStab', 'atomMean',
       'atomStd', 'halogen', 'transition', 'chalcogen', 'metalloid', 'rare',
       'other', 'alkaline'
       
'EN_mean', 'ENStd', 'ENMax', 'ENMin', 'Crystal System', 'symmetry ops', 'ShannonForce', 'meanbv',
       'stdbv', 'meanValenceOcc', 'VegardVolume', 'Hall Number', 'ioniccount',
       'ionicitymean', 'NNdist', 'NNdiststd', 'nndistmax', 'nndistmin',
       'cubic', 'hexagonal', 'monoclinic', 'orthorhombic', 'tetragonal',
       'triclinic', 'trigonal'
'''

def AvgNumberNN(picklestruct):
    ''' calculates cumulative statistics regarding number of nearest neighbors in a 4 angstrom radius'''
    NNCount = list()
    for site in picklestruct:
        neighbors = picklestruct.get_neighbors(site, 4);
        NNCount.append(len(neighbors));
    data = {'averageNumNN': np.mean(NNCount), 'STDNumNN':np.std(NNCount)}
    return data

def AtomicNumberDensity(picklestruct): #gives a rough idea of the 'electron density'
    #this may be weird as a lot of electrons in heavy elements are tightly bound to the nucleus
    total = 0; initialVol = picklestruct.volume;
    for sites in picklestruct.sites:
        elem = sites.specie
        atmNum = sites.specie.data['Atomic no']
        total+=atmNum;
    data = {'chargedensityvoldens':total/initialVol}
    return data

#if the unit cell atoms consist of mostly positive oxidation state elements, then why would it let lithium come in?
def CellOxidationStateDensity(picklestruct): #normalize against the total number of elements...
    '''
    this may fail because of oxidation num
    :param picklestruct:
    :return:
    '''
    numElements = len(picklestruct.sites); initialVol = picklestruct.volume;
    positiveOxPop = 0; negativeOxPop = 0;
    for site in picklestruct.sites:
        elem = site.specie.value;
        if(elem not in ShannonData.keys()):
            continue;
        ShannonPoint = ShannonData[elem]; #Shannon Radii contains ONLY POSITIVE OXidation STATE MATERIALS!!!
        check = True; counter = 0;
        for i in ShannonPoint:
            if(i['oxidation_num'] < 0): #so this is a little superfluous
                check = False;
                break;
            else: counter+=1;
        if(check == True):
            positiveOxPop+=1;
        elif(counter == len(ShannonPoint)-1):
            continue; #no valence
        else:
            negativeOxPop+=1;
    data = {'positiveoxidationpop':positiveOxPop/numElements, 'positiveoxpopdens':positiveOxPop/initialVol, \
            'negativeoxdensity':negativeOxPop/initialVol}
    return data

def SpaceGroup(picklestruct):
    '''
    #numbers ranging from 1 to 230, would be nice to find a way to weight these, do not one hot encode...
    :param picklestruct:
    :return:
    '''
    data = {'SpaceGroup':picklestruct.get_space_group_info()[1]};
    return data;

def deltaShannonRadii(picklestruct):
    '''
    will not fail
    :param picklestruct:
    :return:
    '''
    initialVol = picklestruct.volume;
    deltaVolList = list();
    for site in picklestruct.sites:
        elem = site.specie.value;
        if(elem not in ShannonData.keys()):
            continue;
        ShannonPoint = ShannonData[elem];
        maxSeen = 0; minSeen = float('Inf');
        for dictionary in ShannonPoint:
            rad = dictionary['ionic_radius'];
            if(rad > maxSeen): maxSeen = rad;
            if(rad < minSeen): minSeen = rad;
        deltaVol = ch.sphereVol(maxSeen) - ch.sphereVol(minSeen);
        deltaVolList.append(deltaVol);
    deltaVolList = deltaVolList/(initialVol)**(1/3)
    if(len(deltaVolList) == 0):
        deltaVolList = [0];
    data = {'deltaShannon': np.mean(deltaVolList), 'deltaShannonstd': np.std(deltaVolList), \
            'deltashannonmin':np.min(deltaVolList), \
            'deltashannonmax':np.max(deltaVolList)}
    return data

def deltaShannonCrystalRadii(picklestruct):
    '''
    same as above function but ysing crystal radius instead of ionic radius
    :param picklestruct:
    :return:
    '''
    initialVol = picklestruct.volume;
    deltaVolList = list();
    for site in picklestruct.sites:
        elem = site.specie.value;
        if(elem not in ShannonData.keys()):
            continue;
        ShannonPoint = ShannonData[elem];
        maxSeen = 0; minSeen = float('Inf');
        for dictionary in ShannonPoint:
            rad = dictionary['crystal_radius'];
            if(rad > maxSeen): maxSeen = rad;
            if(rad < minSeen): minSeen = rad;
        deltaVol = ch.sphereVol(maxSeen) - ch.sphereVol(minSeen);
        deltaVolList.append(deltaVol);
    #scale devltaVolList
    deltaVolList = deltaVolList/(initialVol)**(1/3)
    if(len(deltaVolList) == 0):
        deltaVolList = [0];
    data = {'deltacrystal1': np.mean(deltaVolList),
               'deltacrystal2': np.std(deltaVolList), 'deltacrystal3': np.min(deltaVolList),\
            'deltacrystal4': np.max(deltaVolList)}
    return data

def IonRadVsLattice(picklestruct):
    #attempts to account for how rectangular vs cubic the cell is by taking the smallest
    #unit cell length, and seeing how much larger or smaller than it is compared to the lithium atom volume
    '''
    :param picklestruct:
    :param ion:
    :return:
    '''
    ion = 'Li';
    initialvol = picklestruct.volume
    unitcelllengths = picklestruct.lattice.abc #we sould not scal ehere as we are comparing lattice to lithium ion radius, which is fixed
    minlength = np.min(unitcelllengths);
    Lirad = mg.Element(ion).average_ionic_radius
    diff = minlength/Lirad;
    data = {'li-ion fittability': diff/initialvol**(1/3)};
    return data;

#get the nearest neighbors to every element in the lattice, calculate average distances

#Li always has a +1 oxidation state in an atom, which means upon lithiation, materials should reduce
#their oxidation state... get a sense of how willing the constituents are to decrease their oxidation state
def oxidationStateFlexibility(picklestruct):
    #if we normalize against the cell volume, it gives a sense of what the oxidation difference can do per unit volume
    #at the same time, the unnormalized version of this data is already a top feature
    initialvol = picklestruct.volume
    diffs = list();
    for site in picklestruct.sites:
        minox = site.specie.min_oxidation_state
        maxox = site.specie.max_oxidation_state;
        diff = maxox-minox; diffs.append(diff);
    diffs = [i/len(picklestruct.sites) for i in diffs];
    data = {'oxidation flexibility': np.mean(diffs), 'oxidation flexibility Std': np.std(diffs), \
            'oxstateflexmin': np.min(diffs), 'oxstateflexmax': np.max(diffs)};
    return data;

def oxidationStateVolumeFlexibility(picklestruct):
    #how much does atomic volume change when charge state changes
    #we can use physical volumes as we normalize to teh volume of the unit cell
    volume = picklestruct.volume;
    VolChange = 0; volDiff = 0;
    for site in picklestruct.sites:
        elem = site.specie.value;
        if(elem not in ShannonData.keys()):
            continue;
        ShannonPoint = ShannonData[elem]
        maxrad = 0; minrad = float('inf');
        for i in ShannonPoint:
            rad = i['ionic_radius'];
            if(rad > maxrad):
                maxrad = rad;
            if(rad<minrad):
                minrad = rad;
        volDiff = ch.sphereVol(maxrad)-ch.sphereVol(minrad);
    VolChange += volDiff;
    data = {'volume flexibility of cell': VolChange/volume}
    return data; #does this need a normalization to the unit cell?

def VolumeByAvgIonicRadius(picklestruct):
    '''
    volume calculated using AVERAGE ionic radii.
    :param picklestruct:
    :return:
    '''
    volume = picklestruct.volume;
    Vtot = 0;
    for site in picklestruct.sites:
        elem = site.specie;
        avgionicrad = elem.average_ionic_radius
        Vtot += ch.sphereVol(avgionicrad);
    data = {'avgIonicRadVol': (volume - Vtot)/volume}; #should this be normalized
    return data

#SHANNON RADII MINING...all features should be normalized so they can be comparable between compounds
def VolumeByShannonRadii(picklestruct):
    '''
    more precies than previous cuz it tries to use ionic radius using the shannon number
    :param picklestruct:
    :return:
    '''
    volume = picklestruct.volume;
    Vtot = 0;
    for site in picklestruct.sites:
        elem = site.specie.value;
        if not hasattr(elem, 'coordination_no'):
            continue;
        coordin_no = site.coordination_no;
        if(elem not in ShannonData.keys()):
            continue;
        ShannonPoint = ShannonData[elem];
        v = 0; rad = 0;
        for i in ShannonPoint: #only positive shannon point data in our data set.
            if(i['coordination_no'] == coordin_no):
                rad= i['ionic_radius'];
                break; #we've found the correct ionic radius, so stop searching Shannon points

        ## =================== Potential source of inaccuracy right here ==================================##
        if(rad == 0): #if rad is still zero, that means we didn't find the shannon point, so just use the avg ionic radius
            #as a suitable proxy for the average ionic radius
            rad = site.specie.average_ionic_radius;
        v = ch.sphereVol(rad);
        Vtot += v;
    data = {'ShannonRadii': (volume - Vtot)/volume};
    return data;

def VolumeFlexibilityByShannonRadii(picklestruct):
    #change in volume when anion charge state is modified
    '''
    :param picklestruct:
    :return:
    '''
    startVol = VolumeByShannonRadii(picklestruct);
    startVol = startVol['ShannonRadii'];
    TotDeltaVol = 0;
    for site in picklestruct.sites:
        elem = site.specie.value;
        if not hasattr(elem, 'coordination_no'):
            continue;
        coordin_no = site.coordination_no;
        if(elem not in ShannonData.keys()):
            continue;
        ShannonPoint = ShannonData[elem];
        deltaVol = 0;
        if(sh.isAnion(ShannonPoint) == True):
            originalRad = sh.getIonicRadiusWithCoordination(ShannonPoint, coordin_no);
            originalOx = sh.getOxNumbGivenCoordination(ShannonPoint, coordin_no);
            newRad = sh.getIonicRadGivenOx(ShannonPoint, originalOx - 1);
            if(newRad == None):
                newRad = originalRad
            deltaVol = ch.sphereVol(originalRad) - ch.sphereVol(newRad);
        TotDeltaVol+=deltaVol
    data = {'volumeshannonflex': TotDeltaVol/startVol};
    return data;

def ShannonRatio(picklestruct):
    '''
    Z/IR ratio
    :param picklestruct:
    :return:
    '''
    rads = list();
    for site in picklestruct.sites:
        elem = site.specie.value;
        if not hasattr(elem, 'coordination_no'):
            continue;
        coordin_no = site.coordination_no;
        if(elem not in ShannonData.keys()):
            continue;
        ShannonPoint = ShannonData[elem];
        rad = 0; charge = np.mean(site.specie.common_oxidation_states)
        for i in ShannonPoint:
            if('coordination_no' not in i.keys()):
                continue;
            if(i['coordination_no'] == coordin_no):
                rad= i['Z/IR'];
                rads.append(rad);
                break; #we've found the correct ionic radius, so stop searching Shannon points
        if(rad == 0): #if rad is still zero, that means we didn't find the shannon point, so just use the avg ionic radius
            #as a suitable proxy for the average ionic radius
            rads.append(charge/site.specie.average_ionic_radius);
    data = {'ShannonRatio': np.mean(rads)}
    return data

def ElectronegativitySolid(picklestruct): #taken from Davies and Butler using a geometric mean
    chi_total = 1; root = 0;
    for site in picklestruct.sites:
        elemElectroneg = site.specie.X;
        chi_total = chi_total*elemElectroneg;
        root+=1;
    data = {'solid electronegativity': elemElectroneg**(1/root)};
    return data

def Forces(sitesDat):#input is the sites datastructure from the structures_asdict from structures_query
    '''
    calculates an array of forces given the forces from the relaxation...sort of like a stability measures
    '''
    Fmax = 0; Ftot =0;
    for i in range(len(sitesDat)):
        elem = sitesDat[i];
        atom = mg.Element(elem['label']);
        forces = elem['properties']['forces']
        F = 0; #we need to calculate magnitude
        for j in range(3):
            F += forces[j] ** 2
        F = F ** .5;
        if (F > Fmax):
            Fmax = F;
        Ftot+=F;
    data = {'Forces': Ftot/len(sitesDat), 'maxForce':Fmax};
    return data;

def coordinationNumber(sitesDat): #this data point is a little bit too discrete
    '''
    statistics about number of nearest neighbors
    :param sitesDat:
    :return:
    '''
    totcoordinNum = 0; coordinNumList = list();
    for i in range(len(sitesDat)):
        elem = sitesDat[i];
        if('coordination_no' not in elem['properties'].keys()):
            continue;
        coordinNum = elem['properties']['coordination_no'];
        coordinNumList.append(coordinNum)
        totcoordinNum += coordinNum;
    data = {'Coordination':np.mean(coordinNumList),  'coordination std': np.std(coordinNumList)};
    return data;

def numberDensity(structure): #number density of the entire lattice of the unlithiated compound
    volume = structure['lattice']['volume'];
    data = {'numberDensity': len(structure['sites'])/volume};
    return data;



def avgDistancefromCoM(picklestruct):
    '''
    remember fractional coords
    :param picklestruct:
    :return: average distance of sites in structure from the COM
    '''

    def CenterofMass(picklestruct): #function needs to be imbedded
        '''
        centerofmass in terms of fractional coordinates
        :param picklestruct:
        :return:
        '''
        numerator = np.array([0.0, 0.0, 0.0])
        denominator = list();
        for sites in picklestruct.sites:
            mass = sites.specie.data['Atomic mass']
            cellPosition = sites.frac_coords;  # normalized
            numerator += cellPosition * mass
            denominator.append(mass);
        return numerator / (np.mean(denominator));

    RCM = CenterofMass(picklestruct)
    totalR = list();
    for i in range(len(picklestruct.sites)):
        latticesite = picklestruct.sites[i];
        # extract element and calculate mass
        #atom = mg.Element(latticesite.specie.value);
        R = latticesite.frac_coords; #this has to be in fractional coordinates
        dr = 0;
        for j in range(3):
            dx = R[j] - RCM[j];
            dr += dx **2;
        dr = dr**0.5;
        totalR.append(dr);
    #no normalization since we're in fractional coords anyways
    data = {'avgCentralDistance':np.mean(totalR), 'avgCentralDistance Std': np.std(totalR)};
    return data;

def ChargeMomentOfInertia(picklestruct):
    '''
    moment of inertia seems sort of extraneous
    :param picklestruct:
    :return:
    '''
    def CenterofMass(picklestruct): #function needs to be imbedded
        '''
        centerofmass in terms of fractional coordinates
        :param picklestruct:
        :return:
        '''
        numerator = np.array([0.0, 0.0, 0.0])
        denominator = list();
        for sites in picklestruct.sites:
            mass = sites.specie.data['Atomic mass']
            cellPosition = sites.frac_coords;  # normalized
            numerator += cellPosition * mass
            denominator.append(mass);
        return numerator / (np.mean(denominator));
    SPA = psa.SpacegroupAnalyzer(picklestruct);
    picklestruct = SPA.get_conventional_standard_structure();
    Rcm = CenterofMass(picklestruct); #fractional coords
    I = 0;
    for site in picklestruct.sites:
        elemCharge = site.specie.max_oxidation_state;
        coords = site._fcoords;
        dist = coords-Rcm;
        I += elemCharge*np.dot(dist,dist);
    data = {'charge moment of inertia': I };
    return data;

def MassMomentOfInertia(picklestruct):
    def CenterofMass(picklestruct): #function needs to be imbedded
        '''
        centerofmass in terms of fractional coordinates
        :param picklestruct:
        :return:
        '''
        numerator = np.array([0.0, 0.0, 0.0])
        denominator = list();
        for sites in picklestruct.sites:
            mass = sites.specie.data['Atomic mass']
            cellPosition = sites.frac_coords;  # normalized
            numerator += cellPosition * mass
            denominator.append(mass);
        return numerator / (np.mean(denominator));
    SPA = psa.SpacegroupAnalyzer(picklestruct);
    picklestruct = SPA.get_conventional_standard_structure();
    Rcm = CenterofMass(picklestruct); #fractional coords
    I = 0;
    for site in picklestruct.sites:
        Mass = site.specie.data['Atomic mass'];
        coords = site._fcoords;
        dist = coords-Rcm;
        I += Mass*np.dot(dist,dist);
    data = {'mass moment of inertia': I};
    return data;

#our attempt to do a simple first principles approximation of the vegard coefficients
def VegardCoefficientsApprox(picklestruct):
    '''
    probably want to cite a reference for vegard
    :param picklestruct:
    :return:
    '''
    latticeParams = picklestruct.lattice.abc;
    a = latticeParams[0]; b = latticeParams[1]; c = latticeParams[2]
    volumeinit = picklestruct.volume;
    LiLatt = [2.9, 2.9, 2.9] #units of angstroms
    #we'll lithiate to 10% of the composition
    predx = LiLatt[0]*0.1 + 0.9*latticeParams[0];
    predy = LiLatt[1]*0.1 + 0.9*latticeParams[1];
    predz = LiLatt[2]*0.1 + 0.9*latticeParams[2];
    #predx, predy, predz by themselves aren't that meaningful
    data = {'VegardVolume': predx*predy*predz/volumeinit}
    return data

def getCellSymmetryOps(picklestruct):
    symmetry = psa.SpacegroupAnalyzer(picklestruct);
    numSGOps = len(symmetry.get_symmetry_operations());
    data = {'symmetry ops':numSGOps}
    return data;

def ionicityOfLattice(picklestruct):
    # AS a general rule, ionic bonds are stronger than covalent bonds
    # perhaps materials with more ionic bonds tend to resist lithium intercalation more
    r = 4;  # four angstroms is a very good motivation for bond length (though this will overestimate small bonds)
    # we have to scale r as we play with fractional coordinates
    '''
        we will compare nearest neighbors to see how much ionic contrast there is between elements...
    '''
    #initialvol = picklestruct.volume;
    originalcell = copy.copy(picklestruct);
    ionicCount = list();
    ## iterate through the original cell
    for site in originalcell.sites:
        elementElectroneg = site.specie.X;
        sitecoord = site.coords;
        subcount = list();
        subionicity = list();
        neighborsArray = originalcell.get_neighbors(site, r)
        ionic = 0;
        for siteneighbor in neighborsArray:
            neighborElectroneg = siteneighbor[0].specie.X;
            neighborcoord = siteneighbor[0].frac_coords;  # we should use fractional coordinates here...
            dist = ch.getDist(neighborcoord, sitecoord);
            if (abs(elementElectroneg - neighborElectroneg) > 2):
                ionic += 1;
            subionicity.append(abs(elementElectroneg - neighborElectroneg)); #seems like it would be natural to normalize
        ionicCount.append(ionic)
    data = {'ioniccount': np.mean(ionicCount), 'ionicitymean': np.mean(subionicity)}
    return data

def avgDistanceOfNearestNeighbors(
        picklestruct):  # WE NEED TO CALCULATE WITH A SUPERCELL, and CANNOT USE FRACCOORDS==SLOW
    # also should take the ratio against the Li radius
    # we may need the pymatgen spacegroup analyzer
    # make a copy or else when we make supercell, we overwrite originalcell too
    initialvol = picklestruct.volume;
    originalcell = copy.copy(
        picklestruct);  # we need to displace the origiinal cell so it is in the center of the supercell!!!!
    radius = 4  # 4 angstroms is well motivated bond length, no scaling needed because we use physical dist to get nearest neighbors
    avgNNdist = list();
    #print(len(originalcell))
    for site in originalcell:  # could be on the order of 100
        distances = list();
        neighborsArray = originalcell.get_neighbors(site, radius);  # radius should be nearest neighbors
        # print(len(neighborsArray))
        sitecoord = site.coords;
        for siteneighbor in neighborsArray:  # order of 10, let's say
            neighborcoord = siteneighbor[0].coords;  # we should use fractional coordinates here...
            dist = ch.getDist(neighborcoord, sitecoord);
            distances.append(dist);
        avgNNdist.append(np.mean(distances))

    avgNNdist = [i / initialvol ** (1 / 3) for i in avgNNdist];
    data = {'NNdist': np.mean(avgNNdist),
            'NNdiststd': np.std(avgNNdist), 'nndistmax': np.max(avgNNdist), 'nndistmin': np.min(avgNNdist)};
    return data;
# feature is already something we can find in the matdata miner
# def getCrystalSystem(picklestruct):
#     symmetry = psa.SpacegroupAnalyzer(picklestruct);
#     numeric = cs.CrystalSysClassFeat(symmetry.get_crystal_system());
#     data = {'Crystal System': numeric}
#     return data;
        # def Hall_Number(picklestruct): #return hall_number, which is just another way of listing a spacegroup number
#     symmetryDat = psa.SpacegroupAnalyzer(picklestruct);
#     data = {'Hall Number': symmetryDat.get_symmetry_dataset()['hall_number']}
#     return data;