# -*- coding: utf-8 -*-
"""
The testing code for Q-routing tractography method.
The q-values calculated in training code is loaded.
Probabilistic streamline tractography is performed on Q-values,
instead of fODF.
This code is used to produce the results that are submiited to 
the 2nd round of the Iron Tract Challenge by team 13.
Andaç Hamamcı, Mert Yıldız
Medical Imaging Lab., Yeditepe Univ.
andac.hamamci@yeditepe.edu.tr
https://imagingyeditepe.github.io/
"""

import numpy as np
import nibabel as nib

'''
Set input file names
'''
QFILE_NAME = 'Qtable_IronTrack_last.npz'
SEEDFILE_NAME = 'prep.inject.nii.gz'
FAFILE_NAME = 'tensor_FA.nii.gz'

'''
Set tissue priors based on FA using a treshold of 0.22
'''
faimg = nib.load(FAFILE_NAME)
fadata = faimg.get_data()
tissueprior = np.copy(np.array(fadata))

from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
classifier = ThresholdStoppingCriterion(tissueprior, .22)

'''
Set the seeds for tractography. 
64000 seeds per voxel in the seed region.
'''
seedimg = nib.load(SEEDFILE_NAME)
seed_data = seedimg.get_data()
seed_mask = (seed_data > 0)

from dipy.tracking import utils
seeds = utils.seeds_from_mask(seed_mask, density=[40, 40, 40], affine=seedimg.affine)
###############################################################################

'''
Perform probabilistic streamline tractography but on Q-values instead of fODF
'''
loaded = np.load(QFILE_NAME)
Q = loaded['Q']

#Costs were defined as -ln(fODF). Do the inverse !
pmf = np.exp(-Q)

#The sphere on which the q-values were defined
nbh=np.array([[1,0,0],[0,1,0],[1,1,0],[1,-1,0],[-1,1,0],[-1,0,0],[0,-1,0],[-1,-1,0],[-1,0,-1],[0,0,1],[0,-1,-1],[0,0,-1],[-1,-1,-1],[-1,0,1],[0,1,-1],[0,-1,1],[1,1,-1],[-1,1,1],[1,0,1],[0,1,1],[1,1,1],[1,-1,1],[1,0,-1],[-1,-1,1],[-1,1,-1],[1,-1,-1]],dtype=np.int32)

from dipy.core.sphere import Sphere
pmfsphere = Sphere(xyz=nbh)  

from dipy.direction import ProbabilisticDirectionGetter
prob_dg = ProbabilisticDirectionGetter.from_pmf(pmf,max_angle=45.,sphere=pmfsphere)

# Generate streamlines object.
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
streamlines_generator = LocalTracking(prob_dg, classifier, seeds, seedimg.affine, step_size=.5, max_cross=1)
streamlines = Streamlines(streamlines_generator)

#Save trk file
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
sft = StatefulTractogram(streamlines, seedimg, Space.RASMM)
save_trk(sft, 'hcpl_tracts_022_200m_valid.trk', bbox_valid_check=False)
