""" @package forcebalance.potential Electrostatic Potential fitting module

@author Moses K.J. Chung
@date 05/2025
"""
from __future__ import division

from builtins import zip
import os
import shutil

from forcebalance.nifty import col, eqcgmx, flat, floatornan, fqcgmx, invert_svd, kb, printcool, printcool_dictionary, bohr2ang, warn_press_key
import numpy as np
from forcebalance.target import Target
from forcebalance.molecule import Molecule, format_xyz_coord
from re import match, sub
import subprocess
from subprocess import PIPE
from forcebalance.finite_difference import fdwrap, f1d2p, f12d3p, in_fd
from collections import OrderedDict

from forcebalance.output import getLogger
logger = getLogger(__name__)

class Potential(Target):
    """ Subclass of Target for fitting force fields to electrostatic potential.

    Currently Tinker is supported.

    """

    def __init__(self,options,tgt_opts,forcefield):
        """Initialization."""
        
        # Initialize the SuperClass!
        super(Potential,self).__init__(options,tgt_opts,forcefield)
        
        #======================================#
        # Options that are given by the parser #
        #======================================#
        self.set_option(tgt_opts, 'potential_denom')
        self.set_option(tgt_opts, 'optimize_geometry')

        self.denoms = {}
        self.denoms['potential'] = self.potential_denom
        
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## LPW 2018-02-11: This is set to True if the target calculates
        ## a single-point property over several existing snapshots.
        self.loop_over_snapshots = False
        ## The mdata.txt file that contains the moments.
        self.pfnm = os.path.join(self.tgtdir,"pdata.txt")
        ## Read in the reference data
        self.read_reference_data()
        ## Build keyword dictionaries to pass to engine.
        engine_args = OrderedDict(list(self.OptionDict.items()) + list(options.items()))
        engine_args.pop('name', None)
        ## Create engine object.
        self.engine = self.engine_(target=self, **engine_args)

    def indicate(self):
        """ Print qualitative indicator. """
        banner = "Electrostatic Potential (Kcal/mole per unit charge)"
        headings = ["Atom", "Calc.", "Ref.", "Delta", "Term"]
        data = OrderedDict([(i+1, ["%.4f" % self.calc_potential[i], "%.4f" % self.ref_potential[i], "%.4f" % self.potential_diff[i], "%.4f" % (self.potential_diff[i]/self.potential_denom)**2]) for i in range(len(self.ref_potential))])
        self.printcool_table(data, headings, banner)
        return

    def read_reference_data(self):
        """ Read the reference data from a file. """
        ## Number of atoms
        self.na = -1
        self.ref_potential = []
        an = 0
        ln = 0
        cn = -1
        pn = -1
        for line in open(self.pfnm):
            line = line.split('#')[0] # Strip off comments
            s = line.split()
            if len(s) == 0:
                pass
            elif len(s) == 1 and self.na == -1:
                self.na = int(s[0])
                xyz = np.zeros((self.na, 3))
                cn = ln + 1
            elif ln == cn:
                pass
            elif an < self.na and len(s) == 4:
                xyz[an, :] = np.array([float(i) for i in s[1:]])
                an += 1
            elif an == self.na and s[0].lower() == 'potential':
                pn = ln + 1
            elif pn > 0 and ln >= pn and len(self.ref_potential) < self.na:
                self.ref_potential.append(float(s[0]))
            else:
                logger.info("%s\n" % line)
                logger.error("This line doesn't comply with our potential file format!\n")
                raise RuntimeError
            ln += 1
        
        if len(self.ref_potential) != self.na:
            logger.error("Number of potentials does not match number of atoms.\n")
            raise RuntimeError

        self.ref_potential = np.array(self.ref_potential)

        return

    def get(self, pvals, AGrad=False, AHess=False):
        """ Evaluate objective function. """
        Answer = {'X':0.0, 'G':np.zeros(self.FF.np), 'H':np.zeros((self.FF.np, self.FF.np))}

        def get_potential(pvals_):
            self.FF.make(pvals_)
            potential_ = self.engine.electrostatic_potential()
            return potential_

        calc_potential = get_potential(pvals)
        D = calc_potential - self.ref_potential
        self.potential_diff = D
        dV = np.zeros((self.FF.np,len(calc_potential)))
        if AGrad or AHess:
            for p in self.pgrad:
                dV[p,:], _ = f12d3p(fdwrap(get_potential, pvals, p), h = self.h, f0 = calc_potential)
        Answer['X'] = np.dot(D,D) / self.potential_denom**2
        for p in self.pgrad:
            Answer['G'][p] = 2*np.dot(D, dV[p,:]) / self.potential_denom**2
            for q in self.pgrad:
                Answer['H'][p,q] = 2*np.dot(dV[p,:], dV[q,:]) / self.potential_denom**2
        if not in_fd():
            self.calc_potential = calc_potential
            self.objective = Answer['X']
        return Answer
