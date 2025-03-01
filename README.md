# Nanoscale-Surfactant-Transport
# Published work: Nanoscale Surfactant Transport: Bridging Molecular and Continuum Models, MR Rahman, JP Ewen, L Shen, DM Heyes, D Dini, ER Smith, Journal of Fluid Mechanics, 2025


Key descriptions of the system

MARTINI water model:

The MARTINI model, in general, maps four molecules and the associated hydrogens to one coarse-grain bead. The interactions between the CG particles in the system take place by means of Lennard-Jones interactions, whereas, the charged groups share Coulombic interactions. 
The water beads in the regular MARTINI water model, however, do not have any charge and therefore, do not respond to electrostatic fields. 
In majority of cases involving bulk water, this draw back is overcome by considering a uniform dielectric constant, but suffers when an interface or any polar solvent is involved. 

Polarizable water model: 

The MARTINI polarizable water model (Yesylevskyy et al., 2010) is made of beads that consists three particles instead of a single particle as in the regular model. The central particle (W) is charge-neutral, whereas, the remaining two particles are positively (WP) and negatively charged (WM). W interacts with other particles in the system through Lennard-Jones potential. Such LJ interaction is absent in case of WP and WM, but they interact by means of a pure Coulombic function. Within the same bead, WP and WM do not interact with each other. 
The total mass of one CG bead in the polarizable water model is 72 amu which corresponds to the mass of four real water molecules. 

CG description of SDS:

The SDS is described by (Anogiannakis et al. 2019, Weiand et al. 2023) 5 CG beads, 3 apolar $C_1$ beads each mapping four methylene units, the terminal negatively charged hydrophilic sulphate head group is mapped as a single $Q_a$ bead, and the sodium cation mapped as a $Q_d$ bead. 


sds --> Head group: type 1, Tail: type 2, Sodium: type 5,
water --> types 3 and 4.
