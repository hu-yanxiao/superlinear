"""Strict cross-backend fixtures; set TWO_CUTOFF_FIXTURES to the audited bundle."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
from ase import Atoms
from pysuperlinear import SuperlinearCalculator

FIXTURES = os.environ.get('TWO_CUTOFF_FIXTURES')
pytestmark = pytest.mark.skipif(not FIXTURES, reason='explicit audited two-cutoff fixtures required')
MODES = ['static', 'effect', 'adaptive_full', 'adaptive_shared-o', 'adaptive_shared-n']


def configurations(path):
    output = []
    for block in path.read_text().split('BEGIN_CFG')[1:]:
        lines = block.splitlines()
        size = int(lines[lines.index(' Size')+1])
        cell_index = lines.index(' Supercell')
        cell = np.array([list(map(float,row.split())) for row in lines[cell_index+1:cell_index+4]])
        atom_index = next(i for i,row in enumerate(lines) if row.startswith(' AtomData:'))
        atoms = np.array([list(map(float,row.split())) for row in lines[atom_index+1:atom_index+1+size]])
        output.append(Atoms(numbers=np.where(atoms[:,1]==0,1,8),positions=atoms[:,2:5],cell=cell,pbc=True))
    return output


@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('index',range(7))
def test_two_cutoff_efs_and_metadata(mode,index):
    root = Path(FIXTURES)
    expected = json.loads((root/(mode+'.efs.json')).read_text())['records'][index]
    atoms = configurations(root/'geometries.cfg')[index]
    calculator = SuperlinearCalculator(root/(mode+'.mtp'),species=['H','O'])
    atoms.calc = calculator
    np.testing.assert_allclose(atoms.get_potential_energy(),expected['energy'],rtol=2e-11,atol=1e-12)
    np.testing.assert_allclose(atoms.get_forces(),expected['forces'],rtol=2e-10,atol=1e-12)
    np.testing.assert_allclose(calculator.results['superlinear_virial'],expected['plus_stress'],rtol=2e-10,atol=1e-12)
    np.testing.assert_allclose(atoms.get_stress(voigt=False),-np.asarray(expected['plus_stress'])/atoms.get_volume(),rtol=2e-10,atol=1e-14)
    assert calculator.metadata['radial_cutoff']==6.5
    assert calculator.metadata['angular_cutoff']==5.
