"""
Single pipeline: extends fc_group/functional_group_dataset.csv in place with
new functional-group molecules (halogen family, O/S/N parallels, position
isomers), sourcing their property columns as reliably as possible, and
overwrites the file with the combined result.

Sources, by column:
    mw, tpsa, hbd, hba, avg_carbon_oxidation_state
        -> computed offline and exactly from SMILES via RDKit.
    boiling_point_c, water_solubility
        -> fetched from PubChem PUG View's "Experimental Properties" section
           (best-effort; not every compound has data there -- left NaN and
           flagged rather than guessed).
    pka, pkah
        -> no reliable free structured API exists for these. Filled in from
           standard literature pKa values (MANUAL_PKA below), same as how
           the original dataset's pKa/pKaH columns were hand-curated -- one
           representative value per functional-group class rather than a
           per-molecule lookup.

Only the molecule *identity* (name, SMILES, functional group, chain length)
is authored directly below -- that defines which molecule it is, not a
property to look up.
"""
import re
import time

import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

ORIGINAL_CSV = 'fc_group/functional_group_dataset.csv'
OUTPUT_CSV = 'fc_group/functional_group_dataset.csv'
COLUMNS = [
    'iupac_name', 'common_name', 'formula', 'carbon_count', 'functional_group',
    'functional_group_structure', 'mw', 'pka', 'pkah', 'tpsa',
    'avg_carbon_oxidation_state', 'hbd', 'hba', 'boiling_point_c', 'water_solubility',
]

PUBCHEM_CID_URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON'
PUBCHEM_VIEW_URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/?heading={heading}'
REQUEST_DELAY_S = 0.25
ELECTRONEGATIVE = {'O', 'N', 'S', 'F', 'Cl', 'Br', 'I'}

# One representative (pka, pkah) per new functional-group class, from standard
# literature pKa tables (e.g. Bordwell/Evans-style aqueous pKa compilations).
# NaN = no confident standard value to cite; left for future manual lookup
# rather than guessed.
MANUAL_PKA = {
    # alpha-C-H pKa carried over from the existing "alkyl halide" (Cl) convention;
    # halogen identity doesn't shift it enough to justify fabricated precision.
    'alkyl fluoride': (48, 'N/A (no conventional aqueous pKaH)'),
    'alkyl bromide':  (48, 'N/A (no conventional aqueous pKaH)'),
    'alkyl iodide':   (48, 'N/A (no conventional aqueous pKaH)'),
    # dialkyl ether/sulfide/sulfoxide conjugate-acid pKaH are well-documented;
    # their own pKa (no acidic proton of note) is left NaN.
    'ether':          (np.nan, -3.5),
    'thioether':      (np.nan, -6.8),
    'sulfoxide':      (np.nan, -1.8),
    'sulfone':        (np.nan, np.nan),  # no confident standard value for either
    # simple aldimine conjugate acid (iminium) pKaH ~7; imine N-H pKa not
    # a standard tabulated value.
    'imine':          (np.nan, 7.2),
    # nitroalkane alpha-C-H is a classic, well-documented acidic proton;
    # nitro basicity is negligible (no meaningful pKaH).
    'nitro':          (8.9, 'N/A (negligible basicity)'),
    # position isomers (propan-2-ol, propan-2-amine, propane-2-thiol, ...) are
    # merged into the primary alcohol/amine/thiol classes -- C1 vs C2 diff-vectors
    # measured at ~0.98-0.99 cosine similarity, so position doesn't distinguish
    # them. pKa/pKaH still differ slightly per exact molecule (isopropanol,
    # isopropylamine, isopropanethiol literature values), independent of the
    # shared class label.
    'alcohol':        (17.1, -2.35),
    'amine':          (35, 10.63),
    'thiol':          (10.9, -6.8),
}

SOLUBILITY_KEYWORDS = [
    ('miscible', 'miscible'),
    ('insoluble', 'very low'),
    ('very slightly soluble', 'very low'),
    ('slightly soluble', 'low'),
    ('sparingly soluble', 'low'),
    ('freely soluble', 'high'),
    ('very soluble', 'high'),
    ('soluble', 'moderate'),
]

# ============================
# New molecules (identity only)
# ============================

NEW_MOLECULES = [
    # --- Halogen family: fluoride, bromide, iodide (parallel to existing chloride rows) ---
    *[
        dict(iupac_name=f'1-Fluoro{stem}', common_name=f'n-{alkyl} fluoride', smiles=smiles,
             carbon_count=c, functional_group='alkyl fluoride', functional_group_structure='-F')
        for stem, alkyl, smiles, c in [
            ('propane', 'propyl', 'CCCF', 3), ('butane', 'butyl', 'CCCCF', 4),
            ('pentane', 'amyl', 'CCCCCF', 5), ('hexane', 'hexyl', 'CCCCCCF', 6),
        ]
    ],
    *[
        dict(iupac_name=f'1-Bromo{stem}', common_name=f'n-{alkyl} bromide', smiles=smiles,
             carbon_count=c, functional_group='alkyl bromide', functional_group_structure='-Br')
        for stem, alkyl, smiles, c in [
            ('propane', 'propyl', 'CCCBr', 3), ('butane', 'butyl', 'CCCCBr', 4),
            ('pentane', 'amyl', 'CCCCCBr', 5), ('hexane', 'hexyl', 'CCCCCCBr', 6),
        ]
    ],
    *[
        dict(iupac_name=f'1-Iodo{stem}', common_name=f'n-{alkyl} iodide', smiles=smiles,
             carbon_count=c, functional_group='alkyl iodide', functional_group_structure='-I')
        for stem, alkyl, smiles, c in [
            ('propane', 'propyl', 'CCCI', 3), ('butane', 'butyl', 'CCCCI', 4),
            ('pentane', 'amyl', 'CCCCCI', 5), ('hexane', 'hexyl', 'CCCCCCI', 6),
        ]
    ],

    # --- O/S/N parallels ---
    dict(iupac_name='Methoxyethane', common_name='ethyl methyl ether', smiles='CCOC',
         carbon_count=3, functional_group='ether', functional_group_structure='-O-'),
    dict(iupac_name='1-Methoxypropane', common_name='methyl propyl ether', smiles='CCCOC',
         carbon_count=4, functional_group='ether', functional_group_structure='-O-'),
    dict(iupac_name='1-Methoxybutane', common_name='methyl butyl ether', smiles='CCCCOC',
         carbon_count=5, functional_group='ether', functional_group_structure='-O-'),
    dict(iupac_name='1-Methoxypentane', common_name='methyl pentyl ether', smiles='CCCCCOC',
         carbon_count=6, functional_group='ether', functional_group_structure='-O-'),

    dict(iupac_name='(Methylsulfanyl)ethane', common_name='ethyl methyl sulfide', smiles='CCSC',
         carbon_count=3, functional_group='thioether', functional_group_structure='-S-'),
    dict(iupac_name='1-(Methylsulfanyl)propane', common_name='methyl propyl sulfide', smiles='CCCSC',
         carbon_count=4, functional_group='thioether', functional_group_structure='-S-'),
    dict(iupac_name='1-(Methylsulfanyl)butane', common_name='methyl butyl sulfide', smiles='CCCCSC',
         carbon_count=5, functional_group='thioether', functional_group_structure='-S-'),
    dict(iupac_name='1-(Methylsulfanyl)pentane', common_name='methyl pentyl sulfide', smiles='CCCCCSC',
         carbon_count=6, functional_group='thioether', functional_group_structure='-S-'),

    dict(iupac_name='Methyl ethyl sulfoxide', common_name='methyl ethyl sulfoxide', smiles='CS(=O)CC',
         carbon_count=3, functional_group='sulfoxide', functional_group_structure='>S=O'),
    dict(iupac_name='Methyl propyl sulfoxide', common_name='methyl propyl sulfoxide', smiles='CS(=O)CCC',
         carbon_count=4, functional_group='sulfoxide', functional_group_structure='>S=O'),
    dict(iupac_name='Methyl butyl sulfoxide', common_name='methyl butyl sulfoxide', smiles='CS(=O)CCCC',
         carbon_count=5, functional_group='sulfoxide', functional_group_structure='>S=O'),
    dict(iupac_name='Methyl pentyl sulfoxide', common_name='methyl pentyl sulfoxide', smiles='CS(=O)CCCCC',
         carbon_count=6, functional_group='sulfoxide', functional_group_structure='>S=O'),

    dict(iupac_name='Methyl ethyl sulfone', common_name='methyl ethyl sulfone', smiles='CS(=O)(=O)CC',
         carbon_count=3, functional_group='sulfone', functional_group_structure='>SO2'),
    dict(iupac_name='Methyl propyl sulfone', common_name='methyl propyl sulfone', smiles='CS(=O)(=O)CCC',
         carbon_count=4, functional_group='sulfone', functional_group_structure='>SO2'),
    dict(iupac_name='Methyl butyl sulfone', common_name='methyl butyl sulfone', smiles='CS(=O)(=O)CCCC',
         carbon_count=5, functional_group='sulfone', functional_group_structure='>SO2'),
    dict(iupac_name='Methyl pentyl sulfone', common_name='methyl pentyl sulfone', smiles='CS(=O)(=O)CCCCC',
         carbon_count=6, functional_group='sulfone', functional_group_structure='>SO2'),

    dict(iupac_name='Propan-1-imine', common_name='propionaldehyde imine', smiles='CCC=N',
         carbon_count=3, functional_group='imine', functional_group_structure='-CH=NH'),
    dict(iupac_name='Butan-1-imine', common_name='butyraldehyde imine', smiles='CCCC=N',
         carbon_count=4, functional_group='imine', functional_group_structure='-CH=NH'),
    dict(iupac_name='Pentan-1-imine', common_name='valeraldehyde imine', smiles='CCCCC=N',
         carbon_count=5, functional_group='imine', functional_group_structure='-CH=NH'),
    dict(iupac_name='Hexan-1-imine', common_name='caproaldehyde imine', smiles='CCCCCC=N',
         carbon_count=6, functional_group='imine', functional_group_structure='-CH=NH'),

    dict(iupac_name='1-Nitropropane', common_name='1-nitropropane', smiles='CCC[N+](=O)[O-]',
         carbon_count=3, functional_group='nitro', functional_group_structure='-NO2'),
    dict(iupac_name='1-Nitrobutane', common_name='1-nitrobutane', smiles='CCCC[N+](=O)[O-]',
         carbon_count=4, functional_group='nitro', functional_group_structure='-NO2'),
    dict(iupac_name='1-Nitropentane', common_name='1-nitropentane', smiles='CCCCC[N+](=O)[O-]',
         carbon_count=5, functional_group='nitro', functional_group_structure='-NO2'),
    dict(iupac_name='1-Nitrohexane', common_name='1-nitrohexane', smiles='CCCCCC[N+](=O)[O-]',
         carbon_count=6, functional_group='nitro', functional_group_structure='-NO2'),

    # --- Position isomers: group at C2 instead of C1. Merged into the primary
    # alcohol/amine/thiol classes below -- C1 vs C2 diff-vectors measured at
    # ~0.98-0.99 cosine similarity, so position doesn't meaningfully distinguish them. ---
    dict(iupac_name='Propan-2-ol', common_name='isopropanol', smiles='CC(O)C',
         carbon_count=3, functional_group='alcohol', functional_group_structure='-OH'),
    dict(iupac_name='Butan-2-ol', common_name='sec-butanol', smiles='CC(O)CC',
         carbon_count=4, functional_group='alcohol', functional_group_structure='-OH'),
    dict(iupac_name='Pentan-2-ol', common_name='sec-amyl alcohol', smiles='CC(O)CCC',
         carbon_count=5, functional_group='alcohol', functional_group_structure='-OH'),
    dict(iupac_name='Hexan-2-ol', common_name='sec-hexyl alcohol', smiles='CC(O)CCCC',
         carbon_count=6, functional_group='alcohol', functional_group_structure='-OH'),

    dict(iupac_name='Propan-2-amine', common_name='isopropylamine', smiles='CC(N)C',
         carbon_count=3, functional_group='amine', functional_group_structure='-NH2'),
    dict(iupac_name='Butan-2-amine', common_name='sec-butylamine', smiles='CC(N)CC',
         carbon_count=4, functional_group='amine', functional_group_structure='-NH2'),
    dict(iupac_name='Pentan-2-amine', common_name='sec-amylamine', smiles='CC(N)CCC',
         carbon_count=5, functional_group='amine', functional_group_structure='-NH2'),
    dict(iupac_name='Hexan-2-amine', common_name='sec-hexylamine', smiles='CC(N)CCCC',
         carbon_count=6, functional_group='amine', functional_group_structure='-NH2'),

    dict(iupac_name='Propane-2-thiol', common_name='isopropyl mercaptan', smiles='CC(S)C',
         carbon_count=3, functional_group='thiol', functional_group_structure='-SH'),
    dict(iupac_name='Butane-2-thiol', common_name='sec-butyl mercaptan', smiles='CC(S)CC',
         carbon_count=4, functional_group='thiol', functional_group_structure='-SH'),
    dict(iupac_name='Pentane-2-thiol', common_name='sec-amyl mercaptan', smiles='CC(S)CCC',
         carbon_count=5, functional_group='thiol', functional_group_structure='-SH'),
    dict(iupac_name='Hexane-2-thiol', common_name='sec-hexyl mercaptan', smiles='CC(S)CCCC',
         carbon_count=6, functional_group='thiol', functional_group_structure='-SH'),
]


# ============================
# RDKit: computed properties
# ============================

def compute_rdkit_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return {
        'mw': round(Descriptors.MolWt(mol), 3),
        'tpsa': round(Descriptors.TPSA(mol), 2),
        'hbd': Lipinski.NumHDonors(mol),
        'hba': Lipinski.NumHAcceptors(mol),
    }


def compute_avg_carbon_oxidation_state(smiles):
    """Standard organic oxidation-state rule: a bond to H contributes -1 (per
    bond order) to the carbon; a bond to a more electronegative atom
    contributes +1 (per bond order); a bond to another C contributes 0.
    Validated against this dataset's existing values, e.g. propane -> -2.667,
    propanenitrile's nitrile carbon (C#N, triple bond) -> +3.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    carbons = [a for a in mol.GetAtoms() if a.GetSymbol() == 'C']
    if not carbons:
        raise ValueError(f"No carbon atoms in {smiles}")
    states = []
    for c in carbons:
        state = 0
        for bond in c.GetBonds():
            other = bond.GetOtherAtom(c)
            order = int(round(bond.GetBondTypeAsDouble()))
            if other.GetSymbol() == 'H':
                state -= order
            elif other.GetSymbol() == 'C':
                state += 0
            elif other.GetSymbol() in ELECTRONEGATIVE:
                state += order
            else:
                raise ValueError(f"Unhandled neighbor element '{other.GetSymbol()}' bonded to carbon in {smiles}")
        states.append(state)
    return sum(states) / len(states)


# ============================
# PubChem: experimental properties (best-effort)
# ============================

def _pubchem_get(url):
    resp = requests.get(url, timeout=20)
    time.sleep(REQUEST_DELAY_S)
    if resp.status_code != 200:
        return None
    return resp.json()


def resolve_cid(name):
    data = _pubchem_get(PUBCHEM_CID_URL.format(name=requests.utils.quote(name)))
    if not data:
        return None
    try:
        return data['IdentifierList']['CID'][0]
    except (KeyError, IndexError):
        return None


def _iter_info_entries(section_json, heading):
    def walk(node):
        if isinstance(node, dict):
            if node.get('TOCHeading') == heading:
                for info in node.get('Information', []):
                    yield info
            for child in node.get('Section', []):
                yield from walk(child)
        elif isinstance(node, list):
            for item in node:
                yield from walk(item)
    yield from walk(section_json.get('Record', {}))


def fetch_boiling_point_c(cid):
    data = _pubchem_get(PUBCHEM_VIEW_URL.format(cid=cid, heading='Boiling+Point'))
    if not data:
        return None
    for info in _iter_info_entries(data, 'Boiling Point'):
        value = info.get('Value', {})
        numbers = value.get('Number')
        unit = value.get('Unit', '')
        if numbers and 'C' in unit:
            return float(numbers[0])
    # fall back: parse first free-text string, handle ranges ("84-85 C") and
    # degF -> degC conversion. The minus sign is only treated as a sign when
    # it starts the string/number (not stuck between two digits, which means
    # a "low-high" range, not a negative value).
    for info in _iter_info_entries(data, 'Boiling Point'):
        for s in info.get('Value', {}).get('StringWithMarkup', []):
            text = s.get('String', '')
            for unit, to_celsius in (('F', lambda v: (v - 32) * 5 / 9), ('C', lambda v: v)):
                m = re.search(rf'(-?\d+\.?\d*)\s*-\s*(-?\d+\.?\d*)\s*\xb0\s*{unit}\b', text)
                if m:
                    lo, hi = float(m.group(1)), float(m.group(2))
                    return round(to_celsius((lo + hi) / 2), 1)
                m = re.search(rf'(?:^|\s)(-?\d+\.?\d*)\s*\xb0\s*{unit}\b', text)
                if m:
                    return round(to_celsius(float(m.group(1))), 1)
    return None


def fetch_water_solubility(cid):
    data = _pubchem_get(PUBCHEM_VIEW_URL.format(cid=cid, heading='Solubility'))
    if not data:
        return None
    for info in _iter_info_entries(data, 'Solubility'):
        for s in info.get('Value', {}).get('StringWithMarkup', []):
            text = s.get('String', '').lower()
            for keyword, bucket in SOLUBILITY_KEYWORDS:
                if keyword in text:
                    return bucket
    # fall back: bucket a numeric mg/L value if present
    for info in _iter_info_entries(data, 'Solubility'):
        value = info.get('Value', {})
        numbers = value.get('Number')
        unit = value.get('Unit', '')
        if numbers and 'mg/L' in unit:
            mg_per_l = float(numbers[0])
            if mg_per_l >= 1e5:
                return 'high'
            if mg_per_l >= 1e4:
                return 'moderate'
            if mg_per_l >= 1e3:
                return 'low'
            return 'very low'
    return None


# ============================
# Orchestration
# ============================

def build_property_rows(molecules):
    rows = []
    missing = []
    for mol in molecules:
        name, smiles = mol['iupac_name'], mol['smiles']
        print(f"Processing {name} ({smiles})...")

        rdkit_props = compute_rdkit_properties(smiles)
        avg_ox = compute_avg_carbon_oxidation_state(smiles)

        n_carbons = sum(1 for a in Chem.MolFromSmiles(smiles).GetAtoms() if a.GetSymbol() == 'C')
        assert n_carbons == mol['carbon_count'], (
            f"{name}: SMILES has {n_carbons} carbons, expected carbon_count={mol['carbon_count']}"
        )

        cid = resolve_cid(name) or resolve_cid(mol['common_name'])
        boiling_point_c = fetch_boiling_point_c(cid) if cid else None
        water_solubility = fetch_water_solubility(cid) if cid else None
        if cid is None:
            missing.append((name, 'no PubChem CID found'))
        else:
            if boiling_point_c is None:
                missing.append((name, 'no boiling point in PubChem experimental data'))
            if water_solubility is None:
                missing.append((name, 'no water solubility in PubChem experimental data'))

        group = mol['functional_group']
        if group not in MANUAL_PKA:
            raise KeyError(f"No MANUAL_PKA entry for functional group '{group}'")
        pka, pkah = MANUAL_PKA[group]

        rows.append({
            'iupac_name': mol['iupac_name'],
            'common_name': mol['common_name'],
            'formula': smiles,
            'carbon_count': mol['carbon_count'],
            'functional_group': group,
            'functional_group_structure': mol['functional_group_structure'],
            'mw': rdkit_props['mw'],
            'pka': pka,
            'pkah': pkah,
            'tpsa': rdkit_props['tpsa'],
            'avg_carbon_oxidation_state': avg_ox,
            'hbd': rdkit_props['hbd'],
            'hba': rdkit_props['hba'],
            'boiling_point_c': boiling_point_c,
            'water_solubility': water_solubility,
        })
    return rows, missing


def load_original():
    """Reads the base dataset and drops any rows this script regenerates, so
    re-running the pipeline (ORIGINAL_CSV == OUTPUT_CSV) recomputes those rows
    fresh instead of duplicating them. Matched by iupac_name rather than
    functional_group: some NEW_MOLECULES entries (e.g. the position isomers)
    share a functional_group label with original, non-regenerated rows
    (propan-1-ol and propan-2-ol are both 'alcohol'), so matching on the label
    would incorrectly drop those original rows too.
    """
    df = pd.read_csv(ORIGINAL_CSV)
    df['functional_group'] = df['functional_group'].replace('alkyl halide', 'alkyl chloride')
    df.loc[df['functional_group'] == 'alkyl chloride', 'functional_group_structure'] = '-Cl'
    new_names = {mol['iupac_name'] for mol in NEW_MOLECULES}
    return df[~df['iupac_name'].isin(new_names)]


def verify(df):
    # Merged classes (alcohol/amine/thiol, which fold in the C2 position
    # isomers) legitimately have >1 molecule at the same chain length -- the
    # real invariant is just that no chain length (3-6) is missing entirely,
    # not that every class has exactly 4 rows.
    problems = []
    for group, sub in df.groupby('functional_group'):
        if group == 'none (alkane)':
            continue
        missing = {3, 4, 5, 6} - set(sub['carbon_count'])
        if missing:
            problems.append(f"'{group}' is missing chain length(s) {sorted(missing)}")
    if problems:
        raise AssertionError("Dataset verification failed:\n  " + "\n  ".join(problems))
    print(f"Verified: {len(df)} rows, {df['functional_group'].nunique()} functional-group classes, "
          f"each non-alkane class covers chain lengths 3-6.")


def main():
    original = load_original()

    new_rows, missing = build_property_rows(NEW_MOLECULES)
    new_df = pd.DataFrame(new_rows)

    combined = pd.concat([original, new_df], ignore_index=True)[COLUMNS]
    verify(combined)

    if missing:
        print(f"\n=== NEEDS MANUAL FOLLOW-UP ({len(missing)} gaps) ===")
        for name, reason in missing:
            print(f"  {name}: {reason}")

    na_boiling = combined['boiling_point_c'].isna().sum()
    na_solubility = combined['water_solubility'].isna().sum()
    print(f"\nboiling_point_c missing for {na_boiling}/{len(combined)} rows "
          f"(PubChem had no experimental data for those compounds)")
    print(f"water_solubility missing for {na_solubility}/{len(combined)} rows")

    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(combined)} rows to {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
