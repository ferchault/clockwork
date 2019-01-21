
import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--smiles', type=str, help='', metavar='SMI')
    parser.add_argument('-f', '--filename', type=str, help='Output file', metavar='FILE')
    parser.add_argument('-n', '--number', type=str, help='Number of conformers', metavar='int', default=1)
    args = parser.parse_args()


    if args.smiles is None:
        print("Please provide --smiles SMI")
        quit()

    if args.filename is None:
        print("Please provide --filename output.sdf")
        quit()

    mol = Chem.MolFromSmiles(args.smiles)

    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol,
                               numConfs=args.number,
                               useExpTorsionAnglePrefs=True,
                               useBasicKnowledge=True)

    writer = Chem.SDWriter(args.filename)
    for conf in mol.GetConformers():
        tm = Chem.Mol(mol, False, conf.GetId())
        writer.write(tm)

    return

if __name__ == '__main__':
    main()
