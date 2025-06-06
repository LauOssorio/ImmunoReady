import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def final_preproc(df):
    droplist = ["1st in vivo Process - Disease",
            "Epitope - Source Organism",
            "Assay - Qualitative Measure",
            "1st in vivo Process - Process Type",
            "cell_type_in_assay"
            ]

    df = df.drop(columns=droplist)

    df.loc[:,'Epitope - Species'] = df['Epitope - Species'].apply(
        lambda x: 'virus' if isinstance(x, str) and 'virus' in x.lower() else x)
    df.loc[:,'Epitope - Species'] = df['Epitope - Species'].apply(
        lambda x: 'bacterium' if isinstance(x, str) and 'bacterium' in x.lower() else x)
    df.loc[:,'Epitope - Species'] = df['Epitope - Species'].apply(
        lambda w: w.split()[0] if isinstance(w, str) else w)

    classification_map = {
    "virus": ["SARS-CoV1", "virus"],
    "bacterium": ["bacterium", "Streptococcus", "Orientia", "Proteus", "Pseudomonas", "Clostridium", "Helicobacter",
                  "Staphylococcus", "Bacillus", "Treponema", "Chlamydia", "Bordetella", "Leptospira",
                  "Haemophilus", "Neisseria", "Salmonella", "Porphyromonas", "Shigella", "Klebsiella",
                  "Enterococcus", "Ehrlichia", "Escherichia", "Yersinia", "Lactobacillus", "Tropheryma",
                  "Bartonella", "Legionella", "Rickettsia", "Wolbachia", "Listeria", "Moraxella", "Bacteroides",
                  "Stenotrophomonas", "Anaplasma", "Clostridioides", "Borrelia", "Acinetobacter", "Francisella",
                  "Vibrio", "Brucella", "Lactococcus", "Acholeplasma", "Azotobacter", "Finegoldia", "Sulfurovum",
                  "Chlorobium", "Rhodococcus", "Mycoplasmoides", "Malacoplasma", "Spiroplasma", "Synechococcus",
                  "Mycoplasma", "Paenarthrobacter", "Paraburkholderia", "Burkholderia", "Runella", "Blastococcus",
                  "Myxococcus", "Mycobacteroides", "Mycolicibacter", "Phocaeicola", "Microscilla", "Prescottella",
                  "Rhodococcoides", "Enterocloster", "Eisenbergiella", "Psychromonas", "Enterobacter",
                  "Mesorhizobium", "Shewanella", "Meiothermus", "Gluconobacter", "Aequorivita", "Paucilactobacillus",
                  "Kozakia", "Alkaliphilus", "Leucobacter", "Roseibium", "Ligilactobacillus", "Streptomyces",
                  "Xanthomonas", "Paenibacillus", "Ectopseudomonas", "Stutzerimonas", "Sporosarcina", "Olsenella",
                  "Schinkia", "Priestia", "Halalkalibacter", "Robertmurraya", "Alkalihalobacillus", "Oceanobacillus",
                  "Akkermansia", "Acidiphilium", "Actinobaculum", "Neobacillus", "Selenomonas", "Serratia",
                  "Campylobacter", "Novosphingobium", "Alcaligenes", "Pseudoduganella", "Dechloromonas"],
    "human": ["Homo"],
    "eucariotic_parasite": ["Plasmodium", "Trypanosoma", "Entamoeba", "Toxoplasma", "Babesia", "Leishmania",
                            "Fasciola", "Schistosoma", "Echinococcus", "Taenia", "Trichomonas", "Onchocerca",
                            "Anisakis", "Necator", "Brugia", "Wuchereria", "Giardia", "Cryptosporidium",
                            "Opisthorchis", "Ascaris", "Ancylostoma", "Trichuris", "Encephalitozoon"],
    "fungus": ["Aspergillus", "Candida", "Alternaria", "Curvularia", "Penicillium", "Trichophyton", "Rhizopus",
               "Saccharomyces", "Neurospora", "Cladosporium", "Paracoccidioides", "Blastomyces", "Trametes",
               "Cryptococcus", "Sporothrix", "Exophiala", "Beauveria", "Colletotrichum", "Parastagonospora",
               "Rhizophagus", "Mucor", "Zygosaccharomyces", "Scedosporium", "Rasamsonia", "Zymoseptoria",
               "Millerozyma", "Metarhizium", "Cladophialophora", "Rhizoctonia", "Cordyceps", "Aureobasidium",
               "Pyrrhoderma", "Fusarium", "Clavispora", "Rosellinia", "Histoplasma", "Henningerozyma",
               "Gloeophyllum", "Sphaerulina", "Coprinopsis", "Ogataea", "Trichosporon"],
    "plant": ["Sesamum", "Arachis", "Lolium", "Phleum", "Parietaria", "Poa", "Ambrosia", "Cryptomeria", "Myrmecia",
              "Ziziphus", "Juniperus", "Sinapis", "Bertholletia", "Olea", "Oryza", "Cynodon", "Pinus", "Zea",
              "Solanum", "Nicotiana", "Musa", "Mangifera", "Vitis", "Hesperocyparis", "Chamaecyparis", "Triticum",
              "Prunus", "Holcus", "Glycine", "Malus", "Corylus", "Betula", "Cucumis", "Hevea", "Fagopyrum",
              "Anacardium", "Lens", "Carya", "Phaseolus", "Trichosanthes", "Eutrema", "Capsella", "Brassica", "Kali",
              "Chenopodium", "Apium", "Pistacia", "Artemisia", "Daucus", "Zinnia", "Arabidopsis", "Avena", "Alnus",
              "Phalaris", "Quercus", "Dactylis", "Cupressus", "Anthoxanthum", "Fraxinus", "Phoenix", "Plantago",
              "Paspalum", "Fagus", "Castanea", "Hordeum", "Secale", "Sorghum", "Broussonetia"],
    "animal": ["Bos", "Felis", "Gallus", "Pan", "Mus", "Lepidoglyphus", "Gadus", "Dermatophagoides", "Apis", "Turbo",
               "Cavia", "Artemia", "Oryctolagus", "Tetronarce", "Drosophila", "Sus", "Oncorhynchus", "Paralichthys",
               "Danio", "Salmo", "Scylla", "Glossina", "Equus", "Capra", "Macaca", "Bubalus", "Eriocheir",
               "Procambarus", "Pediculus", "Metapenaeus", "Canis", "Magallana", "Merluccius", "Macruronus",
               "Cyprinus", "Scomber", "Palaemon", "Rattus", "Blattella", "Spodoptera", "Mytilus", "Mytilus",
               "Lateolabrax", "Bombyx", "Haliotis", "Oratosquilla", "Perna", "Euphausia"],
    "algae_protist": ["Volvox", "Emiliania", "Pleurocapsa"],
    "archaea": ["Pyrobaculum"],
    "cyanobacteria": ["Nostoc", "Synechocystis"]}

    reverse_d = {val: key for key, values in classification_map.items() for val in values}
    df.loc[:,'Epitope - Species'] = df['Epitope - Species'].map(reverse_d)

    df.loc[:,'1st in vivo Process - Disease Stage'] = df['1st in vivo Process - Disease Stage'].apply(
        lambda x: 'cancer' if isinstance(x, str) and 'cancer' in x.lower() else x)
    df.loc[:,'1st in vivo Process - Disease Stage'] = df['1st in vivo Process - Disease Stage'].apply(
        lambda x: 'cancer' if isinstance(x, str) and 'metastatic' in x.lower() else x)

    df.loc[:,'Assay - Response measured'] = df['Assay - Response measured'].apply(
        lambda x: 'inflamatory molecule' if isinstance(x, str) and 'release' in x.lower() else x)
    df.loc[:,'Assay - Response measured'] = df['Assay - Response measured'].apply(
        lambda x: 'T cell' if isinstance(x, str) and 't cell' in x.lower() else x)

    df = df.drop_duplicates()

    df = df.drop_duplicates(subset="Epitope - Name", keep="first")

    return df


def ohe(df):
    y = df[['Epitope - Name','peptide_safety','target_strength']]
    X = df[['Epitope - Name','Epitope - Species',
            '1st in vivo Process - Process Type','1st in vivo Process - Disease Stage',
            'Assay - Method','Assay - Response measured',
            'cell_type_in_assay']]

    excluded_col = 'Epitope - Name'
    categorical_cols = [col for col in X if col != excluded_col]

    cat_transformer = OneHotEncoder(handle_unknown='ignore',sparse_output=False)

    preproc = ColumnTransformer([
            ('OH encoder', cat_transformer, categorical_cols)
        ],remainder='passthrough')

    X_trans = preproc.fit_transform(X)
    feature_names = preproc.get_feature_names_out()
    feature_names[-1] = 'Epitope - Name'
    X_trans = pd.DataFrame(X_trans,columns=feature_names)

    y_trans = y.copy()

    target_encoder = LabelEncoder().fit(y['peptide_safety'])
    y_trans['peptide_safety'] = target_encoder.transform(y['peptide_safety'])

    fulldata = pd.merge(X_trans,y_trans,'inner','Epitope - Name')

    return fulldata
