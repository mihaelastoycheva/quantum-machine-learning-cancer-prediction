import json

from Bio import Entrez, SeqIO
import pandas as pd

#Email to NCBI
Entrez.email = "msdesign.business@gmail.com"

# RefSeq ID of gene BRCA1 & BRCA2
refseq_ids = {
    "BRCA1": "NM_007294.4",
    "BRCA2": "NM_000059.4"
}

# Load mutations from JSON files
def load_mutations_from_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


mutations = {
    "BRCA1": load_mutations_from_json("brca1_mutations.json"),
    "BRCA2": load_mutations_from_json("brca2_mutations.json")
}

# Function to download the reference sequence
def fetch_reference_sequence(refseq_id):
    handle = Entrez.efetch(db="nucleotide", id=refseq_id, rettype="fasta", retmode="text")
    record = SeqIO.read(handle, "fasta")
    handle.close()
    return str(record.seq)


# Create the dataset
dataset = []

for gene, refseq_id in refseq_ids.items():
    print(f"⬇️ Downloading the reference sequence for {gene}")
    reference_seq = fetch_reference_sequence(refseq_id)

    for mut_name, mut_info in mutations[gene].items():
        print(f"⚙️ Processing mutation: {mut_name} | {gene}")

        # Take 100 nucleotides around the mutation
        pos = mut_info["pos"]
        start = max(0, pos - 50)
        end = pos + 50

        healthy_seq = reference_seq[start:end]

        # Add healthy sequence
        dataset.append({
            "sequence": healthy_seq,
            "label": 0,
            "gene": gene,
            "mutation": "-"
        })

        # Create mutated sequence
        seq_list = list(healthy_seq)
        local_pos = pos - start

        if mut_info["type"] == "del":
            del seq_list[local_pos:local_pos + mut_info["length"]]
        elif mut_info["type"] == "ins":
            seq_list.insert(local_pos, mut_info["base"])
        elif mut_info["type"] == "sub":
            seq_list[local_pos] = mut_info["base"]

        mutated_seq = ''.join(seq_list)

        dataset.append({
            "sequence": mutated_seq,
            "label": 1,
            "gene": gene,
            "mutation": mut_name
        })

# Save the dataset to CVS
df = pd.DataFrame(dataset)
df.to_csv("classical_dataset.csv", index=False)

print("\n✅ Dataset saved as 'classical_dataset.csv'")