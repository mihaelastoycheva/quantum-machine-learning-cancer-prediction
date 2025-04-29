from Bio import Entrez, SeqIO
import pandas as pd

#Email to NCBI
Entrez.email = "msdesign.business@gmail.com"

# RefSeq ID of gene BRCA1 & BRCA2
refseq_ids = {
    "BRCA1": "NM_007294.4",
    "BRCA2": "NM_000059.4"
}

# Real mutation from ClinVar
mutations = {
    "BRCA1": {
        "c.68_69delAG": {"type": "del", "pos": 68, "length": 2},
        "c.5266dupC": {"type": "ins", "pos": 5266, "base": "C"}
    },
    "BRCA2": {
        "c.5946delT": {"type": "del", "pos": 5946, "length": 1},
        "c.3919delG": {"type": "del", "pos": 3919, "length": 1}
    }
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
    print(f"⬇️ Download the reference sequence for {gene}")
    reference_seq = fetch_reference_sequence(refseq_id)

    for mut_name, mut_info in mutations[gene].items():
        print(f"⚙️ Process the mutation: {mut_name} for {gene}")

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

        mutated_seq = ''.join(seq_list)

        dataset.append({
            "sequence": mutated_seq,
            "label": 1,
            "gene": gene,
            "mutation": mut_name
        })

# Save the dataset in CVS file
df = pd.DataFrame(dataset)
df.to_csv("classical_dataset.csv", index=False)

print("\n✅ Dataset saved as 'classical_dataset.csv'")